"""GPU-resident video frame reader using NVDEC via PyNvVideoCodec low-level API.

Demuxes with PyAV (reliable seeking, Annex-B conversion via bitstream
filters) and decodes with PyNvDecoder, for precise PTS-based seeking and
accurate frame counts — even for videos with edit lists, B-frame reordering,
or unreliable container metadata.

Provides the same lazy, sliceable interface as VideoFrames but decodes on GPU
and yields DLPack-compatible frames. Use ``torch.from_dlpack(frame)`` to get
a CUDA tensor with zero-copy.

**Buffer lifetime:** Indexing (``frames[i]``) is safe — the returned object's
DLPack capsule prevents the decoder from being garbage-collected until the
consumer (e.g., torch) frees the tensor::

    t = torch.from_dlpack(frames[42])   # safe, even as a one-liner

Iteration yields into a shared buffer (NPP path) or raw decoder frames. The
decoder stays alive for the loop, but individual buffers may be reused across
batches. Clone if you need to keep frames beyond the current loop body::

    for frame in frames:
        t = torch.from_dlpack(frame).clone()  # safe to keep

Reverse iteration (``frames[::-1]``) yields owned GPU buffers instead (they
are buffered internally per chunk), so keeping those without cloning is safe.

The frame index (a packet scan of the file) is built lazily: forward
iteration and prefix-style slicing stream without it; ``len()``, integer
indexing, negative bounds and reverse iteration build it on first use.

Example:
    >>> import numpy as np
    >>> frames = VideoFramesCuda('video.mp4')
    >>> for frame in frames[::2][:100]:
    ...     t = torch.from_dlpack(frame).clone()
    ...     model(t.permute(2, 0, 1).float() / 255)
    >>> single = torch.from_dlpack(frames[42])

    >>> # High bit depth (10-bit) preservation:
    >>> frames = VideoFramesCuda('10bit.mp4', dtype=np.uint16)
    >>> for frame in frames:
    ...     t = torch.from_dlpack(frame)  # (H, W, 3) uint16 on CUDA
"""

from __future__ import annotations

import bisect
import operator
import threading
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import DTypeLike
import PyNvVideoCodec as nvc

from .._core import build_cfr_source_map
from .compat import cuda_ctx_pushed
from .._pyav import (
    PyAVReader,
    UnsupportedCodecError,
    VideoDecodeError,
    resolve_source_view,
)
from .._selection import FrameSelection
from .decode import _NVDEC_CODECS, _CudaFrameIndex, _NvDecSession
from .post import _PostProcessor, copy_rgb_frame
from .dlpack import _CtxFrame, _FrameWithDecoder, _GpuRgbBuffer

PathLike = Union[str, Path]

# Source pixel formats that carry >8 bits of precision.
_HBD_FORMATS = frozenset(
    {
        nvc.Pixel_Format.P016,
        nvc.Pixel_Format.YUV444_16Bit,
    }
)


# Streaming (decode-from-start-and-skip, no index) is used for forward
# selections whose start is at most this many frames (mirrors VideoFrames)
_STREAM_MAX_SKIP = 256

# Reverse iteration buffers one chunk of copied/converted GPU frames at a
# time; the chunk frame count is derived from this budget ([4, 64] frames)
_REVERSE_CHUNK_BYTES = 256 * 1024 * 1024


# AVColorSpace id → twist matrix name (H.273 ids as in FFmpeg's pixfmt.h).
# Values must be keys of npp_bindings.LUMA_COEFFICIENTS (imported lazily
# there — importing it here would load the NPP shared libraries eagerly).
_AVCOL_SPC_TO_MATRIX = {
    1: 'bt709',
    4: 'fcc',
    5: 'bt601',  # BT.470BG
    6: 'bt601',  # SMPTE 170M
    7: 'smpte240m',
    9: 'bt2020',  # non-constant luminance
}
_SUPPORTED_COLOR_SPACES = frozenset(_AVCOL_SPC_TO_MATRIX.values())



class _CudaLazyIndexState:
    """Frame-index state shared between a VideoFramesCuda and all its views."""

    __slots__ = ('lock', 'index', 'cfr_source_map')

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index: _CudaFrameIndex | None = None
        self.cfr_source_map: list[int] | None = None


# PyAV codec names -> NVDEC codec ids. Only these have NVDEC hardware
class VideoFramesCuda:
    """Lazy, sliceable GPU video frame iterator using NVDEC (low-level API).

    Frames are decoded on GPU and stay on GPU: iteration and indexing yield
    DLPack-compatible objects, so ``torch.from_dlpack(frame)`` gives a CUDA
    tensor with no copy. That GPU residency of the *output* is the difference
    from ``VideoFrames(gpu=True)``, which also decodes with NVDEC but
    downloads every frame to a numpy array.

    Output is not bit-identical to CPU decoding (``VideoFrames``): the
    YUV->RGB conversion runs in CUDA kernels rather than FFmpeg's swscale,
    so pixel values typically differ by a few counts. ``VideoFrames``
    with ``gpu=True`` is bit-identical to CPU decoding instead.

    ``resized()`` (GPU-side NPP resize), ``repeat_each_frame()``, and float
    output dtypes, ``constant_framerate`` and file-like sources are supported
    like on the CPU class. Remaining API gaps vs ``VideoFrames``
    (intentional): ``float64`` and the seek-reliability content probe. Use
    the CPU class when those are needed.

    Decode sessions are bound to the device's primary CUDA context, so
    iteration and DLPack export work from any thread (e.g. prefetch
    threads), including processes where another thread owns torch's CUDA
    state.

    Args:
        video_path: Path to video file (str or Path), or a seekable file-like
            object (must support read, seek, tell). ``BytesIO`` sources
            support any number of concurrently active iterators (each decode
            session gets an independent view); other file-like objects allow
            only one active iterator at a time, since sessions share the
            object's read position.
        gpu: GPU device ordinal (default 0).
        dtype: Output dtype — ``np.uint8`` (default), ``np.uint16``,
            ``np.float16``, or ``np.float32``. For 10-bit sources, ``uint16``
            preserves the full precision via an NVDEC → NPP color-conversion
            pipeline. For 8-bit sources, ``uint16`` scales values to the full
            0–65535 range. Float outputs are scaled to [0, 1] on the GPU
            (uint16 pipeline internally, like the CPU class).
        constant_framerate: False for VFR (native timestamps), True for CFR
            at the original fps, or a number for CFR at that specific fps.
            Uses the same ffmpeg-parity source map as the CPU class, so the
            two classes select identical source frames.
        color_space: ``'auto'`` (default), ``'bt601'``, ``'bt709'``,
            ``'bt2020'``, ``'fcc'``, or ``'smpte240m'``. Only used when NPP
            conversion is active (10-bit + uint16). ``'auto'`` follows the
            stream's colorspace flag; if unspecified, BT.709 is assumed for
            height >= 720, else BT.601. Limited vs full range is always taken
            from the stream. (BT.2020 constant-luminance, ICtCp and other
            non-matrix colorspaces are not supported — use ``VideoFrames``.)
    """

    def __init__(
        self,
        video_path: PathLike,
        *,
        gpu: int = 0,
        dtype: DTypeLike = np.uint8,
        color_space: str = 'auto',
        constant_framerate: bool | float = False,
    ) -> None:
        self._is_fileobj = hasattr(video_path, 'read')
        self.path = video_path if self._is_fileobj else str(video_path)
        self._gpu = gpu
        self._npp_init_lock = threading.Lock()
        self._post: _PostProcessor | None = None

        # Validate dtype.
        dtype = np.dtype(dtype)
        if dtype in (np.dtype(np.float16), np.dtype(np.float32)):
            self._float_dtype: np.dtype | None = dtype
        elif dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
            self._float_dtype = None
        elif dtype == np.dtype(np.float64):
            raise NotImplementedError(
                'dtype=float64 is not supported for GPU decoding (NPP has no '
                'float64 pipeline). Use float32, or convert after torch.from_dlpack().'
            )
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        self.dtype: np.dtype = dtype

        # Read container metadata via PyAV (no packet scan); this also gives
        # the same clean errors as the CPU class for audio-only files and
        # codecs without a decoder
        reader = PyAVReader(resolve_source_view(self.path))
        try:
            width, height = reader.resolution
            self.original_imshape: tuple[int, int] = (height, width)
            self.original_fps: float = float(reader.fps)
            self._codec_name: str = reader.codec_name
            pix_fmt = reader._stream.codec_context.format
            fmt_name = pix_fmt.name if pix_fmt is not None else 'yuv420p'
            cc = reader._stream.codec_context
            self._colorspace_id = int(getattr(cc, 'colorspace', 0) or 0)
            self._range_full = int(getattr(cc, 'color_range', 0) or 0) == 2  # AVCOL_RANGE_JPEG
            self._trc_id = int(getattr(cc, 'color_trc', 2) or 2)
        finally:
            reader.close()

        if self._codec_name not in _NVDEC_CODECS:
            raise UnsupportedCodecError(self.path)
        self._codec = _NVDEC_CODECS[self._codec_name]
        # Bit depth and chroma subsampling from the pixel format name
        # (e.g. 'yuv420p10le', 'yuv444p')
        self._bit_depth: int = 10 if '10' in fmt_name else 12 if '12' in fmt_name else 8
        self._chroma_is_444: bool = '444' in fmt_name

        # The frame index (full packet scan) is built lazily on first
        # length-dependent or seek-based access; forward streaming access
        # never needs it. Shared with all views cloned from this instance.
        self._lazy = _CudaLazyIndexState()
        self._selection = FrameSelection.identity()
        self._dims_checked = False

        # Probe source pixel format (needs one decoded NATIVE frame).
        self._source_format = self._probe_source_format()

        # Decide decode + post-processing strategy. Float outputs use the
        # uint16 pipeline internally (like the CPU class) and scale to [0, 1]
        # as a final NPP stage.
        source_is_hbd = self._source_format in _HBD_FORMATS
        want_16 = dtype != np.dtype(np.uint8)

        if not want_16:
            # uint8 output: library's RGB conversion handles everything
            # (truncates 10-bit to 8-bit automatically).
            self._npp_mode: str | None = None
            self._color_type = nvc.OutputColorType.RGB
        elif source_is_hbd:
            # 10-bit source + uint16 → decode NATIVE, NPP YUV→RGB16.
            self._npp_mode = 'yuv_to_rgb16'
            self._color_type = nvc.OutputColorType.NATIVE
        else:
            # 8-bit source + uint16 → decode RGB (uint8), NPP upscale.
            self._npp_mode = 'scale_8u_16u'
            self._color_type = nvc.OutputColorType.RGB

        self._out_shape: tuple[int, int] | None = None
        self._gamma_resize: bool = False
        self._repeat_count: int = 1

        # Parse constant_framerate: False, True, or a number (target fps)
        if constant_framerate is False:
            self.constant_framerate = False
            self.target_fps = self.original_fps
        elif constant_framerate is True:
            self.constant_framerate = True
            self.target_fps = self.original_fps
        else:
            self.constant_framerate = True
            self.target_fps = float(constant_framerate)

        # Color space (only matters for yuv_to_rgb16 path).
        if color_space == 'auto':
            mapped = _AVCOL_SPC_TO_MATRIX.get(self._colorspace_id)
            if mapped is not None:
                self._color_space = mapped
            else:
                # Unspecified — fall back to height heuristic.
                self._color_space = 'bt709' if self.original_imshape[0] >= 720 else 'bt601'
        elif color_space in _SUPPORTED_COLOR_SPACES:
            self._color_space = color_space
        else:
            raise ValueError(
                f'color_space must be one of {("auto", *sorted(_SUPPORTED_COLOR_SPACES))}, '
                f'got {color_space!r}'
            )

    # ── Public interface ──────────────────────────────────────────────

    def __iter__(self):
        """Decode and yield the selected frames on the GPU, in order.

        Each yielded frame stays in GPU memory and supports ``__dlpack__``
        for zero-copy import into PyTorch, CuPy, etc.
        """
        inner = self._iter_once()
        if self._repeat_count == 1:
            yield from inner
            return
        for obj in inner:
            # Non-owning views first, the original object last: an owned
            # buffer's DLPack export hands its memory over, which must not
            # happen while sibling views are still queued.
            for _ in range(self._repeat_count - 1):
                yield obj.view() if isinstance(obj, _GpuRgbBuffer) else obj
            yield obj

    def _iter_once(self):
        """Decode and yield each selected frame exactly once."""
        # Stream without the index when the selection is a plain forward
        # slice with a small start. If the index already exists, the
        # seek-based paths are at least as good — use them. CFR always needs
        # the index (the source map derives from it).
        if not self._selection.is_resolved and not self.constant_framerate:
            streamable = self._selection.streamable_slice
            if (
                streamable is not None
                and (streamable.start or 0) <= _STREAM_MAX_SKIP
                and self._lazy.index is None
            ):
                yield from self._iter_streamed(streamable)
                return

        frame_range = self._resolved_range()
        if len(frame_range) == 0:
            return

        if self._needs_stages:
            self._post_processor()

        if self.constant_framerate:
            sources = [self._cfr_source_map[i] for i in frame_range]
            if frame_range.step > 0:
                # Nondecreasing source sequence: one forward decode pass.
                yield from self._iter_sources_forward(sources)
            else:
                yield from self._iter_by_index(sources)
            return

        # Large step (either direction): per-frame seeks; range() iterates
        # backward natively for negative steps, and each frame gets its own
        # decoder, so no chunk buffering is needed.
        if abs(frame_range.step) > 30:
            yield from self._iter_by_index(frame_range)
        elif frame_range.step < 0:
            yield from self._iter_reversed(frame_range)
        elif frame_range.start > 0:
            yield from self._iter_with_seek(frame_range)
        else:
            yield from self._iter_sequential(frame_range)

    @property
    def _needs_stages(self) -> bool:
        """Whether output goes through the NPP post-processing stages."""
        return self._npp_mode is not None or self._out_shape is not None

    def _post_processor(self) -> _PostProcessor:
        """The lazily created post-processing pipeline for this instance."""
        if self._post is None:
            with self._npp_init_lock:
                if self._post is None:
                    self._post = _PostProcessor(
                        gpu=self._gpu,
                        dtype=self.dtype,
                        npp_mode=self._npp_mode,
                        source_format=self._source_format,
                        color_space=self._color_space,
                        range_full=self._range_full,
                        float_dtype=self._float_dtype,
                        out_shape=self._out_shape,
                        gamma_resize=self._gamma_resize,
                        original_imshape=self.original_imshape,
                    )
        return self._post

    def _iter_streamed(self, streamable: slice):
        """Decode from the start and skip, without building the frame index."""
        if self._needs_stages:
            self._post_processor()
        session = self._make_session()
        convert = self._needs_stages
        start = streamable.start or 0
        stop = streamable.stop
        step = streamable.step or 1

        frame_count = 0
        yielded = 0
        for _pts, frame in session.iter_from_start():
            if stop is not None and frame_count >= stop:
                break
            if frame_count >= start and (frame_count - start) % step == 0:
                self._check_decoded_dims(frame, session)
                if convert:
                    yield self._convert_frame_shared(frame)
                else:
                    yield _CtxFrame(frame, session)
                yielded += 1
            frame_count += 1
        if yielded == 0 and frame_count == 0 and (stop is None or stop > 0) and start == 0:
            # The decoder produced nothing for a nonempty request — silence
            # here would look like a valid empty video
            raise VideoDecodeError(self.path, 0, RuntimeError('NVDEC decoder produced no frames'))

    def _check_decoded_dims(self, frame, session) -> None:
        """Detect decoders returning unexpected dimensions (e.g. NVDEC
        splitting interlaced MJPEG into half-height fields) instead of
        silently yielding wrong-shaped frames. Checked once per instance;
        only the RGB path is checked (the NPP path validates plane layouts).
        """
        if self._dims_checked or self._npp_mode is not None:
            return
        self._dims_checked = True
        with cuda_ctx_pushed(session._ctx):
            views = frame.cuda()
            view = views[0] if isinstance(views, (list, tuple)) else views
            shape = view.__cuda_array_interface__['shape']
        expected = self.original_imshape
        if tuple(shape[:2]) != expected:
            raise VideoDecodeError(
                self.path,
                0,
                RuntimeError(
                    f'NVDEC decoded frames of size {shape[1]}x{shape[0]} but the '
                    f'container reports {expected[1]}x{expected[0]} (interlaced '
                    f'field decoding?). Use the CPU VideoFrames class for this file.'
                ),
            )

    def __getitem__(self, item):
        """Access a single frame by index or create a sliced lazy view.

        Args:
            item: Frame index (negative indices count from the end) or slice.

        Returns:
            A GPU-resident frame supporting ``__dlpack__`` for an integer
            index, or a new lazy VideoFramesCuda view for a slice.
        """
        if isinstance(item, int):
            length = len(self)
            if item < 0:
                item = length + item
            if item < 0 or item >= length:
                total = self._n_frames_total()
                if len(self._resolved_range()) != total or self._repeat_count != 1:
                    detail = f'view with {length} frames (source video has {total})'
                else:
                    detail = f'video with {length} frames'
                raise IndexError(f'Frame index {item} out of range for {detail}')
            abs_idx = self._resolved_range()[item // self._repeat_count]
            return self._get_frame_by_abs_idx(self._source_index(abs_idx), owns_memory=True)

        if isinstance(item, slice):
            if item.step == 0:
                raise ValueError('Slice step cannot be zero.')
            if self._repeat_count != 1:
                raise NotImplementedError(
                    'Slicing a repeated view is not supported; apply slicing '
                    'before repeat_each_frame().'
                )
            result = self._clone()
            result._selection = self._selection.sliced(item)
            return result

        raise TypeError('Indices must be integers or slices.')

    def resized(self, shape: tuple[int, int], *, gamma_correct: bool = False) -> VideoFramesCuda:
        """Return a new VideoFramesCuda that outputs frames at the given size.

        The resize runs on the GPU (NPP: area averaging for downscaling,
        Lanczos otherwise) on the converted RGB frames, so its interpolation
        does not bit-match the CPU class's swscale resize.

        Args:
            shape: Target size as (height, width), following numpy/image
                convention. The frame is stretched to exactly this size;
                aspect ratio is not preserved.
            gamma_correct: Resample in linear light instead of on the
                gamma-encoded values: pixels are linearized with the exact
                IEC 61966-2-1 sRGB transfer (piecewise: linear toe below
                0.04045, power 2.4 segment above — not a plain power law),
                resized in float32, and re-encoded. This avoids the darkening
                of averaged high-contrast detail that gamma-space resizing
                (the default here, and what swscale/OpenCV/PIL do) produces —
                most visible when downscaling fine patterns. Off by default
                to match the CPU class and common ML-pipeline conventions.
                Not supported for PQ/HLG (HDR) transfer content.
        """
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(x, int) for x in shape)
        ):
            raise TypeError(f'shape must be a (height, width) tuple of two ints, got {shape!r}')
        if gamma_correct and self._trc_id in (16, 18):  # PQ, HLG
            raise NotImplementedError(
                'gamma_correct resizing is not supported for PQ/HLG (HDR) transfer '
                'content — a pure power law would mishandle it.'
            )
        result = self._clone()
        result._out_shape = shape
        result._gamma_resize = gamma_correct
        return result

    def repeat_each_frame(self, n: int) -> VideoFramesCuda:
        """Return a new VideoFramesCuda that yields each selected frame ``n`` times.

        The effective ``fps`` scales by ``n`` accordingly. Apply slicing
        before this, not after: slicing a repeated view raises
        NotImplementedError.

        Args:
            n: Repeat count, at least 1.
        """
        try:
            n = operator.index(n)
        except TypeError:
            raise TypeError(
                f'The repeat count must be an integer, got {type(n).__name__}'
            ) from None
        if n < 1:
            raise ValueError('The repeat count must be at least 1.')
        result = self._clone()
        result._repeat_count *= n
        return result

    def __len__(self) -> int:
        """Exact number of frames in this view.

        Builds the frame index on first use, which scans the file's packets.
        """
        return len(self._resolved_range()) * self._repeat_count

    def __repr__(self) -> str:
        h, w = self.imshape
        # Never trigger the index scan just for a repr
        length = f'{len(self)} frames' if self._selection.is_resolved else 'lazy'
        name = self.path if isinstance(self.path, str) else '<file-like>'
        return f"VideoFramesCuda('{name}', {w}x{h}, {self.fps:.4g} fps, {length}, {self.dtype})"

    @property
    def imshape(self) -> tuple[int, int]:
        """Output frame dimensions as (height, width), after any resize."""
        return self._out_shape if self._out_shape is not None else self.original_imshape

    @property
    def fps(self) -> float:
        """Effective frame rate, accounting for slicing and repetition.

        Uses the selection's effective stride, which is known even before
        the frame count is — reading fps never triggers the index scan.
        """
        return self.target_fps / abs(self._selection.step_product) * self._repeat_count

    def _n_frames_total(self) -> int:
        if self.constant_framerate:
            return len(self._cfr_source_map)
        return self._index.frame_count

    def _source_index(self, abs_idx: int) -> int:
        """Map an absolute output index to a source frame index (CFR-aware)."""
        return self._cfr_source_map[abs_idx] if self.constant_framerate else abs_idx

    @property
    def _cfr_source_map(self) -> list[int]:
        index = self._index  # resolve first: builds under the same lock
        if self._lazy.cfr_source_map is None:
            with self._lazy.lock:
                if self._lazy.cfr_source_map is None:
                    self._lazy.cfr_source_map = build_cfr_source_map(
                        index.frame_pts_frac, self.target_fps
                    )
        return self._lazy.cfr_source_map

    @property
    def _index(self) -> _CudaFrameIndex:
        if self._lazy.index is None:
            with self._lazy.lock:
                if self._lazy.index is None:
                    self._lazy.index = _CudaFrameIndex(
                        resolve_source_view(self.path), codec_name=self._codec_name
                    )
        return self._lazy.index

    def _resolved_range(self) -> range:
        """The concrete frame-index range, resolving the selection if needed."""
        if not self._selection.is_resolved:
            self._selection = self._selection.resolve(self._n_frames_total())
        return self._selection.range

    def close(self) -> None:
        """Release GPU resources allocated for the NPP pipeline.

        The reader remains usable afterwards: the pipeline is re-created
        lazily on the next iteration or indexed access.
        """
        self._close_post()

    def __enter__(self) -> VideoFramesCuda:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self._close_post()

    def _close_post(self) -> None:
        if self._post is not None:
            self._post.close()
            self._post = None

    # ── Internal: clone ──────────────────────────────────────────────

    def _clone(self) -> VideoFramesCuda:
        result = VideoFramesCuda.__new__(VideoFramesCuda)
        result.path = self.path
        result._is_fileobj = self._is_fileobj
        result._gpu = self._gpu
        result._npp_init_lock = threading.Lock()
        result._post = None
        result.dtype = self.dtype
        result._source_format = self._source_format
        result._npp_mode = self._npp_mode
        result._color_type = self._color_type
        result._color_space = self._color_space
        result.original_imshape = self.original_imshape
        result.original_fps = self.original_fps
        result._codec = self._codec
        result._codec_name = self._codec_name
        result._bit_depth = self._bit_depth
        result._chroma_is_444 = self._chroma_is_444
        result._colorspace_id = self._colorspace_id
        result._range_full = self._range_full
        result._float_dtype = self._float_dtype
        result._out_shape = self._out_shape
        result._gamma_resize = self._gamma_resize
        result._trc_id = self._trc_id
        result._repeat_count = self._repeat_count
        result.constant_framerate = self.constant_framerate
        result.target_fps = self.target_fps
        result._selection = self._selection
        result._dims_checked = self._dims_checked
        # Index state is shared: whichever view builds it, all views see it
        result._lazy = self._lazy
        # NPP pipeline state is NOT shared — each clone initializes lazily.
        return result

    # ── Internal: format probing ─────────────────────────────────────

    def _probe_source_format(self) -> nvc.Pixel_Format:
        """Infer the NVDEC native pixel format from demuxer metadata.

        Avoids creating a decoder session just for format probing.
        """
        hbd = self._bit_depth > 8
        if self._chroma_is_444:
            return nvc.Pixel_Format.YUV444_16Bit if hbd else nvc.Pixel_Format.YUV444
        # 420 (and 422, which NVDEC outputs as 420)
        return nvc.Pixel_Format.P016 if hbd else nvc.Pixel_Format.NV12

    # ── Internal: session creation ───────────────────────────────────

    def _make_session(self) -> _NvDecSession:
        """Create a new decode session with the configured color type.

        Each session opens its own view of the source (for ``BytesIO``
        sources this costs one in-memory copy per session; per-frame random
        access creates a session per seek).
        """
        return _NvDecSession(
            resolve_source_view(self.path),
            self._gpu,
            self._codec,
            self._codec_name,
            self._color_type,
        )

    # ── Internal: random access ──────────────────────────────────────

    def _get_frame_by_abs_idx(self, abs_idx: int, *, owns_memory: bool):
        """Get a single frame by absolute index."""
        target_pts = self._index.frame_pts[abs_idx]
        safe_pts = self._index.safe_seek_pts[abs_idx]
        frame, session = self._seek_decode_to(safe_pts, target_pts)
        return self._wrap_frame(frame, session, owns_memory=owns_memory)

    def _seek_decode_to(self, safe_pts: int, target_pts: int):
        """Seek to safe_pts, decode forward to target_pts.

        Returns (frame, session) — caller must keep the session alive while
        using the frame's GPU memory.

        If the file is non-seekable (edit-list files), decodes from the
        beginning and skips to the target PTS.
        """
        # A fresh session per seek for clean decoder state.
        session = self._make_session()
        if self._index.seekable and safe_pts > 0:
            session._src.seek(safe_pts)
        for pts, frame in session._decode_all():
            if pts >= target_pts:
                self._check_decoded_dims(frame, session)
                return frame, session
        raise RuntimeError(f'Failed to decode frame at PTS {target_pts} (seeked to {safe_pts})')

    def _wrap_frame(self, frame, session, *, owns_memory: bool):
        """Wrap a decoded frame for output (NPP stages or DLPack)."""
        if self._needs_stages:
            if owns_memory:
                buf = self._convert_frame_fresh(frame)
                del session
                return buf
            else:
                return self._convert_frame_shared(frame)
        # No NPP: wrap frame + session to prevent GC.
        if owns_memory:
            return _FrameWithDecoder(frame, session)
        else:
            return _CtxFrame(frame, session)

    # ── Iteration paths ──────────────────────────────────────────────

    def _emit(self, frame, owned: bool, session):
        """Convert or wrap a decoded frame for yielding.

        ``owned=False`` follows the iteration contract (shared conversion
        buffer / decoder frame, valid until the next step). ``owned=True``
        produces a buffer independent of the decode session — required when
        frames are buffered past subsequent decodes (reverse chunks), since
        decoder-owned surfaces are recycled from a bounded pool.
        """
        if self._needs_stages:
            return self._convert_frame_fresh(frame) if owned else self._convert_frame_shared(frame)
        return (
            copy_rgb_frame(frame, session, self.original_imshape, self._gpu)
            if owned
            else _CtxFrame(frame, session)
        )

    def _iter_sequential(self, frame_range: range, *, owned: bool = False):
        """Path C: sequential decode from beginning with step."""
        session = self._make_session()
        start = frame_range.start
        stop = frame_range.stop
        step = frame_range.step

        frame_count = 0
        yielded = 0
        for _pts, frame in session.iter_from_start():
            if frame_count >= stop:
                break
            if frame_count >= start and (frame_count - start) % step == 0:
                self._check_decoded_dims(frame, session)
                yield self._emit(frame, owned, session)
                yielded += 1
            frame_count += 1
        if yielded == 0 and frame_count == 0 and stop > start:
            raise VideoDecodeError(self.path, 0, RuntimeError('NVDEC decoder produced no frames'))

    def _iter_with_seek(self, frame_range: range, *, owned: bool = False):
        """Path B: seek to start, then sequential with step."""
        start = frame_range.start
        stop = frame_range.stop
        step = frame_range.step

        target_pts = self._index.frame_pts[start]
        safe_pts = self._index.safe_seek_pts[start]

        session = _NvDecSession(
            self.path,
            self._gpu,
            self._codec,
            self._codec_name,
            self._color_type,
        )

        if self._index.seekable:
            frame_iter = session.iter_from_pts(safe_pts)
        else:
            frame_iter = session.iter_from_start()

        frame_count = 0
        max_frames = stop - start
        for _pts, frame in frame_iter:
            if _pts < target_pts:
                continue
            if frame_count >= max_frames:
                break
            if frame_count % step == 0:
                yield self._emit(frame, owned, session)
            frame_count += 1

    def _iter_sources_forward(self, sources: list[int]):
        """Decode a nondecreasing source-index sequence in one forward pass.

        Used for forward CFR iteration: the source map may repeat a source
        frame (frame doubling) or skip some (frame dropping). Duplicates are
        yielded as views of the frame emitted for their first occurrence,
        without re-decoding.
        """
        if not sources:
            return
        safe_pts = self._index.safe_seek_pts[sources[0]]
        session = self._make_session()
        if self._index.seekable and safe_pts > 0:
            frame_iter = session.iter_from_pts(safe_pts)
        else:
            frame_iter = session.iter_from_start()

        pos = 0
        cur_src = None
        for pts, frame in frame_iter:
            if cur_src is None:
                cur_src = bisect.bisect_left(self._index.frame_pts, pts)
            else:
                cur_src += 1
            emitted = None
            while pos < len(sources) and sources[pos] == cur_src:
                if emitted is None:
                    self._check_decoded_dims(frame, session)
                    emitted = self._emit(frame, False, session)
                    yield emitted
                else:
                    yield emitted.view() if isinstance(emitted, _GpuRgbBuffer) else emitted
                pos += 1
            if pos >= len(sources):
                return
        raise VideoDecodeError(
            self.path,
            pos,
            RuntimeError('NVDEC decoder ended before delivering the selected frames'),
        )

    def _iter_by_index(self, indices):
        """Path A: individual seeks per source index (large step / CFR reverse)."""
        convert = self._needs_stages
        # Keep the previous session alive between yields — its decoder owns
        # the GPU surface pool that the frame points to.
        prev_session = None
        for abs_idx in indices:
            target_pts = self._index.frame_pts[abs_idx]
            safe_pts = self._index.safe_seek_pts[abs_idx]
            frame, session = self._seek_decode_to(safe_pts, target_pts)
            if convert:
                yield self._convert_frame_shared(frame)
            else:
                yield _CtxFrame(frame, session)
            # Assign after yield so the previous session survives while the
            # caller uses the frame
            prev_session = session
        del prev_session

    # ── Reverse iteration ────────────────────────────────────────────

    def _iter_reversed(self, frame_range: range):
        """Iterate a negative-step range via backward chunks decoded forward.

        Chunk frames must outlive the chunk's remaining decodes, so each
        selected frame is copied (uint8) or freshly converted (uint16) into
        an owned GPU buffer before the chunk is yielded in reverse —
        decoder-owned surfaces cannot be buffered, since the decoder recycles
        its bounded surface pool as decoding continues.
        """
        fwd = frame_range[::-1]
        min_chunk, max_chunk, fallback_chunk = self._reverse_chunk_bounds()

        pos = len(fwd)
        while pos > 0:
            lo = self._pick_reverse_chunk_start(fwd, pos, min_chunk, max_chunk, fallback_chunk)
            chunk = fwd[lo:pos]
            if chunk.start > 0:
                buf = list(self._iter_with_seek(chunk, owned=True))
            else:
                buf = list(self._iter_sequential(chunk, owned=True))
            yield from reversed(buf)
            pos = lo

    def _reverse_chunk_bounds(self) -> tuple[int, int, int]:
        """(min, max, fallback) chunk lengths, bounded by the GPU byte budget."""
        h, w = self.imshape
        frame_bytes = max(h * w * 3 * self.dtype.itemsize, 1)
        max_chunk = max(4, min(64, _REVERSE_CHUNK_BYTES // frame_bytes))
        min_chunk = max(1, max_chunk // 2)
        return min_chunk, max_chunk, (min_chunk + max_chunk) // 2

    def _pick_reverse_chunk_start(
        self, fwd: range, hi_pos: int, min_chunk: int, max_chunk: int, fallback_chunk: int
    ) -> int:
        """Position in ``fwd`` where the next (backward) chunk should start.

        Prefers a selected frame that is its own safe seek point within
        [hi_pos - max_chunk, hi_pos - min_chunk], so the chunk's seek lands
        exactly where decoding must begin.
        """
        lo_limit = max(hi_pos - max_chunk, 0)
        for p in range(max(hi_pos - min_chunk, 0), lo_limit - 1, -1):
            if p == 0 or self._is_safe_seek_frame(fwd[p]):
                return p
        return max(hi_pos - fallback_chunk, 0)

    def _is_safe_seek_frame(self, abs_idx: int) -> bool:
        return self._index.frame_pts[abs_idx] == self._index.safe_seek_pts[abs_idx]

    def _convert_frame_shared(self, frame) -> _GpuRgbBuffer:
        """Post-process a frame into the reusable iteration buffers."""
        return self._post_processor().process(frame, fresh=False)

    def _convert_frame_fresh(self, frame) -> _GpuRgbBuffer:
        """Post-process a frame into a freshly allocated owning buffer."""
        return self._post_processor().process(frame, fresh=True)
