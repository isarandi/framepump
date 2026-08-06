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
import ctypes
import itertools
import operator
import threading
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Union

import av
import numpy as np
from av.bitstream import BitStreamFilterContext
from numpy.typing import DTypeLike
import PyNvVideoCodec as nvc

from ._core import build_cfr_source_map
from ._cuda_compat import cuda_ctx_pushed, retain_primary_context
from ._pyav import (
    FrameIndexPyAV,
    PyAVReader,
    UnsupportedCodecError,
    VideoDecodeError,
    _discard_other_streams,
    resolve_source_view,
)
from ._selection import FrameSelection

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
# decoder support.
_NVDEC_CODECS = {
    'h264': nvc.cudaVideoCodec.H264,
    'hevc': nvc.cudaVideoCodec.HEVC,
    'av1': nvc.cudaVideoCodec.AV1,
    'vp9': nvc.cudaVideoCodec.VP9,
    'vp8': nvc.cudaVideoCodec.VP8,
    'mpeg1video': nvc.cudaVideoCodec.MPEG1,
    'mpeg2video': nvc.cudaVideoCodec.MPEG2,
    'mpeg4': nvc.cudaVideoCodec.MPEG4,
    'mjpeg': nvc.cudaVideoCodec.JPEG,
    'vc1': nvc.cudaVideoCodec.VC1,
}

# Codecs whose MP4/MKV packets need conversion to Annex-B start codes for
# the NVDEC parser
_ANNEXB_FILTERS = {'h264': 'h264_mp4toannexb', 'hevc': 'hevc_mp4toannexb'}

# Raw PyNvVideoCodec exceptions, wrapped into VideoDecodeError wherever the
# decoder is driven (unsupported profiles, reconfiguration limits, ...)
_NVC_ERRORS = (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported)


class _PyAVPacketSource:
    """Demux packets with PyAV and present them as NVDEC PacketData.

    PyAV replaces PyNvDemuxer here: its seeking is reliable (PyNvDemuxer's
    Seek() segfaults outright on some driver/library combinations and
    mishandles edit-list files), and its bitstream filters convert AVCC
    packets to the Annex-B form the NVDEC parser expects.
    """

    def __init__(self, video_path: str, codec_name: str) -> None:
        self._container = av.open(video_path)
        self._stream = self._container.streams.video[0]
        _discard_other_streams(self._container, self._stream)
        self._codec_name = codec_name
        self._bsf = self._make_bsf()
        # The NVDEC parser copies the bitstream during Decode(), but hold the
        # most recent buffer anyway so its memory can never be reused early
        self._last_buf = None

    def _make_bsf(self):
        filter_name = _ANNEXB_FILTERS.get(self._codec_name)
        if filter_name is None:
            return None
        return BitStreamFilterContext(filter_name, self._stream)

    def seek(self, pts: int) -> None:
        """Seek to the keyframe at or before ``pts`` (stream time_base units)."""
        self._container.seek(pts, stream=self._stream, backward=True)
        # The filter buffers parameter sets; start it fresh after a jump
        self._bsf = self._make_bsf()

    def packets(self):
        """Yield NVDEC PacketData for the remaining packets.

        Packets without timestamps (raw elementary streams) are fed too —
        skipping them would starve the parser entirely; their PacketData pts
        stays 0 and callers must use positional access for such streams.
        """
        for packet in self._container.demux(self._stream):
            filtered = self._bsf.filter(packet) if self._bsf is not None else (packet,)
            for fp in filtered:
                data = bytes(fp)
                if not data:
                    continue
                buf = ctypes.create_string_buffer(data, len(data))
                self._last_buf = buf
                pd = nvc.PacketData()
                pd.bsl_data = ctypes.addressof(buf)
                pd.bsl = len(data)
                pts = fp.pts if fp.pts is not None else fp.dts
                pd.pts = pts if pts is not None else 0
                yield pd

    def close(self) -> None:
        self._container.close()


# ── Frame index ──────────────────────────────────────────────────────


class _CudaFrameIndex:
    """Frame index for NVDEC access, derived from the shared CPU packet index.

    ``FrameIndexPyAV`` stores display-order PTS as exact Fractions in
    seconds; NVDEC packet feeding and seeking work in raw integer stream
    PTS, so the values are converted back through the stream time base
    (exact — they were produced as integer PTS × time base). The Fraction
    PTS are kept for the CFR source map.
    """

    def __init__(self, video_path: str) -> None:
        reader = PyAVReader(video_path)
        try:
            time_base = reader.time_base
            base = FrameIndexPyAV(video_path, reader=reader)
            # Whether the container supports reliable seeking (raw bitstreams
            # and image-pipe formats do not); reuses the CPU reader's
            # detection and probe. Non-seekable sources decode from the start.
            self.seekable: bool = reader.seekable
        finally:
            reader.close()

        if base.pts_synthesized:
            # Timestampless streams (raw bitstreams): the synthesized PTS
            # never match decoder output, so PTS-based frame location would
            # silently return wrong frames.
            name = video_path if isinstance(video_path, str) else '<file-like>'
            raise RuntimeError(f'No valid packets found in {name}')

        self.frame_pts_frac: list[Fraction] = base.frame_pts
        self.frame_pts: list[int] = [int(p / time_base) for p in base.frame_pts]
        self.safe_seek_pts: list[int] = [int(p / time_base) for p in base.safe_seek_pts]
        self.frame_count: int = base.frame_count


# ── Decode session ───────────────────────────────────────────────────


class _NvDecSession:
    """PyAV packet source + PyNvDecoder for a single decode session.

    Created per iteration/access — not shared across clones or concurrent
    iterations.

    The decoder is bound to the device's *primary* CUDA context, retained for
    the session's lifetime. Without an explicit context, PyNvVideoCodec binds
    to whatever context state the creating thread happens to have, and the
    DLPack export then dies with CUDA_ERROR_INVALID_CONTEXT when a different
    thread (e.g. one holding a torch model) owns the process's CUDA state.
    """

    def __init__(
        self,
        video_path: str,
        gpu: int,
        codec,
        codec_name: str,
        output_color_type,
    ) -> None:
        self._path = video_path
        self._device = None
        self._ctx = None
        self._dec = None
        self._src = None
        self._src = _PyAVPacketSource(video_path, codec_name)
        try:
            self._device, self._ctx = retain_primary_context(gpu)
            self._dec = nvc.CreateDecoder(
                gpuid=gpu,
                codec=codec,
                cudacontext=int(self._ctx),
                cudastream=0,
                usedevicememory=True,
                outputColorType=output_color_type,
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )
        except _NVC_ERRORS as e:
            self.close()
            raise VideoDecodeError(video_path, 0, e) from e
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Drop the decoder and balance the primary-context retain."""
        if self._dec is not None:
            with cuda_ctx_pushed(self._ctx):
                self._dec = None
        if self._device is not None:
            from cuda.bindings import driver

            driver.cuDevicePrimaryCtxRelease(self._device)
            self._device = None
            self._ctx = None
        if self._src is not None:
            self._src.close()
            self._src = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _decode_all(self):
        """Feed every remaining packet and flush; yield (pts, frame).

        NVDEC/driver errors (unsupported profile, mid-stream reconfiguration
        beyond limits, ...) are wrapped so callers never see raw
        PyNvVideoCodec exceptions. Decode calls run under the session context
        (pushed per step, never across a yield, so no context state leaks
        into the consuming thread).
        """
        count = 0
        try:
            for pd in self._src.packets():
                with cuda_ctx_pushed(self._ctx):
                    decoded = [(f.getPTS(), f) for f in self._dec.Decode(pd)]
                for item in decoded:
                    count += 1
                    yield item
            empty = nvc.PacketData()
            while True:
                with cuda_ctx_pushed(self._ctx):
                    decoded = [(f.getPTS(), f) for f in self._dec.Decode(empty)]
                if not decoded:
                    break
                for item in decoded:
                    count += 1
                    yield item
        except _NVC_ERRORS as e:
            raise VideoDecodeError(self._path, count, e) from e

    def iter_from_start(self):
        """Decode all frames sequentially from the beginning.

        Yields (pts, frame) tuples in display order.
        """
        yield from self._decode_all()

    def iter_from_pts(self, start_pts: int):
        """Seek to start_pts and decode forward.

        Yields (pts, frame) tuples in display order, starting from the first
        frame with PTS >= start_pts.
        """
        self._src.seek(start_pts)
        reached = False
        for pts, f in self._decode_all():
            if not reached:
                if pts >= start_pts:
                    reached = True
                else:
                    continue
            yield pts, f


# ── Public class ─────────────────────────────────────────────────────


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
    output dtypes are supported like on the CPU class. Remaining API gaps vs
    ``VideoFrames`` (intentional): ``constant_framerate``, file-like sources,
    ``float64``, and the seek-reliability content probe. Use the CPU class
    when those are needed.

    Decode sessions are bound to the device's primary CUDA context, so
    iteration and DLPack export work from any thread (e.g. prefetch
    threads), including processes where another thread owns torch's CUDA
    state.

    Args:
        video_path: Path to video file.
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
        self.path = str(video_path)
        self._gpu = gpu
        self._npp_init_lock = threading.Lock()

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
        reader = PyAVReader(self.path)
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
            self._init_npp_pipeline()

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

    def _iter_streamed(self, streamable: slice):
        """Decode from the start and skip, without building the frame index."""
        if self._needs_stages:
            self._init_npp_pipeline()
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
        return (
            f"VideoFramesCuda('{self.path}', {w}x{h}, {self.fps:.4g} fps, {length}, {self.dtype})"
        )

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
                    self._lazy.index = _CudaFrameIndex(resolve_source_view(self.path))
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
        self._cleanup_npp()

    def __enter__(self) -> VideoFramesCuda:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self._cleanup_npp()

    # ── Internal: clone ──────────────────────────────────────────────

    def _clone(self) -> VideoFramesCuda:
        result = VideoFramesCuda.__new__(VideoFramesCuda)
        result.path = self.path
        result._gpu = self._gpu
        result._npp_init_lock = threading.Lock()
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
        """Create a new decode session with the configured color type."""
        return _NvDecSession(
            self.path,
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
        return self._copy_rgb_frame(frame, session) if owned else _CtxFrame(frame, session)

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

    def _copy_rgb_frame(self, frame, session) -> _GpuRgbBuffer:
        """Copy a decoder-owned RGB uint8 frame into an owned GPU buffer."""
        from cuda.bindings import driver

        h, w = self.original_imshape

        err, device = driver.cuDeviceGet(self._gpu)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDeviceGet({self._gpu}) failed: {err}')
        err, ctx = driver.cuDevicePrimaryCtxRetain(device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDevicePrimaryCtxRetain failed: {err}')
        try:
            with cuda_ctx_pushed(ctx):
                # The frame's CUDA-array export needs the session's context
                # current (same primary context as ours).
                views = frame.cuda()
                view = views[0] if isinstance(views, (list, tuple)) else views
                cai = view.__cuda_array_interface__
                src_ptr = cai['data'][0]
                strides = cai.get('strides')
                src_pitch = strides[0] if strides else w * 3
                row_bytes = w * 3
                if src_pitch < row_bytes:
                    raise RuntimeError(f'Unexpected RGB frame pitch {src_pitch} for width {w}')
                err, devptr = driver.cuMemAlloc(row_bytes * h)
                if err != driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f'Failed to allocate frame copy buffer: {err}')
                devptr = int(devptr)
                try:
                    copy = driver.CUDA_MEMCPY2D()
                    copy.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
                    copy.srcDevice = src_ptr
                    copy.srcPitch = src_pitch
                    copy.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
                    copy.dstDevice = devptr
                    copy.dstPitch = row_bytes
                    copy.WidthInBytes = row_bytes
                    copy.Height = h
                    (err,) = driver.cuMemcpy2D(copy)
                    if err != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f'Failed to copy RGB frame: {err}')
                except BaseException:
                    driver.cuMemFree(devptr)
                    raise
        finally:
            driver.cuDevicePrimaryCtxRelease(device)

        return _GpuRgbBuffer(devptr, h, w, row_bytes, self._gpu, owns_memory=True, bits=8)

    # ── NPP pipeline ─────────────────────────────────────────────────

    @property
    def _stage_names(self) -> list[str]:
        """The post-decode GPU stages, in execution order."""
        if self._out_shape is not None and self._gamma_resize:
            # Linear-light resize: to float, linearize, resize, re-encode.
            names = [] if self._npp_mode is None else ['convert']
            names += ['to_f32', 'lin_resize']
            if self._float_dtype is None:
                names.append('to_uint')
            elif self._float_dtype == np.dtype(np.float16):
                names.append('f16')
            return names
        if self._npp_mode is None:
            names = ['resize8'] if self._out_shape is not None else []
        else:
            names = ['convert']
            if self._out_shape is not None:
                names.append('resize16')
        if self._float_dtype is not None:
            names.append('f32')
            if self._float_dtype == np.dtype(np.float16):
                names.append('f16')
        return names

    @property
    def _final_format(self) -> tuple[int, int]:
        """(bits, DLPack type code) of the output samples."""
        if self._float_dtype is not None:
            return (16, 2) if self._float_dtype == np.dtype(np.float16) else (32, 2)
        return (8, 1) if self.dtype == np.dtype(np.uint8) else (16, 1)

    def _stage_buffer_size(self, name: str) -> int:
        h, w = self.original_imshape
        th, tw = self.imshape
        final_elem = self._final_format[0] // 8
        return {
            'convert': w * 3 * 2 * h,
            'resize8': tw * 3 * th,
            'resize16': tw * 3 * 2 * th,
            'f32': tw * 3 * 4 * th,
            'f16': tw * 3 * 2 * th,
            'to_f32': w * 3 * 4 * h,
            'lin_resize': tw * 3 * 4 * th,
            'to_uint': tw * 3 * final_elem * th,
        }[name]

    def _init_npp_pipeline(self) -> None:
        """Lazy-initialize NPP post-processing resources (idempotent, thread-safe)."""
        if hasattr(self, '_npp_ctx'):
            return
        with self._npp_init_lock:
            if hasattr(self, '_npp_ctx'):
                return

            from . import npp_bindings
            from cuda.bindings import driver

            # Initialize CUDA driver API (no-op if already initialized).
            driver.cuInit(0)

            # Retain the primary context for this GPU. It is made current only
            # for the duration of the pipeline's own calls, never left current.
            err, device = driver.cuDeviceGet(self._gpu)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'cuDeviceGet({self._gpu}) failed: {err}')
            err, cuda_ctx = driver.cuDevicePrimaryCtxRetain(device)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'cuDevicePrimaryCtxRetain failed: {err}')

            bufs: dict[str, int] = {}
            try:
                with cuda_ctx_pushed(cuda_ctx):
                    # Build NppStreamContext (uses default stream = 0).
                    npp_ctx = npp_bindings.make_npp_stream_context(self._gpu)

                    # Select color twist matrix.
                    twist = npp_bindings.make_yuv_to_rgb_twist(
                        self._color_space, full_range=self._range_full
                    )

                    # One reusable (iteration-shared) buffer per stage.
                    for name in self._stage_names:
                        err, devptr = driver.cuMemAlloc(self._stage_buffer_size(name))
                        if err != driver.CUresult.CUDA_SUCCESS:
                            raise RuntimeError(f'Failed to allocate NPP {name} buffer: {err}')
                        bufs[name] = int(devptr)
            except Exception:
                with cuda_ctx_pushed(cuda_ctx):
                    for ptr in bufs.values():
                        driver.cuMemFree(ptr)
                driver.cuDevicePrimaryCtxRelease(device)
                raise

            # Publish only on full success; the sentinel (_npp_ctx) goes last
            # so the unlocked fast path never sees a partial pipeline.
            self._cuda_device = device
            self._npp_cuda_ctx = cuda_ctx
            self._npp_bindings = npp_bindings
            self._twist = twist
            self._bufs = bufs
            self._npp_ctx = npp_ctx

    def _cleanup_npp(self) -> None:
        """Free NPP pipeline GPU resources."""
        ctx = getattr(self, '_npp_cuda_ctx', None)
        bufs = getattr(self, '_bufs', None)
        if bufs is not None and ctx is not None:
            from cuda.bindings import driver

            with cuda_ctx_pushed(ctx):
                # NPP work on the default stream may still be in flight.
                driver.cuCtxSynchronize()
                for ptr in bufs.values():
                    driver.cuMemFree(ptr)
            del self._bufs
        device = getattr(self, '_cuda_device', None)
        if device is not None:
            from cuda.bindings import driver

            driver.cuDevicePrimaryCtxRelease(device)
            del self._cuda_device
        if ctx is not None:
            del self._npp_cuda_ctx
        # Remove the lazy-init sentinel (and everything published with it) so
        # the pipeline re-initializes on the next use instead of hitting
        # AttributeError on the freed resources.
        for attr in ('_npp_ctx', '_npp_bindings', '_twist'):
            if hasattr(self, attr):
                delattr(self, attr)

    def _run_stages(self, frame, *, fresh: bool) -> _GpuRgbBuffer:
        """Run the post-decode GPU stages (color, resize, dtype) on one frame.

        Shared mode (``fresh=False``) writes into the pipeline's reusable
        buffers — the result is valid until the next iteration step. Fresh
        mode writes the final stage into a new allocation, synchronizes, and
        returns an owning buffer.
        """
        from cuda.bindings import driver

        self._init_npp_pipeline()
        npp = self._npp_bindings
        h, w = self.original_imshape
        th, tw = self.imshape
        fbits, fcode = self._final_format
        final_pitch = tw * 3 * (fbits // 8)
        stages = self._stage_names

        if fresh:
            with cuda_ctx_pushed(self._npp_cuda_ctx):
                err, final_ptr = driver.cuMemAlloc(final_pitch * th)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to allocate frame buffer: {err}')
            final_ptr = int(final_ptr)
        else:
            final_ptr = self._bufs[stages[-1]]

        try:
            with cuda_ctx_pushed(self._npp_cuda_ctx):
                cur: tuple[int, int] | None = None  # (ptr, pitch)
                for i, name in enumerate(stages):
                    dst = final_ptr if (fresh and i == len(stages) - 1) else self._bufs[name]
                    if name == 'convert':
                        self._do_convert(frame, dst)
                        cur = (dst, w * 3 * 2)
                    elif name == 'resize8':
                        ((src_ptr, src_pitch),) = _plane_layouts(frame, 1, w * 3, h)
                        npp.resize_rgb(
                            src_ptr, src_pitch, w, h, dst, tw * 3, tw, th,
                            bits=8, ctx=self._npp_ctx,
                        )  # fmt: skip
                        cur = (dst, tw * 3)
                    elif name == 'resize16':
                        npp.resize_rgb(
                            cur[0], cur[1], w, h, dst, tw * 3 * 2, tw, th,
                            bits=16, ctx=self._npp_ctx,
                        )  # fmt: skip
                        cur = (dst, tw * 3 * 2)
                    elif name == 'f32':
                        npp.rgb16_to_float01(
                            cur[0], cur[1], dst, tw * 3 * 4, tw, th, ctx=self._npp_ctx
                        )
                        cur = (dst, tw * 3 * 4)
                    elif name == 'to_f32':
                        if self._npp_mode is None:
                            ((src_ptr, src_pitch),) = _plane_layouts(frame, 1, w * 3, h)
                            npp.rgb8_to_float01(
                                src_ptr, src_pitch, dst, w * 3 * 4, w, h, ctx=self._npp_ctx
                            )
                        else:
                            npp.rgb16_to_float01(
                                cur[0], cur[1], dst, w * 3 * 4, w, h, ctx=self._npp_ctx
                            )
                        npp.srgb_curve_inplace(dst, w * h * 3, decode=True)
                        cur = (dst, w * 3 * 4)
                    elif name == 'lin_resize':
                        npp.resize_rgb(
                            cur[0], cur[1], w, h, dst, tw * 3 * 4, tw, th,
                            bits=32, ctx=self._npp_ctx,
                        )  # fmt: skip
                        # Re-encode; the kernel clamps to [0, 1] first, which
                        # also absorbs Lanczos over/undershoot from the resize.
                        npp.srgb_curve_inplace(dst, tw * th * 3, decode=False)
                        cur = (dst, tw * 3 * 4)
                    elif name == 'to_uint':
                        bits = self._final_format[0]
                        pitch = tw * 3 * (bits // 8)
                        npp.float01_to_uint(
                            cur[0], cur[1], dst, pitch, tw, th, bits=bits, ctx=self._npp_ctx
                        )
                        cur = (dst, pitch)
                    else:  # 'f16'
                        npp.float32_to_float16(
                            cur[0], cur[1], dst, tw * 3 * 2, tw, th, ctx=self._npp_ctx
                        )
                        cur = (dst, tw * 3 * 2)
                if fresh:
                    # The NPP kernels run async on the default stream and read
                    # decoder-owned frame memory; sync before the caller drops
                    # its frame/session references.
                    (err,) = driver.cuStreamSynchronize(0)
                    if err != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f'cuStreamSynchronize failed: {err}')
        except BaseException:
            if fresh:
                with cuda_ctx_pushed(self._npp_cuda_ctx):
                    driver.cuMemFree(final_ptr)
            raise

        return _GpuRgbBuffer(
            final_ptr,
            th,
            tw,
            final_pitch,
            self._gpu,
            owns_memory=fresh,
            bits=fbits,
            code=fcode,
        )

    def _convert_frame_shared(self, frame) -> _GpuRgbBuffer:
        """Post-process a frame into the reusable iteration buffers."""
        return self._run_stages(frame, fresh=False)

    def _convert_frame_fresh(self, frame) -> _GpuRgbBuffer:
        """Post-process a frame into a freshly allocated owning buffer."""
        return self._run_stages(frame, fresh=True)

    def _do_convert(self, frame, dst_ptr: int) -> None:
        """Dispatch NPP conversion based on mode and source format."""
        h, w = self.original_imshape
        dst_pitch = w * 3 * 2
        npp = self._npp_bindings
        ctx = self._npp_ctx

        with cuda_ctx_pushed(self._npp_cuda_ctx):
            if self._npp_mode == 'yuv_to_rgb16':
                pf = self._source_format
                if pf == nvc.Pixel_Format.P016:
                    (y_ptr, y_pitch), (uv_ptr, uv_pitch) = _plane_layouts(frame, 2, w * 2, h)
                    npp.p016_to_rgb16(
                        y_ptr,
                        y_pitch,
                        uv_ptr,
                        uv_pitch,
                        dst_ptr,
                        dst_pitch,
                        w,
                        h,
                        self._twist,
                        ctx,
                    )
                elif pf == nvc.Pixel_Format.YUV444_16Bit:
                    planes = _plane_layouts(frame, 3, w * 2, h)
                    (y_ptr, y_pitch), (u_ptr, u_pitch), (v_ptr, v_pitch) = planes
                    if not (y_pitch == u_pitch == v_pitch):
                        raise RuntimeError(
                            f'YUV444 conversion requires equal plane pitches, got '
                            f'{y_pitch}/{u_pitch}/{v_pitch}'
                        )
                    npp.yuv444_16bit_to_rgb16(
                        y_ptr,
                        u_ptr,
                        v_ptr,
                        y_pitch,
                        dst_ptr,
                        dst_pitch,
                        w,
                        h,
                        self._twist,
                        ctx,
                    )
                else:
                    raise RuntimeError(f'Unsupported source format for yuv_to_rgb16: {pf}')

            elif self._npp_mode == 'scale_8u_16u':
                ((src_ptr, src_pitch),) = _plane_layouts(frame, 1, w * 3, h)
                npp.rgb8_to_rgb16(
                    src_ptr,
                    src_pitch,
                    dst_ptr,
                    dst_pitch,
                    w,
                    h,
                    ctx,
                )

            else:
                raise RuntimeError(f'Unknown _npp_mode: {self._npp_mode!r}')


def _plane_layouts(frame, num_planes: int, min_row_bytes: int, height: int) -> list:
    """Get (device pointer, row pitch in bytes) for each plane of a decoded frame.

    Prefers the per-plane CUDA-array-interface views from ``frame.cuda()``.
    PyNvVideoCodec's views report strides in elements in current releases
    (a corrected release would report bytes); both units are accepted and the
    result is validated against the row's minimum byte width, so a wrong
    pitch raises instead of producing silently sheared output.

    Falls back to inferring the pitch from plane-pointer deltas when the
    views are unavailable — validated under the explicit assumption that the
    planes are allocated contiguously at a common pitch.
    """
    try:
        views = frame.cuda()
        cais = [view.__cuda_array_interface__ for view in views[:num_planes]]
        if len(cais) != num_planes:
            raise ValueError(f'Expected {num_planes} planes, got {len(cais)}')
    except Exception:
        return _plane_layouts_from_deltas(frame, num_planes, min_row_bytes, height)

    result = []
    for cai in cais:
        ptr = cai['data'][0]
        itemsize = int(cai['typestr'][-1])
        strides = cai.get('strides')
        if strides:
            s0 = strides[0]
            if itemsize == 1 or s0 < min_row_bytes:
                pitch = s0 if s0 >= min_row_bytes else s0 * itemsize
            else:
                # Ambiguous window (small widths): s0 could be a byte pitch,
                # or an element count that still exceeds the row byte width.
                # Corroborate with the plane-pointer delta; raise rather than
                # guess, per this function's contract.
                pitch = _disambiguate_pitch(frame, num_planes, s0, itemsize, height)
        else:
            pitch = min_row_bytes
        if pitch < min_row_bytes or pitch % itemsize:
            raise RuntimeError(
                f'Cannot determine plane pitch: stride {strides} (itemsize {itemsize}) '
                f'is inconsistent with a row width of {min_row_bytes} bytes'
            )
        result.append((ptr, pitch))
    return result


def _disambiguate_pitch(frame, num_planes: int, s0: int, itemsize: int, height: int) -> int:
    """Decide whether a reported stride is a byte pitch or an element count.

    Uses the luma→chroma plane-pointer delta (pitch * plane height for the
    contiguous allocations NVDEC produces) as the tiebreaker.
    """
    candidates = {s0, s0 * itemsize}
    if len(candidates) == 1:
        return s0
    if num_planes >= 2:
        try:
            delta = frame.GetPtrToPlane(1) - frame.GetPtrToPlane(0)
        except Exception:
            delta = None
        if delta is not None:
            matching = [c for c in sorted(candidates) if delta == c * height]
            if len(matching) == 1:
                return matching[0]
    raise RuntimeError(
        f'Cannot determine whether the reported stride {s0} (itemsize {itemsize}) '
        f'is a byte pitch or an element count; refusing to guess.'
    )


def _plane_layouts_from_deltas(frame, num_planes: int, min_row_bytes: int, height: int) -> list:
    """Infer a common plane pitch from consecutive plane-pointer deltas."""
    ptrs = [frame.GetPtrToPlane(i) for i in range(num_planes)]
    if num_planes == 1:
        return [(ptrs[0], min_row_bytes)]
    delta = ptrs[1] - ptrs[0]
    if delta <= 0 or delta % height:
        raise RuntimeError(
            f'Cannot infer plane pitch: plane delta {delta} is not a positive '
            f'multiple of the plane height {height} (planes not contiguous?)'
        )
    pitch = delta // height
    if pitch < min_row_bytes or pitch % 2:
        raise RuntimeError(
            f'Inferred plane pitch {pitch} is inconsistent with a row width of '
            f'{min_row_bytes} bytes'
        )
    return [(ptr, pitch) for ptr in ptrs]


# ── GPU RGB buffer with DLPack export ────────────────────────────────


class _GpuRgbBuffer:
    """DLPack-compatible wrapper around a GPU-resident packed uint16 RGB buffer.

    For iteration: ``owns_memory=False``, the buffer is shared and reused.
    For indexing: ``owns_memory=True``, freed when the consumer releases it.

    An owning buffer retains the primary context of its device so the free —
    which may run on any thread, at any time, via ``__del__`` or the DLPack
    deleter — always has a valid context to run under, even after the parent
    ``VideoFramesCuda`` released its own retain.
    """

    __slots__ = (
        '_devptr',
        '_height',
        '_width',
        '_pitch',
        '_gpu_id',
        '_owns_memory',
        '_own_device',
        '_own_ctx',
        '_shape_arr',
        '_strides_arr',
        '_bits',
        '_code',
    )

    def __init__(
        self,
        devptr: int,
        height: int,
        width: int,
        pitch: int,
        gpu_id: int,
        *,
        owns_memory: bool,
        bits: int = 16,
        code: int = 1,
    ) -> None:
        self._devptr = devptr
        self._height = height
        self._width = width
        self._pitch = pitch
        self._gpu_id = gpu_id
        self._owns_memory = owns_memory
        self._bits = bits
        self._code = code  # DLPack type code: 1 = kDLUInt, 2 = kDLFloat
        self._own_device = None
        self._own_ctx = None
        if owns_memory:
            from cuda.bindings import driver

            err, device = driver.cuDeviceGet(gpu_id)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'cuDeviceGet({gpu_id}) failed: {err}')
            err, ctx = driver.cuDevicePrimaryCtxRetain(device)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'cuDevicePrimaryCtxRetain failed: {err}')
            self._own_device = device
            self._own_ctx = ctx
        # Must outlive any DLPack capsule (DLTensor holds raw pointers).
        self._shape_arr = (ctypes.c_int64 * 3)(height, width, 3)
        self._strides_arr = (ctypes.c_int64 * 3)(width * 3, 3, 1)

    def __dlpack__(self, *args, **kwargs):
        if self._owns_memory is False and self._devptr == 0:
            raise RuntimeError(
                'This buffer already handed its memory to a previous __dlpack__ '
                'export; it cannot be exported again.'
            )
        mt = _DLManagedTensor()
        mt.dl_tensor.data = self._devptr
        mt.dl_tensor.device = _DLDevice(2, self._gpu_id)  # kDLCUDA
        mt.dl_tensor.ndim = 3
        mt.dl_tensor.dtype = _DLDataType(self._code, self._bits, 1)
        mt.dl_tensor.shape = ctypes.cast(self._shape_arr, ctypes.POINTER(ctypes.c_int64))
        mt.dl_tensor.strides = ctypes.cast(self._strides_arr, ctypes.POINTER(ctypes.c_int64))
        mt.dl_tensor.byte_offset = 0

        key = next(_prevent_gc_counter)

        if self._owns_memory:
            # Hand the allocation and its primary-context retain over to the
            # DLPack consumer's deleter.
            devptr = self._devptr
            device, ctx = self._own_device, self._own_ctx
            self._devptr = 0
            self._owns_memory = False
            self._own_device = None
            self._own_ctx = None
            mt.deleter = _GPU_BUFFER_FREE_DELETER
        else:
            devptr = 0  # sentinel: don't free
            device, ctx = None, None
            mt.deleter = _GPU_BUFFER_NOFREE_DELETER

        mt.manager_ctx = key
        _prevent_gc_store[key] = (
            devptr,
            mt,
            self._shape_arr,
            self._strides_arr,
            device,
            ctx,
        )

        return _PyCapsule_New(ctypes.addressof(mt), b'dltensor', None)

    def __dlpack_device__(self):
        return (2, self._gpu_id)  # kDLCUDA

    def view(self) -> '_GpuRgbBuffer':
        """A non-owning alias of this buffer (used for frame repetition).

        Views must be consumed before the owner's memory is handed over or
        freed; the repeat wrapper yields views first, the original last.
        """
        return _GpuRgbBuffer(
            self._devptr,
            self._height,
            self._width,
            self._pitch,
            self._gpu_id,
            owns_memory=False,
            bits=self._bits,
            code=self._code,
        )

    def __del__(self):
        if self._owns_memory and self._devptr:
            _free_owned_buffer(self._devptr, self._own_device, self._own_ctx)


class _CtxFrame:
    """Proxy that binds a decoder-owned frame's exports to the session context.

    PyNvVideoCodec's exports (``__dlpack__``, ``cuda()``) require the decode
    session's CUDA context to be current on the calling thread; consumers may
    iterate from any thread (prefetch patterns), so the proxy pushes the
    session's context around the export and forwards everything else to the
    wrapped frame. Holding the session also keeps the decoder — owner of the
    frame's GPU surface pool — alive.
    """

    __slots__ = ('_frame', '_session')

    def __init__(self, frame, session):
        self._frame = frame
        self._session = session

    def __dlpack__(self, *args, **kwargs):
        with cuda_ctx_pushed(self._session._ctx):
            return self._frame.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._frame.__dlpack_device__()

    def cuda(self):
        with cuda_ctx_pushed(self._session._ctx):
            return self._frame.cuda()

    def __getattr__(self, name):
        return getattr(self._frame, name)


class _FrameWithDecoder:
    """DLPack-compatible wrapper that prevents the decode session from GC.

    When ``VideoFramesCuda[i]`` returns a frame, the underlying session
    must stay alive (its decoder owns the GPU surface pool). This wrapper
    holds references to both and produces a DLPack capsule whose deleter
    prevents GC until the consumer is done with the data. Like ``_CtxFrame``,
    exports run under the session's CUDA context.
    """

    __slots__ = ('_frame', '_decoder')

    def __init__(self, frame, decoder):
        self._frame = frame
        self._decoder = decoder

    def __dlpack__(self, *args, **kwargs):
        with cuda_ctx_pushed(self._decoder._ctx):
            capsule = self._frame.__dlpack__(*args, **kwargs)
        return _dlpack_prevent_gc(capsule, self._decoder, self._frame)

    def __dlpack_device__(self):
        return self._frame.__dlpack_device__()

    def cuda(self):
        with cuda_ctx_pushed(self._decoder._ctx):
            return self._frame.cuda()

    def __getattr__(self, name):
        return getattr(self._frame, name)


# ── DLPack prevent-GC wrapping ────────────────────────────────────────
#
# PyNvVideoCodec's DLPack deleter does NOT prevent the decoder from freeing
# its GPU surface pool. We wrap the capsule in a new DLManagedTensor whose
# deleter holds Python references (decoder, frame) alive until the consumer
# (e.g., torch) is done with the data.

# DLPack ABI structs (stable since DLPack 0.2, 2017)


class _DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ('code', ctypes.c_uint8),
        ('bits', ctypes.c_uint8),
        ('lanes', ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),
        ('device', _DLDevice),
        ('ndim', ctypes.c_int32),
        ('dtype', _DLDataType),
        ('shape', ctypes.POINTER(ctypes.c_int64)),
        ('strides', ctypes.POINTER(ctypes.c_int64)),
        ('byte_offset', ctypes.c_uint64),
    ]


_DeleterFunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ('dl_tensor', _DLTensor),
        ('manager_ctx', ctypes.c_void_p),
        ('deleter', _DeleterFunc),
    ]


# PyCapsule C API

_py = ctypes.pythonapi

_PyCapsule_GetPointer = _py.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_PyCapsule_New = _py.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_PyCapsule_SetName = _py.PyCapsule_SetName
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

# Prevent-GC store: maps integer keys to tuples of Python objects that must
# stay alive until the DLPack consumer calls the deleter.
_prevent_gc_store: dict[int, tuple] = {}
# Keys start at 1: key 0 stored into the c_void_p manager_ctx field becomes
# NULL and reads back as None, so the deleter's pop() would never find it and
# the entry (decoder session or GPU buffer) would be pinned forever.
_prevent_gc_counter = itertools.count(1)


# ── _GpuRgbBuffer deleters (module-level to survive GC) ──────────────


def _free_owned_buffer(devptr, device, ctx) -> None:
    """Free an owned allocation under its owning primary context.

    May run on any thread (GC, or a DLPack consumer's deleter), so the
    owning context is pushed for the free and the retain released after.
    Failures are reported as warnings — deleters cannot raise.
    """
    from cuda.bindings import driver

    if ctx is not None:
        (err,) = driver.cuCtxPushCurrent(ctx)
        if err != driver.CUresult.CUDA_SUCCESS:
            warnings.warn(f'Failed to push context to free GPU buffer: {err}')
            return
    try:
        (err,) = driver.cuMemFree(devptr)
        if err != driver.CUresult.CUDA_SUCCESS:
            warnings.warn(f'Failed to free GPU frame buffer: {err}')
    finally:
        if ctx is not None:
            driver.cuCtxPopCurrent()
            driver.cuDevicePrimaryCtxRelease(device)


def _gpu_buffer_free_deleter_impl(managed_ptr):
    """DLPack deleter for owned GPU buffers: frees the allocation."""
    mt = _DLManagedTensor.from_address(managed_ptr)
    entry = _prevent_gc_store.pop(mt.manager_ctx, None)
    if entry is not None and entry[0]:
        _free_owned_buffer(entry[0], entry[4], entry[5])


def _gpu_buffer_nofree_deleter_impl(managed_ptr):
    """DLPack deleter for shared (iteration) buffers: no-op on GPU memory."""
    mt = _DLManagedTensor.from_address(managed_ptr)
    _prevent_gc_store.pop(mt.manager_ctx, None)


_GPU_BUFFER_FREE_DELETER = _DeleterFunc(_gpu_buffer_free_deleter_impl)
_GPU_BUFFER_NOFREE_DELETER = _DeleterFunc(_gpu_buffer_nofree_deleter_impl)


# ── _FrameWithDecoder deleter ────────────────────────────────────────


def _prevent_gc_deleter_impl(managed_ptr):
    """Called by the DLPack consumer when the tensor is freed."""
    mt = _DLManagedTensor.from_address(managed_ptr)
    ctx = _prevent_gc_store.pop(mt.manager_ctx, None)
    if ctx is None:
        return

    # ctx layout: (orig_mt_ptr, capsule, wrapper_mt, deleter_ref, ...)
    orig_mt_ptr = ctx[0]

    # Call original deleter (frees the original DLManagedTensor struct).
    # Read the raw function pointer to safely handle NULL.
    deleter_voidp = ctypes.c_void_p.from_address(
        orig_mt_ptr + _DLManagedTensor.deleter.offset
    ).value
    if deleter_voidp:
        _DeleterFunc(deleter_voidp)(orig_mt_ptr)

    # ctx is dropped here, releasing decoder, frame, capsule, wrapper struct.


# Must be module-level to prevent GC of the callback itself.
_PREVENT_GC_DELETER = _DeleterFunc(_prevent_gc_deleter_impl)


def _dlpack_prevent_gc(capsule, *prevent_gc_refs):
    """Wrap a DLPack capsule so that *prevent_gc_refs* stay alive.

    Returns a new ``dltensor`` PyCapsule backed by the same GPU data.
    The original capsule is marked consumed. When the consumer eventually
    frees the tensor, our deleter releases all prevent-GC references
    (typically the decoder and frame) and calls the original deleter.
    """
    orig_ptr = _PyCapsule_GetPointer(capsule, b'dltensor')
    orig_mt = _DLManagedTensor.from_address(orig_ptr)

    # Mark original capsule as consumed (prevents its destructor from
    # calling the original deleter — we'll call it ourselves).
    _PyCapsule_SetName(capsule, b'used_dltensor')

    # Create our wrapper DLManagedTensor with the same dl_tensor
    # (shallow copy — data/shape/strides pointers stay the same).
    wrapper = _DLManagedTensor()
    ctypes.memmove(
        ctypes.addressof(wrapper.dl_tensor),
        ctypes.addressof(orig_mt.dl_tensor),
        ctypes.sizeof(_DLTensor),
    )
    wrapper.deleter = _PREVENT_GC_DELETER

    # Store everything that must stay alive:
    #   capsule     — keeps original DLManagedTensor struct (shape/strides) alive
    #   wrapper     — keeps our ctypes struct alive
    #   _PREVENT_GC_DELETER — prevents callback from being collected
    #   prevent_gc_refs — decoder, frame, etc.
    key = next(_prevent_gc_counter)
    _prevent_gc_store[key] = (orig_ptr, capsule, wrapper, _PREVENT_GC_DELETER, *prevent_gc_refs)
    wrapper.manager_ctx = key

    return _PyCapsule_New(ctypes.addressof(wrapper), b'dltensor', None)
