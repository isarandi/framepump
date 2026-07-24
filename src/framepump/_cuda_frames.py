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
import threading
import warnings
from pathlib import Path
from typing import Union

import av
import numpy as np
from av.bitstream import BitStreamFilterContext
from numpy.typing import DTypeLike
import PyNvVideoCodec as nvc

from ._cuda_compat import cuda_ctx_pushed
from ._pyav import (
    PyAVReader,
    UnsupportedCodecError,
    VideoDecodeError,
    _discard_other_streams,
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


class _CudaLazyIndexState:
    """Frame-index state shared between a VideoFramesCuda and all its views."""

    __slots__ = ('lock', 'index')

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index: _FrameIndexNvDec | None = None


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


class _FrameIndexNvDec:
    """Packet-based frame index built from PyAV demuxing.

    Same algorithm as FrameIndexPyAV._build_from_packets() — collects
    file-order PTS, builds running_max_at array, computes safe seek points
    via bisect.  PTS values are raw integers in the stream's time_base,
    matching the values fed to the NVDEC decoder through PacketData.
    """

    def __init__(self, video_path: str) -> None:
        # Collect PTS in file (packet) order via PyAV.
        file_order_pts: list[int] = []
        running_max_at: list[int] = []
        running_max = -1

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for pkt in container.demux(stream):
                pts = pkt.pts if pkt.pts is not None else pkt.dts
                if pts is None or pts < 0:
                    continue

                file_order_pts.append(pts)
                running_max = max(running_max, pts)
                running_max_at.append(running_max)

        if not file_order_pts:
            raise RuntimeError(f'No valid packets found in {video_path}')

        # Display-order PTS: sorted, deduplicated.
        self.frame_pts: list[int] = sorted(set(file_order_pts))
        self.frame_count: int = len(self.frame_pts)

        # Safe seek points: for each target PTS, find the last packet in file
        # order whose running_max <= target.  This ensures all reference frames
        # needed to decode the target have been seen.
        self.safe_seek_pts: list[int] = []
        for target in self.frame_pts:
            idx = bisect.bisect_right(running_max_at, target) - 1
            if idx >= 0:
                self.safe_seek_pts.append(file_order_pts[idx])
            else:
                self.safe_seek_pts.append(min(file_order_pts[0], 0))

        # Whether the container supports reliable seeking (raw bitstreams and
        # image-pipe formats do not); reuses the CPU reader's detection and
        # probe. Non-seekable sources decode from the start instead.
        reader = PyAVReader(video_path)
        try:
            self.seekable: bool = reader.seekable
        finally:
            reader.close()


# ── Decode session ───────────────────────────────────────────────────


class _NvDecSession:
    """PyAV packet source + PyNvDecoder for a single decode session.

    Created per iteration/access — not shared across clones or concurrent
    iterations.
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
        self._src = _PyAVPacketSource(video_path, codec_name)
        try:
            self._dec = nvc.CreateDecoder(
                gpuid=gpu,
                codec=codec,
                usedevicememory=True,
                outputColorType=output_color_type,
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )
        except _NVC_ERRORS as e:
            raise VideoDecodeError(video_path, 0, e) from e

    def _decode_all(self):
        """Feed every remaining packet and flush; yield (pts, frame).

        NVDEC/driver errors (unsupported profile, mid-stream reconfiguration
        beyond limits, ...) are wrapped so callers never see raw
        PyNvVideoCodec exceptions.
        """
        count = 0
        try:
            for pd in self._src.packets():
                for f in self._dec.Decode(pd):
                    count += 1
                    yield f.getPTS(), f
            empty = nvc.PacketData()
            while True:
                frames = self._dec.Decode(empty)
                if not frames:
                    break
                for f in frames:
                    count += 1
                    yield f.getPTS(), f
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

    API gaps vs ``VideoFrames`` (intentional): ``resized()``,
    ``repeat_each_frame()``, and ``constant_framerate`` are not supported on
    the GPU path. Use the CPU ``VideoFrames`` class when those are needed, or
    apply resizing/repetition on the returned CUDA tensors.

    Args:
        video_path: Path to video file.
        gpu: GPU device ordinal (default 0).
        dtype: Output dtype — ``np.uint8`` (default) or ``np.uint16``.
            For 10-bit sources, ``uint16`` preserves the full precision
            via an NVDEC → NPP color-conversion pipeline.  For 8-bit
            sources, ``uint16`` scales values to the full 0–65535 range.
        color_space: ``'auto'`` (default), ``'bt601'``, or ``'bt709'``.
            Only used when NPP conversion is active (10-bit + uint16).
            ``'auto'`` selects BT.709 for height >= 720, else BT.601.
    """

    def __init__(
        self,
        video_path: PathLike,
        *,
        gpu: int = 0,
        dtype: DTypeLike = np.uint8,
        color_space: str = 'auto',
    ) -> None:
        self.path = str(video_path)
        self._gpu = gpu
        self._npp_init_lock = threading.Lock()

        # Validate dtype.
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            if np.issubdtype(dtype, np.floating):
                raise NotImplementedError(
                    f'dtype={dtype} is not yet supported for GPU decoding. '
                    f'Use np.uint8 or np.uint16, then convert after '
                    f'torch.from_dlpack().'
                )
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
            self._colorspace_id = int(getattr(reader._stream.codec_context, 'colorspace', 0) or 0)
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

        # Decide decode + post-processing strategy.
        source_is_hbd = self._source_format in _HBD_FORMATS
        want_16 = dtype == np.uint16

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

        # Color space (only matters for yuv_to_rgb16 path).
        if color_space == 'auto':
            if self._colorspace_id == 1:  # AVCOL_SPC_BT709
                self._color_space = 'bt709'
            elif self._colorspace_id in (5, 6):  # BT470BG / SMPTE170M (BT.601)
                self._color_space = 'bt601'
            else:
                # Unspecified — fall back to height heuristic.
                self._color_space = 'bt709' if self.original_imshape[0] >= 720 else 'bt601'
        elif color_space in ('bt601', 'bt709'):
            self._color_space = color_space
        else:
            raise ValueError(
                f"color_space must be 'auto', 'bt601', or 'bt709', " f'got {color_space!r}'
            )

    # ── Public interface ──────────────────────────────────────────────

    def __iter__(self):
        """Decode and yield the selected frames on the GPU, in order.

        Each yielded frame stays in GPU memory and supports ``__dlpack__``
        for zero-copy import into PyTorch, CuPy, etc.
        """
        # Stream without the index when the selection is a plain forward
        # slice with a small start. If the index already exists, the
        # seek-based paths are at least as good — use them.
        if not self._selection.is_resolved:
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

        if self._npp_mode is not None:
            self._init_npp_pipeline()

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

    def _iter_streamed(self, streamable: slice):
        """Decode from the start and skip, without building the frame index."""
        if self._npp_mode is not None:
            self._init_npp_pipeline()
        session = self._make_session()
        convert = self._npp_mode is not None
        start = streamable.start or 0
        stop = streamable.stop
        step = streamable.step or 1

        frame_count = 0
        yielded = 0
        for _pts, frame in session.iter_from_start():
            if stop is not None and frame_count >= stop:
                break
            if frame_count >= start and (frame_count - start) % step == 0:
                self._check_decoded_dims(frame)
                if convert:
                    yield self._convert_frame_shared(frame)
                else:
                    yield frame
                yielded += 1
            frame_count += 1
        if yielded == 0 and frame_count == 0 and (stop is None or stop > 0) and start == 0:
            # The decoder produced nothing for a nonempty request — silence
            # here would look like a valid empty video
            raise VideoDecodeError(self.path, 0, RuntimeError('NVDEC decoder produced no frames'))

    def _check_decoded_dims(self, frame) -> None:
        """Detect decoders returning unexpected dimensions (e.g. NVDEC
        splitting interlaced MJPEG into half-height fields) instead of
        silently yielding wrong-shaped frames. Checked once per instance;
        only the RGB path is checked (the NPP path validates plane layouts).
        """
        if self._dims_checked or self._npp_mode is not None:
            return
        self._dims_checked = True
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
                total = self._index.frame_count
                if len(self._resolved_range()) != total:
                    detail = f'view with {length} frames (source video has {total})'
                else:
                    detail = f'video with {length} frames'
                raise IndexError(f'Frame index {item} out of range for {detail}')
            abs_idx = self._resolved_range()[item]
            return self._get_frame_by_abs_idx(abs_idx, owns_memory=True)

        if isinstance(item, slice):
            if item.step == 0:
                raise ValueError('Slice step cannot be zero.')
            result = self._clone()
            result._selection = self._selection.sliced(item)
            return result

        raise TypeError('Indices must be integers or slices.')

    def __len__(self) -> int:
        """Exact number of frames in this view.

        Builds the frame index on first use, which scans the file's packets.
        """
        return len(self._resolved_range())

    def __repr__(self) -> str:
        h, w = self.imshape
        # Never trigger the index scan just for a repr
        length = f'{len(self)} frames' if self._selection.is_resolved else 'lazy'
        return (
            f"VideoFramesCuda('{self.path}', {w}x{h}, {self.fps:.4g} fps, {length}, {self.dtype})"
        )

    @property
    def imshape(self) -> tuple[int, int]:
        """Frame dimensions as (height, width)."""
        return self.original_imshape

    @property
    def fps(self) -> float:
        """Effective frame rate, accounting for slicing.

        Uses the selection's effective stride, which is known even before
        the frame count is — reading fps never triggers the index scan.
        """
        return self.original_fps / abs(self._selection.step_product)

    @property
    def _index(self) -> _FrameIndexNvDec:
        if self._lazy.index is None:
            with self._lazy.lock:
                if self._lazy.index is None:
                    self._lazy.index = _FrameIndexNvDec(self.path)
        return self._lazy.index

    def _resolved_range(self) -> range:
        """The concrete frame-index range, resolving the selection if needed."""
        if not self._selection.is_resolved:
            self._selection = self._selection.resolve(self._index.frame_count)
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
        frame, dec = self._seek_decode_to(safe_pts, target_pts)
        return self._wrap_frame(frame, dec, owns_memory=owns_memory)

    def _seek_decode_to(self, safe_pts: int, target_pts: int):
        """Seek to safe_pts, decode forward to target_pts.

        Returns (frame, decoder) — caller must keep decoder alive while
        using the frame's GPU memory.

        If the file is non-seekable (edit-list files), decodes from the
        beginning and skips to the target PTS.
        """
        # Create a fresh packet source + decoder per seek for clean state.
        src = _PyAVPacketSource(self.path, self._codec_name)
        try:
            dec = nvc.CreateDecoder(
                gpuid=self._gpu,
                codec=self._codec,
                usedevicememory=True,
                outputColorType=self._color_type,
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )

            if self._index.seekable and safe_pts > 0:
                src.seek(safe_pts)

            for pd in src.packets():
                for f in dec.Decode(pd):
                    if f.getPTS() >= target_pts:
                        self._check_decoded_dims(f)
                        return f, dec
            empty = nvc.PacketData()
            while True:
                frames = dec.Decode(empty)
                if not frames:
                    break
                for f in frames:
                    if f.getPTS() >= target_pts:
                        self._check_decoded_dims(f)
                        return f, dec
        except _NVC_ERRORS as e:
            raise VideoDecodeError(self.path, 0, e) from e
        raise RuntimeError(f'Failed to decode frame at PTS {target_pts} (seeked to {safe_pts})')

    def _wrap_frame(self, frame, dec, *, owns_memory: bool):
        """Wrap a decoded frame for output (NPP conversion or DLPack)."""
        if self._npp_mode is not None:
            self._init_npp_pipeline()
            if owns_memory:
                buf = self._convert_frame_fresh(frame)
                del dec
                return buf
            else:
                return self._convert_frame_shared(frame)
        # No NPP: wrap frame + decoder to prevent GC.
        if owns_memory:
            return _FrameWithDecoder(frame, dec)
        else:
            return frame

    # ── Iteration paths ──────────────────────────────────────────────

    def _emit(self, frame, owned: bool):
        """Convert or wrap a decoded frame for yielding.

        ``owned=False`` follows the iteration contract (shared conversion
        buffer / raw decoder frame, valid until the next step). ``owned=True``
        produces a buffer independent of the decode session — required when
        frames are buffered past subsequent decodes (reverse chunks), since
        decoder-owned surfaces are recycled from a bounded pool.
        """
        if self._npp_mode is not None:
            return self._convert_frame_fresh(frame) if owned else self._convert_frame_shared(frame)
        return self._copy_rgb_frame(frame) if owned else frame

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
                self._check_decoded_dims(frame)
                yield self._emit(frame, owned)
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
                yield self._emit(frame, owned)
            frame_count += 1

    def _iter_by_index(self, frame_range: range):
        """Path A: individual seeks for each frame (large step)."""
        convert = self._npp_mode is not None
        # Keep the previous decoder alive between yields — the decoder owns
        # the GPU surface pool that the frame points to.
        prev_dec = None
        for abs_idx in frame_range:
            target_pts = self._index.frame_pts[abs_idx]
            safe_pts = self._index.safe_seek_pts[abs_idx]
            frame, dec = self._seek_decode_to(safe_pts, target_pts)
            if convert:
                yield self._convert_frame_shared(frame)
            else:
                yield frame
            # Assign after yield so previous decoder survives while caller uses the frame
            prev_dec = dec
        del prev_dec

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

    def _copy_rgb_frame(self, frame) -> _GpuRgbBuffer:
        """Copy a decoder-owned RGB uint8 frame into an owned GPU buffer."""
        from cuda.bindings import driver

        h, w = self.original_imshape
        views = frame.cuda()
        view = views[0] if isinstance(views, (list, tuple)) else views
        cai = view.__cuda_array_interface__
        src_ptr = cai['data'][0]
        strides = cai.get('strides')
        src_pitch = strides[0] if strides else w * 3
        row_bytes = w * 3
        if src_pitch < row_bytes:
            raise RuntimeError(f'Unexpected RGB frame pitch {src_pitch} for width {w}')

        err, device = driver.cuDeviceGet(self._gpu)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDeviceGet({self._gpu}) failed: {err}')
        err, ctx = driver.cuDevicePrimaryCtxRetain(device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDevicePrimaryCtxRetain failed: {err}')
        try:
            with cuda_ctx_pushed(ctx):
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

    def _init_npp_pipeline(self) -> None:
        """Lazy-initialize NPP conversion resources (idempotent, thread-safe)."""
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

            try:
                with cuda_ctx_pushed(cuda_ctx):
                    # Build NppStreamContext (uses default stream = 0).
                    npp_ctx = npp_bindings.make_npp_stream_context(self._gpu)

                    # Select color twist matrix.
                    if self._color_space == 'bt709':
                        twist = npp_bindings.BT709_YUV_TO_RGB_16
                    else:
                        twist = npp_bindings.BT601_YUV_TO_RGB_16

                    # Allocate reusable output buffer for iteration.
                    h, w = self.original_imshape
                    out_pitch = w * 3 * 2  # uint16 packed RGB: 3 ch * 2 bytes
                    err, devptr = driver.cuMemAlloc(out_pitch * h)
                    if err != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f'Failed to allocate NPP output buffer: {err}')
            except Exception:
                driver.cuDevicePrimaryCtxRelease(device)
                raise

            # Publish only on full success; the sentinel (_npp_ctx) goes last
            # so the unlocked fast path never sees a partial pipeline.
            self._cuda_device = device
            self._npp_cuda_ctx = cuda_ctx
            self._npp_bindings = npp_bindings
            self._twist = twist
            self._out_pitch = out_pitch
            self._iter_buf_ptr = int(devptr)
            self._npp_ctx = npp_ctx

    def _cleanup_npp(self) -> None:
        """Free NPP pipeline GPU resources."""
        ctx = getattr(self, '_npp_cuda_ctx', None)
        buf = getattr(self, '_iter_buf_ptr', None)
        if buf is not None and ctx is not None:
            from cuda.bindings import driver

            with cuda_ctx_pushed(ctx):
                # NPP work on the default stream may still be in flight.
                driver.cuCtxSynchronize()
                driver.cuMemFree(buf)
            del self._iter_buf_ptr
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
        for attr in ('_npp_ctx', '_npp_bindings', '_twist', '_out_pitch'):
            if hasattr(self, attr):
                delattr(self, attr)

    def _convert_frame_shared(self, frame) -> _GpuRgbBuffer:
        """Convert a frame into the reusable iteration buffer."""
        self._do_convert(frame, self._iter_buf_ptr)
        h, w = self.original_imshape
        return _GpuRgbBuffer(
            self._iter_buf_ptr,
            h,
            w,
            self._out_pitch,
            self._gpu,
            owns_memory=False,
        )

    def _convert_frame_fresh(self, frame) -> _GpuRgbBuffer:
        """Convert a frame into a freshly allocated buffer (for indexing)."""
        from cuda.bindings import driver

        h, w = self.original_imshape
        pitch = w * 3 * 2
        with cuda_ctx_pushed(self._npp_cuda_ctx):
            err, devptr = driver.cuMemAlloc(pitch * h)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to allocate frame buffer: {err}')
        devptr = int(devptr)

        try:
            self._do_convert(frame, devptr)
            with cuda_ctx_pushed(self._npp_cuda_ctx):
                # The NPP kernels run async on the default stream and read the
                # decoder-owned frame memory; sync before the caller drops its
                # frame/decoder references.
                (err,) = driver.cuStreamSynchronize(0)
                if err != driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f'cuStreamSynchronize failed: {err}')
            return _GpuRgbBuffer(
                devptr,
                h,
                w,
                pitch,
                self._gpu,
                owns_memory=True,
            )
        except BaseException:
            with cuda_ctx_pushed(self._npp_cuda_ctx):
                driver.cuMemFree(devptr)
            raise

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
    ) -> None:
        self._devptr = devptr
        self._height = height
        self._width = width
        self._pitch = pitch
        self._gpu_id = gpu_id
        self._owns_memory = owns_memory
        self._bits = bits
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
        mt.dl_tensor.dtype = _DLDataType(1, self._bits, 1)  # kDLUInt
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

    def __del__(self):
        if self._owns_memory and self._devptr:
            _free_owned_buffer(self._devptr, self._own_device, self._own_ctx)


class _FrameWithDecoder:
    """DLPack-compatible wrapper that prevents the decoder from being GC'd.

    When ``VideoFramesCuda[i]`` returns a frame, the underlying decoder
    must stay alive (it owns the GPU surface pool). This wrapper holds
    references to both and produces a DLPack capsule whose deleter prevents
    GC until the consumer is done with the data.
    """

    __slots__ = ('_frame', '_decoder')

    def __init__(self, frame, decoder):
        self._frame = frame
        self._decoder = decoder

    def __dlpack__(self, *args, **kwargs):
        capsule = self._frame.__dlpack__(*args, **kwargs)
        return _dlpack_prevent_gc(capsule, self._decoder, self._frame)

    def __dlpack_device__(self):
        return self._frame.__dlpack_device__()


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
