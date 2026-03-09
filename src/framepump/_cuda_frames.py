"""GPU-resident video frame reader using NVDEC via PyNvVideoCodec low-level API.

Uses PyNvDemuxer + PyNvDecoder (low-level) instead of SimpleDecoder (high-level)
for precise PTS-based seeking and accurate frame counts — even for videos with
edit lists, B-frame reordering, or unreliable container metadata.

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
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import DTypeLike
import PyNvVideoCodec as nvc

PathLike = Union[str, Path]

# Source pixel formats that carry >8 bits of precision.
_HBD_FORMATS = frozenset({
    nvc.Pixel_Format.P016,
    nvc.Pixel_Format.YUV444_16Bit,
})


# ── Frame index ──────────────────────────────────────────────────────

class _FrameIndexNvDec:
    """Packet-based frame index built from PyNvDemuxer.

    Same algorithm as FrameIndexPyAV._build_from_packets() — collects
    file-order PTS, builds running_max_at array, computes safe seek points
    via bisect.  PTS values are raw integers in the stream's time_base.
    """

    def __init__(self, video_path: str) -> None:
        dmx = nvc.CreateDemuxer(video_path)

        # Store metadata from demuxer.
        self.width: int = dmx.Width()
        self.height: int = dmx.Height()
        self.fps: float = dmx.FrameRate()
        self.codec = dmx.GetNvCodecId()
        self.bit_depth: int = dmx.BitDepth()
        self.color_space = dmx.ColorSpace()
        self.chroma_format = dmx.ChromaFormat()

        # Collect PTS in file (packet) order.
        file_order_pts: list[int] = []
        running_max_at: list[int] = []
        running_max = -1

        while True:
            pkt = dmx.Demux()
            if pkt.bsl == 0:
                break

            pts = pkt.pts
            if pts < 0:
                pts = pkt.dts
            if pts < 0:
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

        # Detect edit-list files where PyNvDemuxer.Seek() crashes (SIGSEGV).
        # Compare our packet count with the container header's nb_frames.
        # A large discrepancy (e.g., 437 vs 30) indicates edit-list trimming
        # by the demuxer, which breaks its Seek() function.
        self.seekable: bool = True
        try:
            import av
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                container_nframes = stream.frames
            if container_nframes > 0 and self.frame_count > 0:
                ratio = container_nframes / self.frame_count
                if ratio > 1.5 or ratio < 0.67:
                    self.seekable = False
        except Exception:
            pass  # If av.open fails, assume seekable


# ── Decode session ───────────────────────────────────────────────────

class _NvDecSession:
    """Wraps PyNvDemuxer + PyNvDecoder for a single decode session.

    Created per iteration/access — not shared across clones or concurrent
    iterations.
    """

    def __init__(
        self,
        video_path: str,
        gpu: int,
        codec,
        output_color_type,
    ) -> None:
        self._dmx = nvc.CreateDemuxer(video_path)
        self._dec = nvc.CreateDecoder(
            gpuid=gpu,
            codec=codec,
            usedevicememory=True,
            outputColorType=output_color_type,
            latency=nvc.DisplayDecodeLatencyType.LOW,
        )

    def iter_from_start(self):
        """Decode all frames sequentially from the beginning.

        Yields (pts, frame) tuples in display order.
        """
        while True:
            pkt = self._dmx.Demux()
            if pkt.bsl == 0:
                # Flush buffered frames.
                empty = nvc.PacketData()
                while True:
                    frames = self._dec.Decode(empty)
                    if not frames:
                        break
                    for f in frames:
                        yield f.getPTS(), f
                return

            frames = self._dec.Decode(pkt)
            for f in frames:
                yield f.getPTS(), f

    def iter_from_pts(self, start_pts: int):
        """Seek to start_pts and decode forward.

        Yields (pts, frame) tuples in display order, starting from the first
        frame with PTS >= start_pts.
        """
        self._dmx.Seek(start_pts)
        reached = False

        while True:
            pkt = self._dmx.Demux()
            if pkt.bsl == 0:
                empty = nvc.PacketData()
                while True:
                    frames = self._dec.Decode(empty)
                    if not frames:
                        break
                    for f in frames:
                        pts = f.getPTS()
                        if not reached:
                            if pts >= start_pts:
                                reached = True
                            else:
                                continue
                        yield pts, f
                return

            frames = self._dec.Decode(pkt)
            for f in frames:
                pts = f.getPTS()
                if not reached:
                    if pts >= start_pts:
                        reached = True
                    else:
                        continue
                yield pts, f


# ── Public class ─────────────────────────────────────────────────────

class VideoFramesCuda:
    """Lazy, sliceable GPU video frame iterator using NVDEC (low-level API).

    Frames are decoded on GPU and stay on GPU. Both iteration and indexing
    yield DLPack-compatible objects. Use ``torch.from_dlpack(frame)`` to get
    a CUDA tensor.

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
        gpu: int = 0,
        dtype: DTypeLike = np.uint8,
        color_space: str = 'auto',
    ) -> None:
        self.path = str(video_path)
        self._gpu = gpu

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

        # Build precise frame index from packets (no decoding).
        self._index = _FrameIndexNvDec(self.path)

        self.original_imshape: tuple[int, int] = (self._index.height, self._index.width)
        self.original_fps: float = self._index.fps
        self._frame_range: range = range(self._index.frame_count)

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
            cs = self._index.color_space
            if cs == nvc.ColorSpace.BT_709:
                self._color_space = 'bt709'
            elif cs == nvc.ColorSpace.BT_601:
                self._color_space = 'bt601'
            else:
                # UNSPEC — fall back to height heuristic.
                self._color_space = 'bt709' if self._index.height >= 720 else 'bt601'
        elif color_space in ('bt601', 'bt709'):
            self._color_space = color_space
        else:
            raise ValueError(
                f"color_space must be 'auto', 'bt601', or 'bt709', "
                f"got {color_space!r}"
            )

    # ── Public interface ──────────────────────────────────────────────

    def __iter__(self):
        frame_range = self._frame_range
        if len(frame_range) == 0:
            return

        if self._npp_mode is not None:
            self._init_npp_pipeline()

        try:
            if frame_range.step > 30:
                yield from self._iter_by_index(frame_range)
            elif frame_range.start > 0:
                yield from self._iter_with_seek(frame_range)
            else:
                yield from self._iter_sequential(frame_range)
        finally:
            pass

    def __getitem__(self, item):
        if isinstance(item, int):
            length = len(self)
            if item < 0:
                item = length + item
            if item < 0 or item >= length:
                raise IndexError(
                    f'Frame index {item} out of range for video with '
                    f'{length} frames'
                )
            abs_idx = self._frame_range[item]
            return self._get_frame_by_abs_idx(abs_idx, owns_memory=True)

        if isinstance(item, slice):
            if item.step is not None and item.step < 0:
                raise ValueError('Negative step not supported.')
            if item.step == 0:
                raise ValueError('Slice step cannot be zero.')
            result = self._clone()
            result._frame_range = self._frame_range[item]
            return result

        raise TypeError('Indices must be integers or slices.')

    def __len__(self) -> int:
        return len(self._frame_range)

    def __repr__(self) -> str:
        h, w = self.imshape
        return (
            f"VideoFramesCuda('{self.path}', {w}x{h}, "
            f"{self.fps:.4g} fps, {len(self)} frames, {self.dtype})"
        )

    @property
    def imshape(self) -> tuple[int, int]:
        """Frame dimensions as (height, width)."""
        return self.original_imshape

    @property
    def fps(self) -> float:
        """Effective frame rate, accounting for slicing."""
        return self.original_fps / self._frame_range.step

    def close(self) -> None:
        """Release GPU resources allocated for the NPP pipeline."""
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
        result.dtype = self.dtype
        result._source_format = self._source_format
        result._npp_mode = self._npp_mode
        result._color_type = self._color_type
        result._color_space = self._color_space
        result.original_imshape = self.original_imshape
        result.original_fps = self.original_fps
        result._frame_range = self._frame_range
        # Share index (read-only, immutable).
        result._index = self._index
        # NPP pipeline state is NOT shared — each clone initializes lazily.
        return result

    # ── Internal: format probing ─────────────────────────────────────

    def _probe_source_format(self) -> nvc.Pixel_Format:
        """Infer the NVDEC native pixel format from demuxer metadata.

        Avoids creating a decoder session just for format probing.
        """
        idx = self._index
        chroma = idx.chroma_format
        hbd = idx.bit_depth > 8

        chroma_444 = getattr(nvc.cudaVideoChromaFormat, '444')
        if chroma == chroma_444:
            return nvc.Pixel_Format.YUV444_16Bit if hbd else nvc.Pixel_Format.YUV444
        # 420 (and 422, which NVDEC outputs as 420)
        return nvc.Pixel_Format.P016 if hbd else nvc.Pixel_Format.NV12

    # ── Internal: session creation ───────────────────────────────────

    def _make_session(self) -> _NvDecSession:
        """Create a new decode session with the configured color type."""
        return _NvDecSession(
            self.path, self._gpu, self._index.codec, self._color_type,
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
        # Create fresh demuxer + decoder per seek for clean state.
        dmx = nvc.CreateDemuxer(self.path)
        dec = nvc.CreateDecoder(
            gpuid=self._gpu,
            codec=self._index.codec,
            usedevicememory=True,
            outputColorType=self._color_type,
            latency=nvc.DisplayDecodeLatencyType.LOW,
        )

        if self._index.seekable and safe_pts > 0:
            dmx.Seek(safe_pts)

        while True:
            pkt = dmx.Demux()
            if pkt.bsl == 0:
                # Flush.
                empty = nvc.PacketData()
                while True:
                    frames = dec.Decode(empty)
                    if not frames:
                        break
                    for f in frames:
                        if f.getPTS() >= target_pts:
                            return f, dec
                raise RuntimeError(
                    f'Failed to decode frame at PTS {target_pts} '
                    f'(seeked to {safe_pts})'
                )

            frames = dec.Decode(pkt)
            for f in frames:
                if f.getPTS() >= target_pts:
                    return f, dec

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

    def _iter_sequential(self, frame_range: range):
        """Path C: sequential decode from beginning with step."""
        session = self._make_session()
        convert = self._npp_mode is not None
        start = frame_range.start
        stop = frame_range.stop
        step = frame_range.step

        frame_count = 0
        for _pts, frame in session.iter_from_start():
            if frame_count >= stop:
                break
            if frame_count >= start and (frame_count - start) % step == 0:
                if convert:
                    yield self._convert_frame_shared(frame)
                else:
                    yield frame
            frame_count += 1

    def _iter_with_seek(self, frame_range: range):
        """Path B: seek to start, then sequential with step."""
        start = frame_range.start
        stop = frame_range.stop
        step = frame_range.step

        target_pts = self._index.frame_pts[start]
        safe_pts = self._index.safe_seek_pts[start]

        session = _NvDecSession(
            self.path, self._gpu, self._index.codec, self._color_type,
        )
        convert = self._npp_mode is not None

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
                if convert:
                    yield self._convert_frame_shared(frame)
                else:
                    yield frame
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

    # ── NPP pipeline ─────────────────────────────────────────────────

    def _init_npp_pipeline(self) -> None:
        """Lazy-initialize NPP conversion resources."""
        if hasattr(self, '_npp_ctx'):
            return

        from . import npp_bindings
        from cuda.bindings import driver

        # Initialize CUDA driver API (no-op if already initialized).
        driver.cuInit(0)

        # Ensure a CUDA context is active for this GPU.
        err, device = driver.cuDeviceGet(self._gpu)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDeviceGet({self._gpu}) failed: {err}')
        err, cuda_ctx = driver.cuDevicePrimaryCtxRetain(device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDevicePrimaryCtxRetain failed: {err}')
        err, = driver.cuCtxSetCurrent(cuda_ctx)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuCtxSetCurrent failed: {err}')
        self._cuda_device = device

        self._npp_bindings = npp_bindings

        # Build NppStreamContext (uses default stream = 0).
        self._npp_ctx = npp_bindings.make_npp_stream_context(self._gpu)

        # Select color twist matrix.
        if self._color_space == 'bt709':
            self._twist = npp_bindings.BT709_YUV_TO_RGB_16
        else:
            self._twist = npp_bindings.BT601_YUV_TO_RGB_16

        # Allocate reusable output buffer for iteration.
        h, w = self.original_imshape
        self._out_pitch = w * 3 * 2  # uint16 packed RGB: 3 channels * 2 bytes
        buf_size = self._out_pitch * h
        err, devptr = driver.cuMemAlloc(buf_size)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to allocate NPP output buffer: {err}')
        self._iter_buf_ptr = int(devptr)

    def _cleanup_npp(self) -> None:
        """Free NPP pipeline GPU resources."""
        buf = getattr(self, '_iter_buf_ptr', None)
        if buf is not None:
            from cuda.bindings import driver
            driver.cuMemFree(buf)
            del self._iter_buf_ptr
        device = getattr(self, '_cuda_device', None)
        if device is not None:
            from cuda.bindings import driver
            driver.cuDevicePrimaryCtxRelease(device)
            del self._cuda_device

    def _convert_frame_shared(self, frame) -> _GpuRgbBuffer:
        """Convert a frame into the reusable iteration buffer."""
        self._do_convert(frame, self._iter_buf_ptr)
        h, w = self.original_imshape
        return _GpuRgbBuffer(
            self._iter_buf_ptr, h, w, self._out_pitch, 2, self._gpu,
            owns_memory=False,
        )

    def _convert_frame_fresh(self, frame) -> _GpuRgbBuffer:
        """Convert a frame into a freshly allocated buffer (for indexing)."""
        from cuda.bindings import driver

        h, w = self.original_imshape
        pitch = w * 3 * 2
        err, devptr = driver.cuMemAlloc(pitch * h)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to allocate frame buffer: {err}')
        devptr = int(devptr)

        self._do_convert(frame, devptr)
        return _GpuRgbBuffer(
            devptr, h, w, pitch, 2, self._gpu, owns_memory=True,
        )

    def _do_convert(self, frame, dst_ptr: int) -> None:
        """Dispatch NPP conversion based on mode and source format."""
        h, w = self.original_imshape
        dst_pitch = w * 3 * 2
        npp = self._npp_bindings
        ctx = self._npp_ctx

        if self._npp_mode == 'yuv_to_rgb16':
            pf = self._source_format
            if pf == nvc.Pixel_Format.P016:
                y_ptr = frame.GetPtrToPlane(0)
                uv_ptr = frame.GetPtrToPlane(1)
                y_pitch = (uv_ptr - y_ptr) // h
                npp.p016_to_rgb16(
                    y_ptr, y_pitch, uv_ptr, y_pitch,
                    dst_ptr, dst_pitch, w, h, self._twist, ctx,
                )
            elif pf == nvc.Pixel_Format.YUV444_16Bit:
                y_ptr = frame.GetPtrToPlane(0)
                u_ptr = frame.GetPtrToPlane(1)
                v_ptr = frame.GetPtrToPlane(2)
                plane_pitch = (u_ptr - y_ptr) // h
                npp.yuv444_16bit_to_rgb16(
                    y_ptr, u_ptr, v_ptr, plane_pitch,
                    dst_ptr, dst_pitch, w, h, self._twist, ctx,
                )
            else:
                raise RuntimeError(
                    f'Unsupported source format for yuv_to_rgb16: {pf}'
                )

        elif self._npp_mode == 'scale_8u_16u':
            src_ptr = frame.GetPtrToPlane(0)
            src_pitch = w * 3  # uint8 packed RGB: 3 bytes per pixel
            npp.rgb8_to_rgb16(
                src_ptr, src_pitch, dst_ptr, dst_pitch, w, h, ctx,
            )

        else:
            raise RuntimeError(f'Unknown _npp_mode: {self._npp_mode!r}')


# ── GPU RGB buffer with DLPack export ────────────────────────────────

class _GpuRgbBuffer:
    """DLPack-compatible wrapper around a GPU-resident packed RGB buffer.

    For iteration: ``owns_memory=False``, the buffer is shared and reused.
    For indexing: ``owns_memory=True``, freed when the consumer releases it.
    """

    __slots__ = (
        '_devptr', '_height', '_width', '_pitch', '_bytes_per_ch',
        '_gpu_id', '_owns_memory', '_shape_arr', '_strides_arr',
    )

    def __init__(
        self, devptr: int, height: int, width: int, pitch: int,
        bytes_per_ch: int, gpu_id: int, *, owns_memory: bool,
    ) -> None:
        self._devptr = devptr
        self._height = height
        self._width = width
        self._pitch = pitch
        self._bytes_per_ch = bytes_per_ch
        self._gpu_id = gpu_id
        self._owns_memory = owns_memory
        # Must outlive any DLPack capsule (DLTensor holds raw pointers).
        self._shape_arr = (ctypes.c_int64 * 3)(height, width, 3)
        self._strides_arr = (ctypes.c_int64 * 3)(width * 3, 3, 1)

    def __dlpack__(self, *args, **kwargs):
        mt = _DLManagedTensor()
        mt.dl_tensor.data = self._devptr
        mt.dl_tensor.device = _DLDevice(2, self._gpu_id)  # kDLCUDA
        mt.dl_tensor.ndim = 3
        if self._bytes_per_ch == 1:
            mt.dl_tensor.dtype = _DLDataType(1, 8, 1)   # kDLUInt 8-bit
        else:
            mt.dl_tensor.dtype = _DLDataType(1, 16, 1)  # kDLUInt 16-bit
        mt.dl_tensor.shape = ctypes.cast(
            self._shape_arr, ctypes.POINTER(ctypes.c_int64)
        )
        mt.dl_tensor.strides = ctypes.cast(
            self._strides_arr, ctypes.POINTER(ctypes.c_int64)
        )
        mt.dl_tensor.byte_offset = 0

        key = next(_prevent_gc_counter)

        if self._owns_memory:
            devptr = self._devptr
            self._devptr = 0
            self._owns_memory = False
            mt.deleter = _GPU_BUFFER_FREE_DELETER
        else:
            devptr = 0  # sentinel: don't free
            mt.deleter = _GPU_BUFFER_NOFREE_DELETER

        mt.manager_ctx = key
        _prevent_gc_store[key] = (
            devptr, mt, self._shape_arr, self._strides_arr,
        )

        return _PyCapsule_New(ctypes.addressof(mt), b'dltensor', None)

    def __dlpack_device__(self):
        return (2, self._gpu_id)  # kDLCUDA

    def __del__(self):
        if self._owns_memory and self._devptr:
            from cuda.bindings import driver
            driver.cuMemFree(self._devptr)


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
_prevent_gc_counter = itertools.count()


# ── _GpuRgbBuffer deleters (module-level to survive GC) ──────────────

def _gpu_buffer_free_deleter_impl(managed_ptr):
    """DLPack deleter for owned GPU buffers: frees the allocation."""
    mt = _DLManagedTensor.from_address(managed_ptr)
    ctx = _prevent_gc_store.pop(mt.manager_ctx, None)
    if ctx is not None and ctx[0]:
        from cuda.bindings import driver
        driver.cuMemFree(ctx[0])


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
    _prevent_gc_store[key] = (orig_ptr, capsule, wrapper, _PREVENT_GC_DELETER,
                              *prevent_gc_refs)
    wrapper.manager_ctx = key

    return _PyCapsule_New(ctypes.addressof(wrapper), b'dltensor', None)
