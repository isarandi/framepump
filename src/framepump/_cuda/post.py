"""Staged post-decode GPU pipeline: color conversion, resize, dtype conversion.

One ``_PostProcessor`` per reader instance, created lazily on first use. It
owns the per-stage reusable buffers, the NPP stream context and its
primary-context retain; ``process`` turns a decoded frame into an output
buffer — shared (reusable) for iteration, owning for indexed access.
"""

from __future__ import annotations

import threading

import numpy as np
import PyNvVideoCodec as nvc

from .compat import cuda_ctx_pushed
from .decode import _plane_layouts
from .dlpack import _GpuRgbBuffer
from .kernels import srgb_curve_inplace


class _PostProcessor:
    """GPU post-processing for one reader configuration."""

    def __init__(
        self,
        *,
        gpu: int,
        dtype: np.dtype,
        npp_mode: str | None,
        source_format,
        color_space: str,
        range_full: bool,
        float_dtype: np.dtype | None,
        out_shape: tuple[int, int] | None,
        gamma_resize: bool,
        original_imshape: tuple[int, int],
    ) -> None:
        self._gpu = gpu
        self.dtype = dtype
        self._npp_mode = npp_mode
        self._source_format = source_format
        self._color_space = color_space
        self._range_full = range_full
        self._float_dtype = float_dtype
        self._out_shape = out_shape
        self._gamma_resize = gamma_resize
        self.original_imshape = original_imshape
        self.imshape = out_shape if out_shape is not None else original_imshape
        self._npp_init_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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

            from .. import npp_bindings
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

    def close(self) -> None:
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

    def process(self, frame, *, fresh: bool) -> _GpuRgbBuffer:
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
                        srgb_curve_inplace(dst, w * h * 3, decode=True)
                        cur = (dst, w * 3 * 4)
                    elif name == 'lin_resize':
                        npp.resize_rgb(
                            cur[0], cur[1], w, h, dst, tw * 3 * 4, tw, th,
                            bits=32, ctx=self._npp_ctx,
                        )  # fmt: skip
                        # Re-encode; the kernel clamps to [0, 1] first, which
                        # also absorbs Lanczos over/undershoot from the resize.
                        srgb_curve_inplace(dst, tw * th * 3, decode=False)
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


def copy_rgb_frame(frame, session, original_imshape, gpu) -> _GpuRgbBuffer:
    """Copy a decoder-owned RGB uint8 frame into an owned GPU buffer."""
    from cuda.bindings import driver

    h, w = original_imshape

    err, device = driver.cuDeviceGet(gpu)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'cuDeviceGet({gpu}) failed: {err}')
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

    return _GpuRgbBuffer(devptr, h, w, row_bytes, gpu, owns_memory=True, bits=8)
