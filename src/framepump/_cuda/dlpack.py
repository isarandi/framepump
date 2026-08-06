"""DLPack export layer: GPU buffers, frame proxies, capsule machinery.

Everything that hands GPU memory across the library boundary lives here.
Ownership rules: shared (iteration) buffers export repeatedly and never
free; owning buffers hand their allocation and primary-context retain to
the consumer's deleter on first export; decoder-owned frames are proxied so
their exports run under the decode session's CUDA context and keep the
session alive. Deleters may run on any thread at any time, so each carries
the context it needs.
"""

from __future__ import annotations

import ctypes
import itertools
import warnings

from .compat import cuda_ctx_pushed

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
        '_batch',
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
        batch: int | None = None,
    ) -> None:
        self._devptr = devptr
        self._height = height
        self._width = width
        self._pitch = pitch
        self._gpu_id = gpu_id
        self._owns_memory = owns_memory
        self._bits = bits
        self._code = code  # DLPack type code: 1 = kDLUInt, 2 = kDLFloat
        self._batch = batch  # leading dimension for stacked (batch, h, w, 3) buffers
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
        if batch is None:
            self._shape_arr = (ctypes.c_int64 * 3)(height, width, 3)
            self._strides_arr = (ctypes.c_int64 * 3)(width * 3, 3, 1)
        else:
            self._shape_arr = (ctypes.c_int64 * 4)(batch, height, width, 3)
            self._strides_arr = (ctypes.c_int64 * 4)(height * width * 3, width * 3, 3, 1)

    def __dlpack__(self, *args, **kwargs):
        if self._owns_memory is False and self._devptr == 0:
            raise RuntimeError(
                'This buffer already handed its memory to a previous __dlpack__ '
                'export; it cannot be exported again.'
            )
        mt = _DLManagedTensor()
        mt.dl_tensor.data = self._devptr
        mt.dl_tensor.device = _DLDevice(2, self._gpu_id)  # kDLCUDA
        mt.dl_tensor.ndim = 3 if self._batch is None else 4
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
            batch=self._batch,
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
