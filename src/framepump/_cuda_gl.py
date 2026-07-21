"""Upload CUDA tensors to OpenGL textures (GPU-to-GPU, no CPU involved).

Uses CUDA-GL interop to DMA-copy from a CUDA device pointer to a GL texture's
backing CUarray. The GL texture must live on the same physical GPU as the CUDA
context. Must be called from the thread that owns the GL context.

Example:
    >>> import torch
    >>> from OpenGL import GL
    >>> uploader = CudaToGLUploader(tex_id, width=1920, height=1080)
    >>> tensor = torch.from_dlpack(frame)  # (H, W, 3) uint8 on cuda
    >>> uploader.upload(tensor)
    >>> uploader.close()
"""

from __future__ import annotations

import warnings

from cuda.bindings import driver  # type: ignore[attr-defined]

from ._cuda_compat import cuda_ctx_pushed

# OpenGL constants
GL_TEXTURE_2D = 0x0DE1


class CudaToGLUploader:
    """Upload CUDA device memory to an OpenGL texture via GPU DMA.

    Registers the GL texture with CUDA once, then each ``upload()`` call
    maps the texture, copies from the CUDA pointer, and unmaps.

    Args:
        texture_id: OpenGL texture name (from ``glGenTextures``).
            Must be allocated with the correct size and format
            (e.g., ``GL_RGB8`` for RGB uint8 frames).
        width: Texture width in pixels.
        height: Texture height in pixels.
        channels: Number of color channels (default 3 for RGB).
    """

    def __init__(self, texture_id: int, width: int, height: int, channels: int = 3):
        self._texture_id = texture_id
        self._width = width
        self._height = height
        self._channels = channels
        self._resource = None
        self._register()

    def _register(self) -> None:
        err, resource = driver.cuGraphicsGLRegisterImage(
            self._texture_id,
            GL_TEXTURE_2D,
            driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f'Failed to register GL texture {self._texture_id} with CUDA: {err}'
            )
        self._resource = resource
        # Unregistering must happen under the registering context; capture it
        # so close()/GC from another thread can restore it.
        err, ctx = driver.cuCtxGetCurrent()
        self._owner_ctx = ctx if err == driver.CUresult.CUDA_SUCCESS and int(ctx) != 0 else None

    def upload(self, tensor) -> None:
        """Copy a CUDA tensor to the GL texture (GPU-to-GPU).

        Args:
            tensor: A uint8 CUDA tensor of shape ``(height, width, channels)``
                matching the registered texture, C-contiguous, exposing
                ``dtype``/``shape``/``device``/``is_contiguous()``/``data_ptr()``
                (e.g., a ``torch.Tensor``). Other dtypes or strided layouts are
                rejected: the copy is a raw 2D DMA whose pitch assumes one byte
                per channel, and the GL texture's internal format must match.
        """
        if self._resource is None:
            raise RuntimeError('Uploader has been closed.')
        _validate_upload_tensor(tensor, self._height, self._width, self._channels)

        src_ptr = tensor.data_ptr()
        row_bytes = self._width * self._channels

        # Map GL texture → CUarray
        (err,) = driver.cuGraphicsMapResources(1, self._resource, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to map GL resource: {err}')

        try:
            err, cu_array = driver.cuGraphicsSubResourceGetMappedArray(self._resource, 0, 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to get mapped array: {err}')

            # CUDA_MEMCPY2D: device pointer → CUarray
            copy = driver.CUDA_MEMCPY2D()
            copy.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
            copy.srcDevice = src_ptr
            copy.srcPitch = row_bytes
            copy.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.dstArray = cu_array
            copy.WidthInBytes = row_bytes
            copy.Height = self._height

            (err,) = driver.cuMemcpy2D(copy)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to copy CUDA→GL: {err}')
        finally:
            (err,) = driver.cuGraphicsUnmapResources(1, self._resource, 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                # Never raise over an in-flight exception, but a failed unmap
                # leaves the texture mapped to CUDA (GL access is then UB)
                warnings.warn(f'Failed to unmap GL resource: {err}', RuntimeWarning, stacklevel=2)

    def close(self) -> None:
        """Unregister the GL texture from CUDA."""
        if self._resource is not None:
            if self._owner_ctx is not None:
                with cuda_ctx_pushed(self._owner_ctx):
                    (err,) = driver.cuGraphicsUnregisterResource(self._resource)
            else:
                (err,) = driver.cuGraphicsUnregisterResource(self._resource)
            if err != driver.CUresult.CUDA_SUCCESS:
                warnings.warn(
                    f'Failed to unregister GL texture from CUDA: {err} '
                    '(interop registration leaked)',
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._resource = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> CudaToGLUploader:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _validate_upload_tensor(tensor, height: int, width: int, channels: int) -> None:
    dtype = getattr(tensor, 'dtype', None)
    shape = getattr(tensor, 'shape', None)
    device = getattr(tensor, 'device', None)
    is_contiguous = getattr(tensor, 'is_contiguous', None)
    if dtype is None or shape is None or device is None or not callable(is_contiguous):
        raise TypeError(
            'upload() expects a CUDA tensor exposing dtype, shape, device and '
            f'is_contiguous(), e.g. a torch.Tensor; got {type(tensor).__name__}'
        )
    if str(dtype).rsplit('.', 1)[-1] != 'uint8':
        raise ValueError(
            'upload() requires a uint8 tensor (the GL texture stores one byte per '
            f'channel), got dtype {dtype}'
        )
    if getattr(device, 'type', None) != 'cuda':
        raise ValueError(f'upload() requires a tensor on a CUDA device, got {device}')
    if tuple(shape) != (height, width, channels):
        raise ValueError(
            'upload() expects shape (height, width, channels) = '
            f'({height}, {width}, {channels}) matching the registered texture, '
            f'got {tuple(shape)}'
        )
    if not is_contiguous():
        raise ValueError('upload() requires a C-contiguous tensor; call .contiguous() first')
