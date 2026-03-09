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

from cuda.bindings import driver  # type: ignore[attr-defined]

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

    def upload(self, tensor) -> None:
        """Copy a CUDA tensor to the GL texture (GPU-to-GPU).

        Args:
            tensor: Any object with a ``.data_ptr()`` method returning a
                CUdeviceptr (e.g., a ``torch.Tensor`` on CUDA). Expected
                shape is ``(H, W, C)`` with contiguous row-major layout.
        """
        if self._resource is None:
            raise RuntimeError('Uploader has been closed.')

        src_ptr = tensor.data_ptr()
        row_bytes = self._width * self._channels

        # Map GL texture → CUarray
        err, = driver.cuGraphicsMapResources(1, self._resource, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to map GL resource: {err}')

        try:
            err, cu_array = driver.cuGraphicsSubResourceGetMappedArray(
                self._resource, 0, 0
            )
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

            err, = driver.cuMemcpy2D(copy)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to copy CUDA→GL: {err}')
        finally:
            driver.cuGraphicsUnmapResources(1, self._resource, 0)

    def close(self) -> None:
        """Unregister the GL texture from CUDA."""
        if self._resource is not None:
            driver.cuGraphicsUnregisterResource(self._resource)
            self._resource = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> CudaToGLUploader:
        return self

    def __exit__(self, *args) -> None:
        self.close()
