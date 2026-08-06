"""NVENC encoder via CUDA path (works with EGL and GLX).

This encoder uses CUDA-GL interop to access GL textures as CUarray, copies
each frame into an internal ring of staging CUDA arrays, and encodes those
with NVENC in CUDA device mode:

    GL texture → CUarray → staging CUarray ring → NVENC CUDA mode → H.264

The staging copy decouples the caller's texture from the encoder pipeline:
the caller may re-render into the source texture immediately after encode()
returns, even with B-frames enabled (NVENC reads inputs asynchronously while
frames are buffered for reordering). Works with both EGL (headless) and GLX
contexts.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from ctypes import c_void_p
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import moderngl

from cuda.bindings import driver  # type: ignore[attr-defined]

from .._cuda.compat import cuda_ctx_pushed
from ._session import EncodedPacket, NvencEncodeSession
from .bindings import (
    NV_ENC_BUFFER_FORMAT_ABGR,
    NV_ENC_DEVICE_TYPE_CUDA,
    NV_ENC_INPUT_IMAGE,
    NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
    NV_ENC_REGISTER_RESOURCE,
    NV_ENC_REGISTER_RESOURCE_VER,
)
from .exceptions import EncoderNotInitialized, NvencError

__all__ = ['NvencCudaEncoder', 'EncodedPacket']

# OpenGL constant
GL_TEXTURE_2D = 0x0DE1


class NvencCudaEncoder(AbstractContextManager['NvencCudaEncoder']):
    """NVENC encoder via CUDA path (EGL + GLX support).

    Uses CUDA-GL interop to read GL textures and NVENC's CUDA device mode
    for encoding. Each frame is copied into an internal staging CUDA array,
    so the source texture is free to be re-rendered as soon as encode()
    returns. Works with both EGL (headless) and GLX contexts.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frame rate (default: 30)
        crf: Constant quality factor (0-51, lower = better quality, default: 15)
        gop: GOP length / keyframe (IDR) interval (default: 250)
        bframes: Number of B-frames (default: 2)
        gpu: CUDA device ordinal; None auto-detects from the GL context.

    Example:
        >>> ctx = moderngl.create_standalone_context()  # EGL headless
        >>> with NvencCudaEncoder(640, 480, fps=30, crf=18) as encoder:
        ...     packets = encoder.encode(texture)
        ...     # packet.data contains H.264 NAL units
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float | Fraction = 30,
        crf: int = 15,
        gop: int = 250,
        bframes: int = 2,
        gpu: int | None = None,
    ) -> None:
        self._width = width
        self._height = height
        fps = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        self._gpu = gpu
        self._closed = False
        self._session: NvencEncodeSession | None = None

        # Texture mappers (texture_id -> mapper)
        self._texture_mappers: dict[int, _GLTextureToCUDA] = {}

        # Staging arrays and their NVENC registrations, filled lazily.
        self._staging_arrays: list[Any] | None = None
        self._registered_staging: list[c_void_p] = []

        # Initialize CUDA context. _cuda_ctx starts as None so __del__ stays
        # safe if context acquisition itself raises.
        self._cuda_device = None
        self._owns_cuda_ctx = False
        self._cuda_ctx: Any = None
        self._cuda_ctx = self._ensure_cuda_context()

        try:
            self._session = NvencEncodeSession(
                device_type=NV_ENC_DEVICE_TYPE_CUDA,
                device=int(self._cuda_ctx),
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                gop=gop,
                bframes=bframes,
            )
        except Exception:
            self._release_owned_ctx()
            # The context is gone; mark closed so __del__ doesn't push it
            self._cuda_ctx = None
            self._closed = True
            raise

    def _ensure_cuda_context(self) -> Any:
        """Ensure CUDA is initialized on the correct device.

        Device selection priority:
        1. Existing CUDA context on the current thread (reuse it)
        2. Explicit gpu device ordinal (if self._gpu is set)
        3. Auto-detect from GL context via cuGLGetDevices

        In cases 2 and 3 the encoder retains the device's primary context; it
        is made current only for the duration of encoder calls, so the
        caller's current context (or the absence of one) is preserved.
        """
        (err,) = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to initialize CUDA: {err}')

        # Check if there's already a CUDA context
        err, ctx = driver.cuCtxGetCurrent()
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to get CUDA context: {err}')

        if ctx is not None and int(ctx) != 0:
            if self._gpu is not None:
                # An explicit ordinal overrides the current context if they
                # disagree (e.g. torch active on another device).
                err, current_device = driver.cuCtxGetDevice()
                if err == driver.CUresult.CUDA_SUCCESS and int(current_device) == self._gpu:
                    return ctx
            else:
                # Use existing context (caller ensures it matches GL)
                return ctx

        # Pick device: explicit ordinal or auto-detect from GL
        if self._gpu is not None:
            err, device = driver.cuDeviceGet(self._gpu)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to get CUDA device {self._gpu}: {err}')
        else:
            device = self._detect_gl_cuda_device()

        err, ctx = driver.cuDevicePrimaryCtxRetain(device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to retain primary CUDA context: {err}')
        self._cuda_device = device
        self._owns_cuda_ctx = True
        return ctx

    def _release_owned_ctx(self) -> None:
        if self._owns_cuda_ctx and self._cuda_device is not None:
            driver.cuDevicePrimaryCtxRelease(self._cuda_device)
            self._cuda_device = None
            self._owns_cuda_ctx = False

    def _detect_gl_cuda_device(self) -> Any:
        """Detect which CUDA device the current GL context is on.

        Returns:
            CUDA device handle for an NVENC-capable GPU backing the current GL context.

        Raises:
            NvencError: If no NVENC-capable NVIDIA GPU is found for the GL context.
        """
        # Query which CUDA devices can interop with the current GL context
        # cuda-python 13+ returns (err, count, devices_list) directly
        max_devices = 16
        err, count, devices = driver.cuGLGetDevices(
            max_devices, driver.CUGLDeviceList.CU_GL_DEVICE_LIST_ALL
        )

        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(
                f'Failed to query CUDA devices for GL context: {err}\n\n'
                'This can happen if:\n'
                '  - No OpenGL context is current (call from the rendering thread)\n'
                '  - The GL context is on a non-NVIDIA GPU (Intel/AMD integrated graphics)\n'
                '  - CUDA-GL interop is not supported by the driver'
            )

        if count == 0:
            raise NvencError(
                'No NVIDIA GPU found for the current OpenGL context.\n\n'
                'NVENC encoding requires the GL texture to be on an NVIDIA GPU.\n'
                'Possible causes:\n'
                '  - OpenGL is running on integrated graphics (Intel/AMD)\n'
                '  - OpenGL is running on a non-NVIDIA discrete GPU\n\n'
                'Solutions:\n'
                '  - Set __NV_PRIME_RENDER_OFFLOAD=1 to use NVIDIA GPU for rendering\n'
                '  - Use DRI_PRIME=1 or similar to select the NVIDIA GPU\n'
                '  - Configure your system to use the NVIDIA GPU for this application\n'
                '  - For EGL: use eglQueryDevicesEXT to enumerate and select NVIDIA device'
            )

        # Find a device with NVENC support (compute capability >= 3.0)
        devices_without_nvenc = []
        for i in range(count):
            device = devices[i]
            err, name_bytes = driver.cuDeviceGetName(256, device)
            name = (
                name_bytes.decode().rstrip('\x00')
                if err == driver.CUresult.CUDA_SUCCESS
                else f'device {i}'
            )

            # Check compute capability (NVENC requires >= 3.0, i.e. Kepler+)
            err, major = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
            )
            if err == driver.CUresult.CUDA_SUCCESS and major >= 3:
                return device  # Found a good one
            devices_without_nvenc.append(f'{name} (compute {major}.x)')

        # No NVENC-capable device found
        raise NvencError(
            f'No NVENC-capable GPU found for the current OpenGL context.\n\n'
            f'Found {count} CUDA device(s) compatible with GL, but none support NVENC:\n'
            + '\n'.join(f'  - {name}' for name in devices_without_nvenc)
            + '\n\nNVENC requires compute capability >= 3.0:\n'
            '  - GeForce GTX 600 series and newer\n'
            '  - Quadro K series and newer\n'
            '  - Tesla K series and newer'
        )

    def encode(self, texture: moderngl.Texture | int) -> list[EncodedPacket]:
        """Encode a frame from an OpenGL texture.

        The texture content is copied into an internal staging CUDA array
        before submission, so the caller may modify or re-render the source
        texture immediately after this call returns.

        Args:
            texture: A moderngl.Texture or OpenGL texture ID (int). Must be RGBA8.

        Returns:
            List of EncodedPackets. Empty if the frame was buffered for
            B-frame reordering; one or more packets when output is ready.
        """
        if self._closed:
            raise EncoderNotInitialized('Encoder has been closed')

        if not isinstance(texture, int):
            # A larger texture would otherwise be silently cropped to the
            # encoder dimensions (raw GL ids cannot be checked).
            if texture.size != (self._width, self._height):
                raise ValueError(
                    f'Texture size {texture.size} does not match encoder '
                    f'dimensions ({self._width}, {self._height})'
                )

        texture_id = self._get_texture_id(texture)

        with cuda_ctx_pushed(self._cuda_ctx):
            # Get or create GL-CUDA mapper
            if texture_id not in self._texture_mappers:
                w, h = self._get_texture_size(texture)
                mapper = _GLTextureToCUDA(texture_id, w, h)
                mapper.register()
                self._texture_mappers[texture_id] = mapper

            mapper = self._texture_mappers[texture_id]

            if self._staging_arrays is None:
                self._create_staging_arrays()

            slot = self._session.next_submit_index % len(self._staging_arrays)

            # Copy the mapped GL texture into the staging array, then release
            # the GL mapping immediately (cuMemcpy2D is synchronous).
            cu_array = mapper.map_and_get_array()
            try:
                copy = driver.CUDA_MEMCPY2D()
                copy.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
                copy.srcArray = cu_array
                copy.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
                copy.dstArray = self._staging_arrays[slot]
                copy.WidthInBytes = self._width * 4
                copy.Height = self._height
                (err,) = driver.cuMemcpy2D(copy)
                if err != driver.CUresult.CUDA_SUCCESS:
                    raise NvencError(f'Failed to copy texture into staging array: {err}')
            finally:
                mapper.unmap()

            return self._session.submit(
                self._registered_staging[slot], self._width, self._height, self._width * 4
            )

    def _create_staging_arrays(self) -> None:
        """Allocate and register the staging CUDA array ring."""
        desc = driver.CUDA_ARRAY_DESCRIPTOR()
        desc.Width = self._width
        desc.Height = self._height
        desc.Format = driver.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8
        desc.NumChannels = 4

        arrays = []
        try:
            for i in range(self._session.ring_size):
                err, array = driver.cuArrayCreate(desc)
                if err != driver.CUresult.CUDA_SUCCESS:
                    raise NvencError(f'Failed to allocate staging array {i}: {err}')
                arrays.append(array)
        except BaseException:
            for array in arrays:
                driver.cuArrayDestroy(array)
            raise
        self._staging_arrays = arrays

        for array in arrays:
            reg = NV_ENC_REGISTER_RESOURCE()
            reg.version = NV_ENC_REGISTER_RESOURCE_VER
            reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY
            reg.width = self._width
            reg.height = self._height
            reg.pitch = self._width * 4
            reg.resourceToRegister = int(array)
            reg.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR
            reg.bufferUsage = NV_ENC_INPUT_IMAGE
            self._registered_staging.append(self._session.register_input(reg))

    def _get_texture_id(self, texture: moderngl.Texture | int) -> int:
        if isinstance(texture, int):
            return texture
        return texture.glo

    def _get_texture_size(self, texture: moderngl.Texture | int) -> tuple[int, int]:
        if isinstance(texture, int):
            return self._width, self._height
        return texture.size

    def flush(self) -> list[EncodedPacket]:
        """Flush any buffered frames from the encoder. Idempotent.

        Call this before close() to retrieve remaining packets when using
        B-frames.

        Returns:
            List of EncodedPackets for any frames still in the reorder buffer.
        """
        if self._closed or self._session is None:
            return []
        with cuda_ctx_pushed(self._cuda_ctx):
            return self._session.flush()

    def close(self) -> None:
        """Release encoder resources.

        Note: Call flush() first if you need remaining buffered packets.
        """
        if self._closed:
            return
        self._closed = True

        if self._cuda_ctx is None:
            # Context acquisition failed during __init__; nothing to release
            return

        with cuda_ctx_pushed(self._cuda_ctx):
            if self._session is not None:
                self._session.close()
            self._registered_staging.clear()

            if self._staging_arrays is not None:
                for array in self._staging_arrays:
                    driver.cuArrayDestroy(array)
                self._staging_arrays = None

            # Unregister GL textures from CUDA
            for mapper in self._texture_mappers.values():
                mapper.unregister()
            self._texture_mappers.clear()

        self._release_owned_ctx()

    def __del__(self) -> None:
        if not self._closed:
            self.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class _GLTextureToCUDA(AbstractContextManager['_GLTextureToCUDA']):
    """Maps GL texture to CUDA array for zero-copy access."""

    def __init__(self, texture_id: int, width: int, height: int):
        self._texture_id = texture_id
        self._width = width
        self._height = height
        self._resource = None
        self._is_mapped = False

    def register(self) -> None:
        if self._resource is not None:
            return
        err, resource = driver.cuGraphicsGLRegisterImage(
            self._texture_id,
            GL_TEXTURE_2D,
            driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to register GL texture with CUDA: {err}')
        self._resource = resource

    def map_and_get_array(self) -> Any:
        if self._resource is None:
            raise RuntimeError('Texture not registered')
        if not self._is_mapped:
            (err,) = driver.cuGraphicsMapResources(1, self._resource, 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to map GL resource: {err}')
            self._is_mapped = True
        err, cu_array = driver.cuGraphicsSubResourceGetMappedArray(self._resource, 0, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get mapped array: {err}')
        return cu_array

    def unmap(self) -> None:
        if self._is_mapped and self._resource is not None:
            driver.cuGraphicsUnmapResources(1, self._resource, 0)
            self._is_mapped = False

    def unregister(self) -> None:
        if self._resource is not None:
            if self._is_mapped:
                self.unmap()
            driver.cuGraphicsUnregisterResource(self._resource)
            self._resource = None

    def __exit__(self, *args: Any) -> None:
        self.unregister()
