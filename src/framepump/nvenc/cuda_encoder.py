"""Zero-copy NVENC encoder via CUDA path (works with EGL and GLX).

This encoder uses CUDA-GL interop to access GL textures as CUarray,
then registers the CUarray directly with NVENC. True zero-copy path:

    GL texture → CUarray (same memory) → NVENC CUDA mode (same memory) → H.264

No intermediate copies. Works with both EGL (headless) and GLX contexts.
"""

from __future__ import annotations

import ctypes
from contextlib import AbstractContextManager
from ctypes import byref, c_void_p, c_uint32
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import moderngl

# OpenGL constant
GL_TEXTURE_2D = 0x0DE1

from cuda.bindings import driver  # type: ignore[attr-defined]

from .._cuda_compat import cuCtxCreate
from .encoder import EncodedPacket
from .exceptions import NvencError, TextureFormatError, EncoderNotInitialized, nvenc_status_message
from .bindings import (
    NvencAPI,
    NVENCAPI_VERSION,
    NV_ENC_DEVICE_TYPE_CUDA,
    NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
    NV_ENC_BUFFER_FORMAT_ABGR,
    NV_ENC_PIC_STRUCT_FRAME,
    NV_ENC_PIC_FLAG_EOS,
    NV_ENC_SUCCESS,
    NV_ENC_ERR_NEED_MORE_INPUT,
    NV_ENC_INPUT_IMAGE,
    NV_ENC_PIC_TYPE_IDR,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
    NV_ENC_INITIALIZE_PARAMS_VER,
    NV_ENC_CONFIG_VER,
    NV_ENC_PRESET_CONFIG_VER,
    NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
    NV_ENC_PIC_PARAMS_VER,
    NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_REGISTER_RESOURCE_VER,
    NV_ENC_MAP_INPUT_RESOURCE_VER,
    GUID,
    NV_ENC_CODEC_H264_GUID,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    NV_ENC_CONFIG,
    NV_ENC_PRESET_CONFIG,
    NV_ENC_INITIALIZE_PARAMS,
    NV_ENC_CREATE_BITSTREAM_BUFFER,
    NV_ENC_REGISTER_RESOURCE,
    NV_ENC_MAP_INPUT_RESOURCE,
    NV_ENC_PIC_PARAMS,
    NV_ENC_LOCK_BITSTREAM,
)
from .presets import DEFAULT_PRESET


class NvencCudaEncoder(AbstractContextManager['NvencCudaEncoder']):
    """Zero-copy NVENC encoder via CUDA path (EGL + GLX support).

    Uses CUDA-GL interop to map GL textures directly to NVENC without
    any memory copies. Works with both EGL (headless) and GLX contexts.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frame rate (default: 30)
        crf: Constant quality factor (0-51, lower = better quality, default: 15)
        gop: GOP length / keyframe interval (default: 250)
        bframes: Number of B-frames (default: 2)

    Example:
        >>> ctx = moderngl.create_standalone_context()  # EGL headless
        >>> with NvencCudaEncoder(640, 480, fps=30, crf=18) as encoder:
        ...     packet = encoder.encode(texture)
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
        self._fps: Fraction = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        self._crf = crf
        self._gop = gop
        self._bframes = bframes
        self._gpu = gpu
        self._preset_config = DEFAULT_PRESET
        self._frame_idx = 0  # Input frame counter (PTS / display order)
        self._output_idx = 0  # Output packet counter (DTS / decode order)
        self._closed = False

        # Initialize CUDA context
        self._cuda_ctx = self._ensure_cuda_context()

        # Texture mappers (texture_id -> mapper)
        self._texture_mappers: dict[int, _GLTextureToCUDA] = {}

        # NVENC state
        self._api = NvencAPI()
        self._encoder = None
        self._registered_arrays: dict[int, tuple] = {}  # cu_array_ptr -> (registered, cu_array)

        # Multiple bitstream buffers for B-frame reordering.
        self._num_bs_buffers = bframes + 2
        self._bitstream_buffers: list = []
        self._next_submit = 0
        self._next_read = 0

        self._init_encoder()

    def _ensure_cuda_context(self) -> Any:
        """Ensure CUDA is initialized on the correct device.

        Device selection priority:
        1. Existing CUDA context on the current thread (reuse it)
        2. Explicit gpu device ordinal (if self._gpu is set)
        3. Auto-detect from GL context via cuGLGetDevices
        """
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to initialize CUDA: {err}')

        # Check if there's already a CUDA context
        err, ctx = driver.cuCtxGetCurrent()
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to get CUDA context: {err}')

        if ctx is not None and int(ctx) != 0:
            # Use existing context (caller's responsibility to ensure it matches GL)
            return ctx

        # No context — pick device: explicit ordinal or auto-detect from GL
        if self._gpu is not None:
            err, device = driver.cuDeviceGet(self._gpu)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to get CUDA device {self._gpu}: {err}')
        else:
            device = self._detect_gl_cuda_device()

        err, ctx = cuCtxCreate(0, device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to create CUDA context: {err}')
        return ctx

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
            max_devices,
            driver.CUGLDeviceList.CU_GL_DEVICE_LIST_ALL
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
            name = name_bytes.decode().rstrip('\x00') if err == driver.CUresult.CUDA_SUCCESS else f'device {i}'

            # Check compute capability (NVENC requires >= 3.0, i.e. Kepler+)
            err, major = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device
            )
            if err == driver.CUresult.CUDA_SUCCESS and major >= 3:
                return device  # Found a good one
            devices_without_nvenc.append(f'{name} (compute {major}.x)')

        # No NVENC-capable device found
        raise NvencError(
            f'No NVENC-capable GPU found for the current OpenGL context.\n\n'
            f'Found {count} CUDA device(s) compatible with GL, but none support NVENC:\n'
            + '\n'.join(f'  - {name}' for name in devices_without_nvenc) +
            '\n\nNVENC requires compute capability >= 3.0:\n'
            '  - GeForce GTX 600 series and newer\n'
            '  - Quadro K series and newer\n'
            '  - Tesla K series and newer'
        )

    def _init_encoder(self) -> None:
        """Initialize NVENC encoder in CUDA device mode."""
        # Open session with CUDA device
        params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
        params.device = int(self._cuda_ctx)
        params.apiVersion = NVENCAPI_VERSION

        encoder = c_void_p()
        status = self._api.nvEncOpenEncodeSessionEx(byref(params), byref(encoder))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to open CUDA encode session'))
        self._encoder = encoder

        # Get codec and preset GUIDs
        codec_guid = NV_ENC_CODEC_H264_GUID
        preset_count = c_uint32()
        self._api.nvEncGetEncodePresetCount(self._encoder, codec_guid, byref(preset_count))
        preset_guids = (GUID * preset_count.value)()
        actual_count = c_uint32()
        self._api.nvEncGetEncodePresetGUIDs(
            self._encoder, codec_guid, preset_guids, preset_count.value, byref(actual_count)
        )
        preset_guid = preset_guids[0]

        # Get preset config (use Ex API with tuning info, required by newer drivers)
        tuning_info = NV_ENC_TUNING_INFO_HIGH_QUALITY
        preset_config = NV_ENC_PRESET_CONFIG()
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER
        self._api.nvEncGetEncodePresetConfigEx(
            self._encoder, codec_guid, preset_guid,
            tuning_info, byref(preset_config))

        # Configure encoder
        config = NV_ENC_CONFIG()
        ctypes.memmove(byref(config), byref(preset_config.presetCfg), ctypes.sizeof(NV_ENC_CONFIG))
        config.version = NV_ENC_CONFIG_VER
        config.gopLength = self._gop
        config.frameIntervalP = self._bframes + 1  # frameIntervalP = 1 + num_b_frames
        config.rcParams.rateControlMode = self._preset_config['rate_control']
        # VBR with targetQuality (CQ mode) - similar to CRF
        config.rcParams.targetQuality = self._crf
        config.rcParams.averageBitRate = 0  # Uncapped
        config.rcParams.maxBitRate = 0  # Uncapped

        self._config = config

        # Initialize
        init_params = NV_ENC_INITIALIZE_PARAMS()
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = codec_guid
        init_params.presetGUID = preset_guid
        init_params.encodeWidth = self._width
        init_params.encodeHeight = self._height
        init_params.darWidth = self._width
        init_params.darHeight = self._height
        init_params.frameRateNum = self._fps.numerator
        init_params.frameRateDen = self._fps.denominator
        init_params.enableEncodeAsync = 0
        init_params.enablePTD = 1
        init_params.tuningInfo = tuning_info
        init_params.encodeConfig = ctypes.pointer(config)

        status = self._api.nvEncInitializeEncoder(self._encoder, byref(init_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to initialize encoder'))

        # Create bitstream buffers
        for i in range(self._num_bs_buffers):
            bs_params = NV_ENC_CREATE_BITSTREAM_BUFFER()
            bs_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
            status = self._api.nvEncCreateBitstreamBuffer(self._encoder, byref(bs_params))
            if status != NV_ENC_SUCCESS:
                raise NvencError(nvenc_status_message(status, f'Failed to create bitstream buffer {i}'))
            self._bitstream_buffers.append(bs_params.bitstreamBuffer)

    def _get_texture_id(self, texture: moderngl.Texture | int) -> int:
        if isinstance(texture, int):
            return texture
        return texture.glo

    def _get_texture_size(self, texture: moderngl.Texture | int) -> tuple[int, int]:
        if isinstance(texture, int):
            return self._width, self._height
        return texture.size

    def _register_cuarray(self, cu_array: Any) -> c_void_p:
        """Register CUarray with NVENC."""
        ptr = int(cu_array)
        if ptr in self._registered_arrays:
            return self._registered_arrays[ptr][0]

        reg = NV_ENC_REGISTER_RESOURCE()
        reg.version = NV_ENC_REGISTER_RESOURCE_VER
        reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY
        reg.width = self._width
        reg.height = self._height
        reg.pitch = self._width * 4
        reg.resourceToRegister = ptr
        reg.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR
        reg.bufferUsage = NV_ENC_INPUT_IMAGE

        status = self._api.nvEncRegisterResource(self._encoder, byref(reg))
        if status != NV_ENC_SUCCESS:
            raise TextureFormatError(nvenc_status_message(status, 'Failed to register CUarray'))

        registered = reg.registeredResource
        self._registered_arrays[ptr] = (registered, cu_array)
        return registered

    def encode(self, texture: moderngl.Texture | int) -> list[EncodedPacket]:
        """Encode a frame from an OpenGL texture (zero-copy).

        Args:
            texture: A moderngl.Texture or OpenGL texture ID (int). Must be RGBA8.

        Returns:
            List of EncodedPackets. Empty if the frame was buffered for
            B-frame reordering; one or more packets when output is ready.
        """
        if self._closed:
            raise EncoderNotInitialized('Encoder has been closed')

        texture_id = self._get_texture_id(texture)

        # Get or create GL-CUDA mapper
        if texture_id not in self._texture_mappers:
            w, h = self._get_texture_size(texture)
            mapper = _GLTextureToCUDA(texture_id, w, h)
            mapper.register()
            self._texture_mappers[texture_id] = mapper

        mapper = self._texture_mappers[texture_id]

        # Map GL texture to CUarray
        cu_array = mapper.map_and_get_array()

        # Register CUarray with NVENC
        registered_resource = self._register_cuarray(cu_array)

        # Map the resource
        map_params = NV_ENC_MAP_INPUT_RESOURCE()
        map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER
        map_params.registeredResource = registered_resource

        status = self._api.nvEncMapInputResource(self._encoder, byref(map_params))
        if status != NV_ENC_SUCCESS:
            mapper.unmap()
            raise NvencError(nvenc_status_message(status, 'Failed to map resource'))

        mapped_resource = map_params.mappedResource
        mapped_format = map_params.mappedBufferFmt

        bs_idx = self._next_submit
        self._next_submit += 1

        try:
            pic_params = NV_ENC_PIC_PARAMS()
            pic_params.version = NV_ENC_PIC_PARAMS_VER
            pic_params.inputWidth = self._width
            pic_params.inputHeight = self._height
            pic_params.inputPitch = self._width * 4
            pic_params.inputBuffer = mapped_resource
            pic_params.outputBitstream = self._bitstream_buffers[bs_idx % self._num_bs_buffers]
            pic_params.bufferFmt = mapped_format
            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
            pic_params.frameIdx = self._frame_idx
            pic_params.inputTimeStamp = self._frame_idx  # PTS = display order

            status = self._api.nvEncEncodePicture(self._encoder, byref(pic_params))
            if status == NV_ENC_ERR_NEED_MORE_INPUT:
                self._frame_idx += 1
                return []
            if status != NV_ENC_SUCCESS:
                raise NvencError(nvenc_status_message(status, f'Failed to encode frame {self._frame_idx}'))
        finally:
            self._api.nvEncUnmapInputResource(self._encoder, mapped_resource)
            mapper.unmap()

        self._frame_idx += 1

        # Read all pending bitstream buffers in submission order
        return self._read_pending(bs_idx)

    def _read_pending(self, up_to_idx: int) -> list[EncodedPacket]:
        """Read all pending bitstream buffers in submission order up to up_to_idx."""
        result = []
        while self._next_read <= up_to_idx:
            buf = self._bitstream_buffers[self._next_read % self._num_bs_buffers]
            lock_bs = NV_ENC_LOCK_BITSTREAM()
            lock_bs.version = NV_ENC_LOCK_BITSTREAM_VER
            lock_bs.outputBitstream = buf
            lock_bs.doNotWait = 0

            status = self._api.nvEncLockBitstream(self._encoder, byref(lock_bs))
            if status != NV_ENC_SUCCESS:
                raise NvencError(
                    f'nvEncLockBitstream failed: {nvenc_status_message(status)}'
                )

            try:
                data = ctypes.string_at(lock_bs.bitstreamBufferPtr, lock_bs.bitstreamSizeInBytes)
                if data:
                    result.append(EncodedPacket(
                        data=data,
                        pts=lock_bs.outputTimeStamp,
                        dts=self._output_idx,
                        is_keyframe=lock_bs.pictureType == NV_ENC_PIC_TYPE_IDR,
                    ))
                    self._output_idx += 1
            finally:
                self._api.nvEncUnlockBitstream(self._encoder, buf)
            self._next_read += 1
        return result

    def flush(self) -> list[EncodedPacket]:
        """Flush any buffered frames from the encoder.

        Call this before close() to retrieve remaining packets when using
        B-frames.

        Returns:
            List of EncodedPackets for any frames still in the reorder buffer.
        """
        if self._closed or self._encoder is None:
            return []

        # Send EOS to signal end of stream
        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS
        self._api.nvEncEncodePicture(self._encoder, byref(pic_params))

        # Read all remaining buffers
        return self._read_pending(self._next_submit - 1)

    def close(self) -> None:
        """Release encoder resources.

        Note: Call flush() first if you need remaining buffered packets.
        """
        if self._closed:
            return
        self._closed = True

        if self._encoder:
            # Unregister CUarrays
            for registered, _ in self._registered_arrays.values():
                self._api.nvEncUnregisterResource(self._encoder, registered)
            self._registered_arrays.clear()

            # Destroy bitstream buffers
            for buf in self._bitstream_buffers:
                self._api.nvEncDestroyBitstreamBuffer(self._encoder, buf)
            self._bitstream_buffers.clear()

            # Destroy encoder
            self._api.nvEncDestroyEncoder(self._encoder)
            self._encoder = None

        # Unregister GL textures from CUDA
        for mapper in self._texture_mappers.values():
            mapper.unregister()
        self._texture_mappers.clear()


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
            driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to register GL texture with CUDA: {err}')
        self._resource = resource

    def map_and_get_array(self) -> Any:
        if self._resource is None:
            raise RuntimeError('Texture not registered')
        if not self._is_mapped:
            err, = driver.cuGraphicsMapResources(1, self._resource, 0)
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
