"""High-level NVENC encoder for OpenGL textures."""

from __future__ import annotations

import ctypes
from ctypes import byref, c_void_p, c_uint32
from dataclasses import dataclass
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl


@dataclass
class EncodedPacket:
    """Encoded video packet with timing information for muxing.

    Attributes:
        data: Raw H.264 NAL units.
        pts: Presentation timestamp (display order).
        dts: Decode timestamp (decode order, may differ with B-frames).
        is_keyframe: True if this is an IDR/I-frame.
    """
    data: bytes
    pts: int
    dts: int
    is_keyframe: bool

from .exceptions import (
    NvencError,
    TextureFormatError,
    EncoderNotInitialized,
    nvenc_status_message,
)
from .bindings import (
    NvencAPI,
    NVENCAPI_VERSION,
    NV_ENC_DEVICE_TYPE_OPENGL,
    NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX,
    NV_ENC_BUFFER_FORMAT_ABGR,
    NV_ENC_PIC_STRUCT_FRAME,
    NV_ENC_PIC_FLAG_EOS,
    NV_ENC_SUCCESS,
    NV_ENC_ERR_NEED_MORE_INPUT,
    NV_ENC_INPUT_IMAGE,
    NV_ENC_PIC_TYPE_IDR,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
    GL_TEXTURE_2D,
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
    NV_ENC_INPUT_RESOURCE_OPENGL_TEX,
    NV_ENC_REGISTER_RESOURCE,
    NV_ENC_MAP_INPUT_RESOURCE,
    NV_ENC_PIC_PARAMS,
    NV_ENC_LOCK_BITSTREAM,
)
from .presets import DEFAULT_PRESET

class NvencEncoder:
    """
    High-level NVENC encoder for OpenGL textures.

    Encodes OpenGL textures directly to H.264 video using NVIDIA's hardware
    encoder without CPU memory transfers (zero-copy).

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frame rate (default: 30)
        crf: Constant quality factor (0-51, lower = better quality, default: 15)
        gop: GOP length / keyframe interval (default: 250)
        bframes: Number of B-frames (default: 2)

    Example:
        >>> with NvencEncoder(640, 480, fps=30, crf=18) as encoder:
        ...     packet = encoder.encode(texture)
        ...     # packet.data contains H.264 NAL units
        ...     # packet.pts, packet.dts for timestamps
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float | Fraction = 30,
        crf: int = 15,
        gop: int = 250,
        bframes: int = 2,
    ) -> None:
        self._width = width
        self._height = height
        self._fps: Fraction = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        self._crf = crf
        self._gop = gop
        self._bframes = bframes
        self._preset_config = DEFAULT_PRESET
        self._frame_idx = 0  # Input frame counter (PTS / display order)
        self._output_idx = 0  # Output packet counter (DTS / decode order)
        self._closed = False

        # Resource tracking
        self._registered_textures: dict[
            int, tuple] = {}  # texture_id -> (registered_resource, gl_tex_struct)

        # Initialize NVENC
        self._api = NvencAPI()
        self._encoder = None

        # Multiple bitstream buffers for B-frame reordering.
        # NVENC may output multiple packets per encode call when B-frames
        # are active (e.g., P + B + B), requiring separate output buffers.
        self._num_bs_buffers = bframes + 2
        self._bitstream_buffers: list = []
        self._next_submit = 0  # Next buffer index to use for encoding
        self._next_read = 0  # Next buffer index to read output from

        self._init_encoder()

    def _init_encoder(self) -> None:
        """Initialize the NVENC encoder session."""
        # Verify the current GL context is on an NVIDIA GPU.
        # NVENC with NV_ENC_DEVICE_TYPE_OPENGL requires an NVIDIA-backed context;
        # on hybrid GPU systems (e.g., AMD iGPU + NVIDIA dGPU), the default context
        # may be on the non-NVIDIA GPU, which causes a segfault in the NVENC driver.
        try:
            import ctypes as _ct
            _gl = _ct.cdll.LoadLibrary('libGL.so.1')
            _gl.glGetString.restype = _ct.c_char_p
            _renderer = (_gl.glGetString(0x1F01) or b'').decode(errors='replace')
        except Exception:
            _renderer = ''

        if _renderer and 'nvidia' not in _renderer.lower():
            raise NvencError(
                f'Current OpenGL context is on a non-NVIDIA GPU: {_renderer}\n\n'
                'NVENC requires an NVIDIA-backed OpenGL context.\n\n'
                'Solutions:\n'
                '  - Set __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia\n'
                '    to route OpenGL to the NVIDIA GPU\n'
                '  - For headless (EGL): unset DISPLAY to use the CUDA encoder path'
            )

        # Open encode session
        params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        params.deviceType = NV_ENC_DEVICE_TYPE_OPENGL
        params.device = None
        params.apiVersion = NVENCAPI_VERSION

        encoder = c_void_p()
        status = self._api.nvEncOpenEncodeSessionEx(byref(params), byref(encoder))
        if status != NV_ENC_SUCCESS:
            raise NvencError(
                nvenc_status_message(status, 'Failed to open NVENC encode session') + '\n\n'
                'The current OpenGL context may be on a GPU without NVENC support\n'
                '(e.g., Intel/AMD integrated graphics).\n\n'
                'Solutions:\n'
                '  - Set __NV_PRIME_RENDER_OFFLOAD=1 to route OpenGL to the NVIDIA GPU\n'
                '  - Use DRI_PRIME=1 to select the NVIDIA GPU\n'
                '  - For headless (EGL): set DISPLAY= to use the CUDA encoder path instead'
            )
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

        # Initialize encoder
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
        init_params.enableEncodeAsync = 0  # Synchronous mode (Linux only)
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
        """Extract OpenGL texture ID from moderngl.Texture or int."""
        if isinstance(texture, int):
            return texture
        return texture.glo

    def _register_texture(self, texture_id: int) -> c_void_p:
        """Register an OpenGL texture with NVENC."""
        if texture_id in self._registered_textures:
            return self._registered_textures[texture_id][0]

        gl_tex_resource = NV_ENC_INPUT_RESOURCE_OPENGL_TEX()
        gl_tex_resource.texture = texture_id
        gl_tex_resource.target = GL_TEXTURE_2D

        register_params = NV_ENC_REGISTER_RESOURCE()
        register_params.version = NV_ENC_REGISTER_RESOURCE_VER
        register_params.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX
        register_params.width = self._width
        register_params.height = self._height
        register_params.pitch = self._width * 4  # RGBA = 4 bytes per pixel
        register_params.resourceToRegister = ctypes.addressof(gl_tex_resource)
        register_params.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR
        register_params.bufferUsage = NV_ENC_INPUT_IMAGE

        status = self._api.nvEncRegisterResource(self._encoder, byref(register_params))
        if status != NV_ENC_SUCCESS:
            raise TextureFormatError(nvenc_status_message(status, f'Failed to register texture {texture_id}'))

        registered = register_params.registeredResource
        # Keep gl_tex_resource alive
        self._registered_textures[texture_id] = (registered, gl_tex_resource)
        return registered

    def encode(self, texture: moderngl.Texture | int) -> list[EncodedPacket]:
        """
        Encode a frame from an OpenGL texture.

        Args:
            texture: A moderngl.Texture object or OpenGL texture ID (int).
                     The texture must be RGBA8 format.

        Returns:
            List of EncodedPackets. Empty if the frame was buffered for
            B-frame reordering; one or more packets when output is ready.

        Note:
            Ensure OpenGL commands are complete before calling this method.
            If using moderngl, call ctx.finish() first.
        """
        if self._closed:
            raise EncoderNotInitialized('Encoder has been closed')

        texture_id = self._get_texture_id(texture)
        registered_resource = self._register_texture(texture_id)

        # Map the resource
        map_params = NV_ENC_MAP_INPUT_RESOURCE()
        map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER
        map_params.registeredResource = registered_resource

        status = self._api.nvEncMapInputResource(self._encoder, byref(map_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to map resource'))

        mapped_resource = map_params.mappedResource
        mapped_format = map_params.mappedBufferFmt

        bs_idx = self._next_submit
        self._next_submit += 1

        try:
            # Encode the frame
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
            # Unregister all textures
            for registered, _ in self._registered_textures.values():
                self._api.nvEncUnregisterResource(self._encoder, registered)
            self._registered_textures.clear()

            # Destroy bitstream buffers
            for buf in self._bitstream_buffers:
                self._api.nvEncDestroyBitstreamBuffer(self._encoder, buf)
            self._bitstream_buffers.clear()

            # Destroy encoder
            self._api.nvEncDestroyEncoder(self._encoder)
            self._encoder = None

    def __enter__(self) -> NvencEncoder:
        return self

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
