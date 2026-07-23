"""Shared NVENC encode-session core used by the GL and CUDA encoders.

Owns everything the two encoder classes have in common: session setup,
encoder configuration, buffer-ring sizing, frame submission with B-frame
reordering, output draining, EOS flush, and teardown. The public encoder
classes are thin input adapters: per frame they prepare a registered input
resource (a staging GL texture or a staging CUDA array) and hand it to
submit().

Buffer lifetime follows the NVENC contract: every submitted input stays
mapped until the bitstream produced from it has been consumed, and the
ring sizes are derived from the finalized encoder config, so the encoder
can never be asked to read a buffer that has been reused.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_void_p
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from .bindings import (
    NV_ENC_CODEC_H264_GUID,
    NV_ENC_CONFIG,
    NV_ENC_CONFIG_VER,
    NV_ENC_CREATE_BITSTREAM_BUFFER,
    NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
    NV_ENC_ERR_NEED_MORE_INPUT,
    NV_ENC_INITIALIZE_PARAMS,
    NV_ENC_INITIALIZE_PARAMS_VER,
    NV_ENC_LOCK_BITSTREAM,
    NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_MAP_INPUT_RESOURCE,
    NV_ENC_MAP_INPUT_RESOURCE_VER,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
    NV_ENC_PARAMS_RC_VBR,
    NV_ENC_PIC_FLAG_EOS,
    NV_ENC_PIC_PARAMS,
    NV_ENC_PIC_PARAMS_VER,
    NV_ENC_PIC_STRUCT_FRAME,
    NV_ENC_PIC_TYPE_IDR,
    NV_ENC_PRESET_CONFIG,
    NV_ENC_PRESET_CONFIG_VER,
    NV_ENC_PRESET_P4_GUID,
    NV_ENC_REGISTER_RESOURCE,
    NV_ENC_SUCCESS,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
    NVENCAPI_VERSION,
    NvencAPI,
)
from .exceptions import (
    EncoderNotInitialized,
    NvencError,
    TextureFormatError,
    nvenc_status_message,
)

# Bit 5 of NV_ENC_RC_PARAMS.rcFlags (after enableMinQP, enableMaxQP,
# enableInitialRCQP, enableAQ, reservedBitField1).
_RC_FLAG_ENABLE_LOOKAHEAD = 1 << 5


@dataclass
class EncodedPacket:
    """Encoded video packet with timing information for muxing.

    Attributes:
        data: Raw H.264 NAL units.
        pts: Presentation timestamp (display order).
        dts: Decode timestamp (decode order, may differ with B-frames).
        is_keyframe: True if this is an IDR frame.
    """

    data: bytes
    pts: int
    dts: int
    is_keyframe: bool


class NvencEncodeSession:
    """One NVENC encode session: configuration, buffer rings, submit/drain.

    Args:
        device_type: NV_ENC_DEVICE_TYPE_OPENGL or NV_ENC_DEVICE_TYPE_CUDA.
        device: Device handle matching device_type (None for OpenGL,
            the CUDA context handle as int for CUDA).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate as a Fraction.
        crf: Constant quality factor (0-51, lower is better quality).
        gop: GOP length; also used as the IDR period.
        bframes: Number of B-frames between reference frames.
        dar_size: Display aspect ratio as (width, height); defaults to the
            coded dimensions. Pass the display size when width/height carry
            macroblock padding.
        tune_config: Optional hook called with the NV_ENC_CONFIG after the
            base configuration is built and before the encoder is
            initialized. May adjust rate control, profile, chroma format or
            VUI settings; the buffer rings are sized from the config as the
            hook leaves it.
        open_error_hint: Extra text appended to a session-open failure.
    """

    def __init__(
        self,
        *,
        device_type: int,
        device: int | None,
        width: int,
        height: int,
        fps: Fraction,
        crf: int = 15,
        gop: int = 250,
        bframes: int = 2,
        dar_size: tuple[int, int] | None = None,
        tune_config: Callable[[NV_ENC_CONFIG], None] | None = None,
        open_error_hint: str = '',
    ) -> None:
        self._api = NvencAPI()
        self._encoder = None
        self._closed = False
        self._flushed = False
        self._frame_idx = 0  # Input frame counter (PTS / display order)
        self._output_idx = 0  # Output packet counter (DTS / decode order)
        self._next_submit = 0  # Next submission index
        self._next_read = 0  # Next submission index to read output for
        self._bitstream_buffers: list = []
        self._registered_resources: list = []
        # Submission index -> (mapped input resource, per-frame cleanup).
        # Inputs stay mapped until their output has been consumed (NVENC
        # reads input surfaces asynchronously while B-frames are pending).
        self._pending_inputs: dict[int, tuple[c_void_p, Callable[[], None] | None]] = {}

        self._open_session(device_type, device, open_error_hint)
        try:
            self._configure(width, height, fps, crf, gop, bframes, dar_size, tune_config)

            # Ring sizing from the *finalized* config: the encoder may hold
            # frameIntervalP - 1 frames for reordering plus lookaheadDepth for
            # lookahead, and one more submission is in flight; one extra slot
            # of margin.
            config = self._config
            if config.rcParams.rcFlags & _RC_FLAG_ENABLE_LOOKAHEAD:
                lookahead = config.rcParams.lookaheadDepth
            else:
                lookahead = 0
            self._ring_size = config.frameIntervalP + lookahead + 1
            self._create_bitstream_buffers()
        except BaseException:
            self.close()
            raise

    @property
    def ring_size(self) -> int:
        """Number of slots input staging rings must provide."""
        return self._ring_size

    @property
    def next_submit_index(self) -> int:
        """Submission index the next submit() call will use."""
        return self._next_submit

    def _open_session(self, device_type: int, device: int | None, hint: str) -> None:
        params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        params.deviceType = device_type
        params.device = device
        params.apiVersion = NVENCAPI_VERSION

        encoder = c_void_p()
        status = self._api.nvEncOpenEncodeSessionEx(byref(params), byref(encoder))
        if status != NV_ENC_SUCCESS:
            message = nvenc_status_message(status, 'Failed to open NVENC encode session')
            if hint:
                message += '\n\n' + hint
            raise NvencError(message)
        self._encoder = encoder

    def _configure(
        self,
        width: int,
        height: int,
        fps: Fraction,
        crf: int,
        gop: int,
        bframes: int,
        dar_size: tuple[int, int] | None,
        tune_config: Callable[[NV_ENC_CONFIG], None] | None,
    ) -> None:
        codec_guid = NV_ENC_CODEC_H264_GUID
        preset_guid = NV_ENC_PRESET_P4_GUID
        tuning_info = NV_ENC_TUNING_INFO_HIGH_QUALITY

        preset_config = NV_ENC_PRESET_CONFIG()
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER
        status = self._api.nvEncGetEncodePresetConfigEx(
            self._encoder, codec_guid, preset_guid, tuning_info, byref(preset_config)
        )
        if status != NV_ENC_SUCCESS:
            raise NvencError(self._error(status, 'Failed to query preset config'))

        config = NV_ENC_CONFIG()
        ctypes.memmove(byref(config), byref(preset_config.presetCfg), ctypes.sizeof(NV_ENC_CONFIG))
        config.version = NV_ENC_CONFIG_VER
        config.gopLength = gop
        config.frameIntervalP = bframes + 1  # frameIntervalP = 1 + num_b_frames
        # The preset default IDR period is independent of gopLength; align it
        # so a custom gop yields true random-access points at that interval.
        config.encodeCodecConfig.h264Config.idrPeriod = gop
        config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR
        # VBR with targetQuality (CQ mode) - similar to CRF
        config.rcParams.targetQuality = crf
        config.rcParams.averageBitRate = 0  # Uncapped
        config.rcParams.maxBitRate = 0  # Uncapped
        # Lookahead would deepen the pipeline beyond the buffer rings sized
        # below; disable it explicitly rather than trusting the preset.
        config.rcParams.rcFlags &= ~_RC_FLAG_ENABLE_LOOKAHEAD
        config.rcParams.lookaheadDepth = 0
        if tune_config is not None:
            tune_config(config)
        self._config = config

        dar_width, dar_height = dar_size if dar_size is not None else (width, height)
        init_params = NV_ENC_INITIALIZE_PARAMS()
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = codec_guid
        init_params.presetGUID = preset_guid
        init_params.encodeWidth = width
        init_params.encodeHeight = height
        init_params.darWidth = dar_width
        init_params.darHeight = dar_height
        init_params.frameRateNum = fps.numerator
        init_params.frameRateDen = fps.denominator
        init_params.enableEncodeAsync = 0  # Synchronous mode (Linux only)
        init_params.enablePTD = 1
        init_params.tuningInfo = tuning_info
        init_params.encodeConfig = ctypes.pointer(config)

        status = self._api.nvEncInitializeEncoder(self._encoder, byref(init_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(self._error(status, 'Failed to initialize encoder'))

    def _create_bitstream_buffers(self) -> None:
        for i in range(self._ring_size):
            bs_params = NV_ENC_CREATE_BITSTREAM_BUFFER()
            bs_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
            status = self._api.nvEncCreateBitstreamBuffer(self._encoder, byref(bs_params))
            if status != NV_ENC_SUCCESS:
                raise NvencError(self._error(status, f'Failed to create bitstream buffer {i}'))
            self._bitstream_buffers.append(bs_params.bitstreamBuffer)

    def register_input(self, register_params: NV_ENC_REGISTER_RESOURCE) -> c_void_p:
        """Register an input resource; the caller fills the type-specific fields.

        The session unregisters all registered resources at close().
        """
        status = self._api.nvEncRegisterResource(self._encoder, byref(register_params))
        if status != NV_ENC_SUCCESS:
            raise TextureFormatError(self._error(status, 'Failed to register input resource'))
        self._registered_resources.append(register_params.registeredResource)
        return register_params.registeredResource

    def submit(
        self,
        registered_resource: c_void_p,
        width: int,
        height: int,
        pitch: int,
        cleanup: Callable[[], None] | None = None,
    ) -> list[EncodedPacket]:
        """Map a registered input, submit it for encoding, drain ready output.

        The mapped input (and the optional cleanup callback) is retained
        until the bitstream produced from this submission has been consumed.

        Returns:
            Ready EncodedPackets; empty while frames are buffered for
            B-frame reordering.
        """
        if self._closed:
            raise EncoderNotInitialized('Encoder has been closed')
        if self._flushed:
            raise NvencError('Cannot encode after flush(); create a new encoder')

        map_params = NV_ENC_MAP_INPUT_RESOURCE()
        map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER
        map_params.registeredResource = registered_resource

        status = self._api.nvEncMapInputResource(self._encoder, byref(map_params))
        if status != NV_ENC_SUCCESS:
            if cleanup is not None:
                cleanup()
            raise NvencError(self._error(status, 'Failed to map resource'))

        mapped_resource = map_params.mappedResource
        mapped_format = map_params.mappedBufferFmt

        bs_idx = self._next_submit
        self._next_submit += 1

        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.inputWidth = width
        pic_params.inputHeight = height
        pic_params.inputPitch = pitch
        pic_params.inputBuffer = mapped_resource
        pic_params.outputBitstream = self._bitstream_buffers[bs_idx % self._ring_size]
        pic_params.bufferFmt = mapped_format
        pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
        pic_params.frameIdx = self._frame_idx
        pic_params.inputTimeStamp = self._frame_idx  # PTS = display order

        status = self._api.nvEncEncodePicture(self._encoder, byref(pic_params))
        if status not in (NV_ENC_SUCCESS, NV_ENC_ERR_NEED_MORE_INPUT):
            # Roll back the submission slot: nothing was submitted for it, so a
            # later flush must not lock its (never-filled) bitstream buffer.
            self._next_submit = bs_idx
            self._api.nvEncUnmapInputResource(self._encoder, mapped_resource)
            if cleanup is not None:
                cleanup()
            raise NvencError(self._error(status, f'Failed to encode frame {self._frame_idx}'))

        self._pending_inputs[bs_idx] = (mapped_resource, cleanup)
        self._frame_idx += 1

        if status == NV_ENC_ERR_NEED_MORE_INPUT:
            return []
        return self._read_pending(bs_idx)

    def _read_pending(self, up_to_idx: int) -> list[EncodedPacket]:
        """Read pending bitstream buffers in submission order up to up_to_idx."""
        result = []
        while self._next_read <= up_to_idx:
            buf = self._bitstream_buffers[self._next_read % self._ring_size]
            lock_bs = NV_ENC_LOCK_BITSTREAM()
            lock_bs.version = NV_ENC_LOCK_BITSTREAM_VER
            lock_bs.outputBitstream = buf
            lock_bs.doNotWait = 0

            status = self._api.nvEncLockBitstream(self._encoder, byref(lock_bs))
            if status != NV_ENC_SUCCESS:
                raise NvencError(self._error(status, 'nvEncLockBitstream failed'))

            try:
                data = ctypes.string_at(lock_bs.bitstreamBufferPtr, lock_bs.bitstreamSizeInBytes)
                if data:
                    result.append(
                        EncodedPacket(
                            data=data,
                            pts=lock_bs.outputTimeStamp,
                            dts=self._output_idx,
                            is_keyframe=lock_bs.pictureType == NV_ENC_PIC_TYPE_IDR,
                        )
                    )
                    self._output_idx += 1
            finally:
                self._api.nvEncUnlockBitstream(self._encoder, buf)
            # This submission's output is consumed; its input may be released.
            self._release_input(self._next_read)
            self._next_read += 1
        return result

    def _release_input(self, submit_idx: int) -> None:
        entry = self._pending_inputs.pop(submit_idx, None)
        if entry is None:
            return
        mapped_resource, cleanup = entry
        self._api.nvEncUnmapInputResource(self._encoder, mapped_resource)
        if cleanup is not None:
            cleanup()

    def flush(self) -> list[EncodedPacket]:
        """Send EOS and drain remaining packets. Idempotent.

        Returns:
            EncodedPackets for frames that were still in the reorder buffer.
        """
        if self._closed or self._encoder is None or self._flushed:
            return []
        self._flushed = True

        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS
        status = self._api.nvEncEncodePicture(self._encoder, byref(pic_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(self._error(status, 'Failed to submit end-of-stream'))

        packets = self._read_pending(self._next_submit - 1)
        for submit_idx in sorted(self._pending_inputs):
            self._release_input(submit_idx)
        return packets

    def close(self) -> None:
        """Release all session resources, draining any queued frames first."""
        if self._closed:
            return
        if self._encoder is not None and not self._flushed and self._pending_inputs:
            # Destroying the encoder while frames sit in its reorder queue is
            # undefined behavior (intermittent segfault in nvEncDestroyEncoder);
            # drain via EOS and discard the output. Best-effort: if the drain
            # fails, teardown must still proceed.
            try:
                self.flush()
            except Exception:
                pass
        self._closed = True

        if self._encoder:
            for submit_idx in sorted(self._pending_inputs):
                self._release_input(submit_idx)
            for registered in self._registered_resources:
                self._api.nvEncUnregisterResource(self._encoder, registered)
            self._registered_resources.clear()

            for buf in self._bitstream_buffers:
                self._api.nvEncDestroyBitstreamBuffer(self._encoder, buf)
            self._bitstream_buffers.clear()

            self._api.nvEncDestroyEncoder(self._encoder)
            self._encoder = None

    def _error(self, status: int, context: str) -> str:
        """Build an error message, including the driver's last-error detail."""
        detail = None
        if self._encoder:
            try:
                raw = self._api.nvEncGetLastErrorString(self._encoder)
                detail = raw.decode(errors='replace') if raw else None
            except Exception:
                detail = None
        return nvenc_status_message(status, context, detail=detail)
