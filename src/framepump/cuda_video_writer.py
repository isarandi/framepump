"""CUDA video writer with GPU JPEG decoding and NVENC encoding.

Pipeline for 4:2:0 JPEGs: JPEG → nvJPEG → NVENC (zero-copy, IYUV format)
Pipeline for 4:4:4 JPEGs: JPEG → nvJPEG → NVENC (zero-copy, YUV444 format)

nvJPEG decodes directly into NVENC-registered device buffer - no GPU copies.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager
from ctypes import byref, c_uint32, c_void_p
from fractions import Fraction
from pathlib import Path
from typing import Any, Union

import av
import simplepyutils as spu
from cuda.bindings import driver

from ._cuda_compat import cuCtxCreate
from ._temp_file import TempFile
from .encoder_config import EncoderConfig
from .video_writing import VideoOutput
from .nvenc.bindings import (GUID, NVENCAPI_VERSION, NV_ENC_BUFFER_FORMAT_IYUV,
                             NV_ENC_BUFFER_FORMAT_NV16, NV_ENC_BUFFER_FORMAT_YUV444,
                             NV_ENC_CODEC_H264_GUID, NV_ENC_CONFIG,
                             NV_ENC_CONFIG_VER, NV_ENC_CREATE_BITSTREAM_BUFFER,
                             NV_ENC_CREATE_BITSTREAM_BUFFER_VER, NV_ENC_DEVICE_TYPE_CUDA,
                             NV_ENC_ERR_NEED_MORE_INPUT,
                             NV_ENC_H264_PROFILE_HIGH_422_GUID,
                             NV_ENC_H264_PROFILE_HIGH_444_GUID,
                             NV_ENC_INITIALIZE_PARAMS, NV_ENC_INITIALIZE_PARAMS_VER,
                             NV_ENC_INPUT_IMAGE, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                             NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER,
                             NV_ENC_MAP_INPUT_RESOURCE, NV_ENC_MAP_INPUT_RESOURCE_VER,
                             NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
                             NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_PIC_FLAG_EOS,
                             NV_ENC_PIC_PARAMS, NV_ENC_PIC_PARAMS_VER, NV_ENC_PIC_STRUCT_FRAME,
                             NV_ENC_PIC_TYPE_IDR, NV_ENC_PRESET_CONFIG, NV_ENC_PRESET_CONFIG_VER,
                             NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY,
                             NV_ENC_REGISTER_RESOURCE, NV_ENC_REGISTER_RESOURCE_VER,
                             NV_ENC_SUCCESS, NvencAPI)
from .nvenc.exceptions import NvencError, nvenc_status_message
from .nvenc.presets import DEFAULT_PRESET
from .nvjpeg import NvjpegPhasedDecoder
from .nvjpeg.bindings import NVJPEG_CSS_420, NVJPEG_CSS_444
from .video_writing import AbstractVideoWriter

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
#  H.264 SPS bitstream surgery: add frame_cropping_rect
# ---------------------------------------------------------------------------
# NVENC doesn't auto-set frame_cropping when encodeWidth > display width for
# all chroma formats (notably 422). We parse the SPS NAL, find the
# frame_cropping_flag bit, set it to 1, and insert crop offsets.

class _BitReader:
    """Read individual bits from a byte buffer."""
    __slots__ = ('_data', '_pos', '_len')

    def __init__(self, data: bytes | bytearray):
        self._data = data
        self._pos = 0
        self._len = len(data) * 8

    @property
    def remaining(self) -> int:
        return self._len - self._pos

    def read(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self._data[self._pos >> 3] >> (7 - (self._pos & 7))) & 1)
            self._pos += 1
        return v

    def read_ue(self) -> int:
        zeros = 0
        while self.read(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.read(zeros) if zeros else 0

    def read_se(self) -> int:
        v = self.read_ue()
        return ((v + 1) >> 1) * (1 if v & 1 else -1) if v else 0


def _ue_bits(val: int) -> list[int]:
    """Encode an unsigned Exp-Golomb value as a list of bits."""
    v = val + 1
    n = v.bit_length()
    return [0] * (n - 1) + [(v >> i) & 1 for i in range(n - 1, -1, -1)]


def _nal_to_rbsp(data: bytes) -> bytearray:
    """Remove emulation prevention bytes (00 00 03 xx → 00 00 xx)."""
    out = bytearray()
    i = 0
    while i < len(data):
        if (i + 2 < len(data) and data[i] == 0 and data[i + 1] == 0
                and data[i + 2] == 3):
            out.append(0)
            out.append(0)
            i += 3
        else:
            out.append(data[i])
            i += 1
    return out


def _rbsp_to_nal(data: bytes | bytearray) -> bytes:
    """Insert emulation prevention bytes where needed."""
    out = bytearray()
    zeros = 0
    for b in data:
        if zeros == 2 and b <= 3:
            out.append(3)
            zeros = 0
        zeros = zeros + 1 if b == 0 else 0
        out.append(b)
    return bytes(out)


def _skip_scaling_list(r: _BitReader, size: int) -> None:
    last = 8
    nxt = 8
    for _ in range(size):
        if nxt != 0:
            delta = r.read_se()
            nxt = (last + delta + 256) % 256
        last = nxt if nxt != 0 else last


def _find_sps_crop_pos(rbsp: bytes | bytearray) -> tuple[int, int]:
    """Parse SPS RBSP to locate frame_cropping_flag.

    Returns (flag_bit_pos, after_crop_bit_pos).
    flag_bit_pos: bit index of frame_cropping_flag.
    after_crop_bit_pos: bit index after the crop offsets (or flag+1 if flag=0).
    """
    r = _BitReader(rbsp)
    r.read(8)  # NAL header byte

    profile_idc = r.read(8)
    r.read(8)  # constraint flags + reserved
    r.read(8)  # level_idc
    r.read_ue()  # seq_parameter_set_id

    if profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134):
        cfi = r.read_ue()  # chroma_format_idc
        if cfi == 3:
            r.read(1)  # separate_colour_plane_flag
        r.read_ue()  # bit_depth_luma_minus8
        r.read_ue()  # bit_depth_chroma_minus8
        r.read(1)  # qpprime_y_zero_transform_bypass_flag
        if r.read(1):  # seq_scaling_matrix_present_flag
            for i in range(12 if cfi == 3 else 8):
                if r.read(1):  # scaling_list_present
                    _skip_scaling_list(r, 16 if i < 6 else 64)

    r.read_ue()  # log2_max_frame_num_minus4
    poc_type = r.read_ue()
    if poc_type == 0:
        r.read_ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif poc_type == 1:
        r.read(1)  # delta_pic_order_always_zero_flag
        r.read_se()  # offset_for_non_ref_pic
        r.read_se()  # offset_for_top_to_bottom_field
        for _ in range(r.read_ue()):
            r.read_se()

    r.read_ue()  # max_num_ref_frames
    r.read(1)  # gaps_in_frame_num_value_allowed_flag
    r.read_ue()  # pic_width_in_mbs_minus1
    r.read_ue()  # pic_height_in_map_units_minus1

    if not r.read(1):  # frame_mbs_only_flag
        r.read(1)  # mb_adaptive_frame_field_flag

    r.read(1)  # direct_8x8_inference_flag

    # frame_cropping_flag is at current position
    flag_pos = r._pos
    old_flag = r.read(1)
    if old_flag:
        r.read_ue()  # skip old crop_left
        r.read_ue()  # skip old crop_right
        r.read_ue()  # skip old crop_top
        r.read_ue()  # skip old crop_bottom
    return flag_pos, r._pos


def _set_sps_crop(nal_data: bytes, crop_right: int, crop_bottom: int) -> bytes:
    """Modify an SPS NAL unit to set frame_cropping_rect.

    nal_data starts with the NAL header byte (no start code).
    crop_right/crop_bottom are in SPS crop units (not pixels).
    """
    rbsp = _nal_to_rbsp(nal_data)
    flag_pos, after_pos = _find_sps_crop_pos(rbsp)

    # Convert RBSP to bit array
    bits = []
    for b in rbsp:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)

    # Splice: [before crop flag] + [new crop] + [after old crop data]
    new_crop = [1] + _ue_bits(0) + _ue_bits(crop_right) + _ue_bits(0) + _ue_bits(crop_bottom)
    bits = bits[:flag_pos] + new_crop + bits[after_pos:]

    # Bits → bytes
    new_rbsp = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] if i + j < len(bits) else 0)
        new_rbsp.append(byte)
    return _rbsp_to_nal(new_rbsp)


def _patch_sps_crop(data: bytes, crop_right: int, crop_bottom: int) -> bytes:
    """Scan Annex B bitstream for SPS NALs and add crop rect."""
    if crop_right == 0 and crop_bottom == 0:
        return data

    buf = bytearray(data)
    result = bytearray()
    i = 0
    while i < len(buf):
        # Find start code
        if (i + 3 < len(buf) and buf[i] == 0 and buf[i + 1] == 0
                and buf[i + 2] == 0 and buf[i + 3] == 1):
            sc_len = 4
        elif (i + 2 < len(buf) and buf[i] == 0 and buf[i + 1] == 0
              and buf[i + 2] == 1):
            sc_len = 3
        else:
            result.append(buf[i])
            i += 1
            continue

        nal_start = i + sc_len
        # Find next start code (end of this NAL)
        j = nal_start + 1
        while j + 2 < len(buf):
            if buf[j] == 0 and buf[j + 1] == 0 and (buf[j + 2] == 0 or buf[j + 2] == 1):
                break
            j += 1
        else:
            j = len(buf)
        nal_end = j

        nal_bytes = bytes(buf[nal_start:nal_end])
        nal_type = nal_bytes[0] & 0x1F if nal_bytes else 0

        if nal_type == 7:  # SPS
            nal_bytes = _set_sps_crop(nal_bytes, crop_right, crop_bottom)

        result.extend(buf[i:i + sc_len])
        result.extend(nal_bytes)
        i = nal_end

    return bytes(result)


class JpegVideoWriterCUDA(AbstractVideoWriter[bytes], AbstractContextManager['JpegVideoWriterCUDA']):
    """Zero-copy JPEG to video writer using nvJPEG decoder and NVENC encoder.

    Decodes JPEG to YUV420 on GPU with nvJPEG and encodes with NVENC using
    the IYUV (I420) format - all without CPU-GPU data transfers.

    Example:
        >>> with JpegVideoWriterCUDA('output.mp4', fps=30) as writer:
        ...     for jpeg_bytes in jpeg_frames:
        ...         writer.append_data(jpeg_bytes)
    """

    def __init__(
        self,
        video_path: PathLike | None = None,
        fps: float | Fraction | None = None,
        audio_source_path: PathLike | None = None,
        queue_size: int = 32,
        encoder_config: EncoderConfig | None = None,
        gpu: int = 0,
        chroma: str | None = None,
    ) -> None:
        """Create a new CUDA JPEG video writer.

        Args:
            video_path: Output path for the first video sequence (optional).
            fps: Frame rate (required if video_path is provided).
            audio_source_path: Path to copy audio from, for the first sequence.
            queue_size: Unused, present for API compatibility with VideoWriter.
            encoder_config: Encoder configuration (crf, preset, bframes, gop).
            gpu: CUDA device ordinal (default 0).
            chroma: Chroma subsampling override ('420' or '444'). If None,
                auto-detected from the first JPEG frame.
        """
        del queue_size
        self._writer: _CudaSequenceWriter | None = None
        self._accepts_new_frames: bool = False
        self._default_fps = fps
        self._default_encoder_config = encoder_config
        self._gpu = gpu
        self._chroma = chroma

        if video_path is not None:
            if fps is None:
                raise ValueError('fps must be provided if video_path is provided')
            self.start_sequence(video_path, fps, audio_source_path=audio_source_path,
                                encoder_config=encoder_config)

    @property
    def accepts_new_frames(self) -> bool:
        return self._accepts_new_frames

    def start_sequence(
        self,
        video_output: VideoOutput,
        fps: float | Fraction | None = None,
        audio_source_path: PathLike | None = None,
        gpu: bool | int = True,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
    ) -> SequenceContext:
        del gpu  # Always uses GPU; device set in constructor
        if fps is None:
            if self._default_fps is None:
                raise ValueError('fps must be provided if not set in constructor')
            fps = self._default_fps

        if encoder_config is None:
            encoder_config = self._default_encoder_config

        if isinstance(video_output, (str, Path)):
            spu.ensure_parent_dir_exists(video_output)

        if self._writer is not None:
            self._writer.close()
        self._writer = _CudaSequenceWriter(
            video_output,
            fps=fps,
            audio_source_path=audio_source_path,
            encoder_config=encoder_config,
            format=format,
            gpu=self._gpu,
            chroma=self._chroma,
        )
        self._accepts_new_frames = True
        return SequenceContext(self)

    def append_data(self, data: bytes) -> None:
        """Append JPEG data to the video."""
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before appending data')
        assert self._writer is not None
        self._writer.write_jpeg(data)

    def end_sequence(self) -> None:
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before ending')
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._accepts_new_frames = False

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._accepts_new_frames = False

    def _abort(self) -> None:
        if self._writer is not None:
            self._writer._abort()
            self._writer = None
        self._accepts_new_frames = False

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any, **kwargs: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()


class SequenceContext(AbstractContextManager['SequenceContext']):
    """Context for a video sequence being written."""

    def __init__(self, multiwriter: AbstractVideoWriter) -> None:
        self.multiwriter = multiwriter

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.multiwriter.end_sequence()




class _CudaSequenceWriter(AbstractContextManager['_CudaSequenceWriter']):
    """Internal writer: nvJPEG decode (YUV) → NVENC encode via device pointer.

    Supports both 4:2:0 (IYUV) and 4:4:4 (YUV444) chroma subsampling.
    """

    def __init__(
        self,
        video_output: VideoOutput,
        fps: float | Fraction,
        audio_source_path: PathLike | None = None,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
        gpu: int = 0,
        chroma: str | None = None,
    ) -> None:
        self._fps_frac = (
            fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        )
        self._audio_source_path = audio_source_path
        self._encoder_config = encoder_config if encoder_config is not None else EncoderConfig()
        self._gpu = gpu
        self._target_chroma = chroma  # None/'native', '420', or '422'

        # Determine output mode: file-like or path
        if isinstance(video_output, (str, Path)):
            self._temp_file = TempFile(video_output)
            self._file_output = None
            self._format = format or Path(video_output).suffix.lstrip('.')
        else:
            # File-like object
            self._temp_file = None
            self._file_output = video_output
            if format is None:
                raise ValueError("format is required when writing to a file-like object")
            self._format = format

        # State initialized on first frame
        self._output_container: av.container.OutputContainer | None = None
        self._audio_input_container: av.container.InputContainer | None = None
        self._video_stream: av.stream.Stream | None = None
        self._audio_stream: av.stream.Stream | None = None
        self._audio_time_base: Fraction = Fraction(1)
        self._audio_pkts: Iterator[av.Packet] = iter([])
        self._audio_putback: av.Packet | None = None

        # CUDA/NVENC state
        self._cuda_ctx = None
        self._owns_cuda_ctx = False
        self._jpeg_decoder: NvjpegPhasedDecoder | None = None
        self._api = None
        self._nvenc_encoder = None
        self._width = 0
        self._height = 0

        # YUV buffers for ping-pong pipeline
        self._yuv_buffers: list[int] = []  # CUdeviceptr list
        self._registered_resources: list = []  # Registered resources
        self._yuv_pitch = 0  # Y plane pitch (256-byte aligned)
        self._uv_pitch = 0  # U/V plane pitch (for 4:2:0, half of Y pitch)
        self._subsampling = None  # NVJPEG_CSS_420 or NVJPEG_CSS_444
        self._downsample_to = None  # '420' or '422' when downsampling from 4:4:4
        self._uv_scratch: int = 0  # Scratch buffer for full-res U/V before downsample
        self._npp_ctx = None  # NppStreamContext for resize calls
        self._current_buffer = 0  # Which buffer to decode into (round-robin)

        # CUDA stream for async GPU decode (transfer + device)
        self._decode_stream = None

        # YUV buffers for decode pipeline (ping-pong)
        self._num_yuv_buffers = 2

        # Multiple bitstream buffers — sized in _init_nvenc after config is known
        self._num_buffers = 0
        self._bitstream_buffers: list = []
        self._next_submit_buffer = 0  # Next buffer to use for encoding
        self._next_read_buffer = 0  # Next buffer to read output from (in submission order)

        # Incremental muxing state
        self._next_dts = 0  # DTS counter for decode-order muxing

        self._frame_idx = 0
        self._closed = False

    def _init_cuda(self) -> None:
        """Initialize CUDA context and stream for async decode."""
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to initialize CUDA: {err}')

        err, ctx = driver.cuCtxGetCurrent()
        if ctx is not None and int(ctx) != 0:
            self._cuda_ctx = ctx
            self._owns_cuda_ctx = False
        else:
            err, device = driver.cuDeviceGet(self._gpu)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to get CUDA device {self._gpu}: {err}')

            err, self._cuda_ctx = cuCtxCreate(0, device)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to create CUDA context: {err}')
            self._owns_cuda_ctx = True

        # Create stream for async decode (allows overlap with encode)
        err, self._decode_stream = driver.cuStreamCreate(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            if self._owns_cuda_ctx:
                driver.cuCtxDestroy(self._cuda_ctx)
                self._cuda_ctx = None
            raise NvencError(f'Failed to create decode stream: {err}')

    def _alloc_buffers(self, width: int, height: int, subsampling: int) -> None:
        """Allocate two GPU buffers for pipeline (ping-pong)."""
        self._width = width
        self._height = height
        self._subsampling = subsampling
        # Determine if we need to downsample from 4:4:4
        if subsampling == NVJPEG_CSS_444 and self._target_chroma in ('420', '422'):
            self._downsample_to = self._target_chroma
        else:
            self._downsample_to = None

        # H.264 macroblocks are 16x16. Pad encode dimensions so NVENC doesn't
        # silently crop. We rewrite the SPS to add frame_cropping_rect so
        # decoders display the original dimensions.
        self._encode_width = ((width + 15) // 16) * 16
        self._encode_height = ((height + 15) // 16) * 16
        enc_h = self._encode_height

        # Align pitch to 256 bytes for optimal GPU access
        self._yuv_pitch = ((self._encode_width + 255) // 256) * 256

        if subsampling == NVJPEG_CSS_420 or self._downsample_to == '420':
            # IYUV layout for NVENC: UV pitch must be exactly Y pitch / 2
            # (NVENC infers UV pitch from Y pitch for IYUV format)
            self._uv_pitch = self._yuv_pitch // 2
            buffer_size = self._yuv_pitch * enc_h + 2 * self._uv_pitch * (enc_h // 2)
        elif self._downsample_to == '422':
            # NV16 layout: Y plane (pitch * height) + interleaved UV (pitch * height)
            # UV pitch = Y pitch (same width as Y, since UV is interleaved: W/2 * 2 = W)
            self._uv_pitch = self._yuv_pitch
            buffer_size = self._yuv_pitch * enc_h * 2
        elif subsampling == NVJPEG_CSS_444:
            # YUV444 layout: Y (pitch * height) + U (pitch * height) + V (pitch * height)
            self._uv_pitch = self._yuv_pitch  # Same pitch for all planes
            buffer_size = self._yuv_pitch * enc_h * 3
        else:
            raise NvencError(f'Unsupported chroma subsampling: {subsampling}')

        # Allocate buffers for pipeline (need multiple for B-frame lookahead)
        for i in range(self._num_yuv_buffers):
            err, devptr = driver.cuMemAlloc(buffer_size)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to allocate YUV device buffer {i}: {err}')
            self._yuv_buffers.append(int(devptr))

        if self._downsample_to:
            # Scratch space for full-res U and V planes before downsample,
            # plus space for two half-width resized planes (for 422 interleave).
            # Shared across ping-pong buffers (decode stream is serialized).
            half_pitch = ((width // 2 + 255) // 256) * 256
            self._scratch_half_pitch = half_pitch
            scratch_size = self._yuv_pitch * height * 2 + half_pitch * height * 2
            err, devptr = driver.cuMemAlloc(scratch_size)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to allocate UV scratch buffer: {err}')
            self._uv_scratch = int(devptr)
            from .npp_bindings import make_npp_stream_context
            self._npp_ctx = make_npp_stream_context(self._gpu)

    def _init_nvenc(self, width: int, height: int, subsampling: int) -> None:
        """Initialize NVENC encoder for YUV input."""
        self._api = NvencAPI()

        # Open session
        params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
        params.device = int(self._cuda_ctx)
        params.apiVersion = NVENCAPI_VERSION

        encoder = c_void_p()
        status = self._api.nvEncOpenEncodeSessionEx(byref(params), byref(encoder))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to open encode session'))
        self._nvenc_encoder = encoder

        codec_guid = NV_ENC_CODEC_H264_GUID
        preset_guid = NV_ENC_PRESET_P4_GUID

        # Get preset config (use Ex version with tuning info for P1-P7 presets)
        preset_config = NV_ENC_PRESET_CONFIG()
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER
        self._api.nvEncGetEncodePresetConfigEx(
            self._nvenc_encoder, codec_guid, preset_guid,
            NV_ENC_TUNING_INFO_HIGH_QUALITY, byref(preset_config)
        )

        # Configure
        config = NV_ENC_CONFIG()
        ctypes.memmove(byref(config), byref(preset_config.presetCfg), ctypes.sizeof(NV_ENC_CONFIG))
        config.version = NV_ENC_CONFIG_VER
        config.gopLength = self._encoder_config.gop
        config.frameIntervalP = self._encoder_config.bframes + 1
        # CONSTQP mode - pure quality control, no bitrate targets needed
        # QP directly controls quantization (lower = higher quality)
        # B-frames use higher QP (factor 1.25 + offset 1.25, like FFmpeg default)
        from .nvenc.bindings import NV_ENC_PARAMS_RC_CONSTQP
        config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP
        qp = self._encoder_config.crf
        config.rcParams.constQP.qpIntra = qp
        config.rcParams.constQP.qpInterP = qp
        config.rcParams.constQP.qpInterB = int(qp * 1.25 + 1.25 + 0.5)

        # Set chroma format and profile for H.264
        if self._downsample_to == '422':
            config.profileGUID = NV_ENC_H264_PROFILE_HIGH_422_GUID
            config.encodeCodecConfig.h264Config.chromaFormatIDC = 2
        elif subsampling == NVJPEG_CSS_444 and not self._downsample_to:
            config.profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID
            config.encodeCodecConfig.h264Config.chromaFormatIDC = 3

        # Signal full-range YUV (JPEG uses full range 0-255, not TV range 16-235)
        h264_vui = config.encodeCodecConfig.h264Config.h264VUIParameters
        h264_vui.videoSignalTypePresentFlag = 1  # Enable video signal type info
        h264_vui.videoFullRangeFlag = 1  # Full range (0-255) not TV range (16-235)
        # Signal BT.601 color space (JPEG standard uses BT.601 matrix coefficients)
        h264_vui.colourDescriptionPresentFlag = 1
        h264_vui.colourPrimaries = 6          # SMPTE 170M (BT.601)
        h264_vui.transferCharacteristics = 6   # SMPTE 170M (BT.601)
        h264_vui.colourMatrix = 6              # SMPTE 170M (BT.601)

        # Initialize encoder
        init_params = NV_ENC_INITIALIZE_PARAMS()
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = codec_guid
        init_params.presetGUID = preset_guid
        init_params.encodeWidth = self._encode_width
        init_params.encodeHeight = self._encode_height
        init_params.darWidth = self._width
        init_params.darHeight = self._height

        # Compute SPS crop offsets (in crop units, not pixels)
        dw = self._encode_width - self._width
        dh = self._encode_height - self._height
        if dw or dh:
            # CropUnitX/Y depend on chroma_format_idc (progressive: frame_mbs_only=1)
            chroma_idc = config.encodeCodecConfig.h264Config.chromaFormatIDC
            cu_x = 1 if chroma_idc == 3 else 2  # 444→1, 420/422→2
            cu_y = 2 if chroma_idc == 1 else 1   # 420→2, 422/444→1
            if dw % cu_x or dh % cu_y:
                fmt = ['', '4:2:0', '4:2:2', '4:4:4'][chroma_idc]
                need_even_w = cu_x > 1
                need_even_h = cu_y > 1
                parts = []
                if need_even_w and self._width % 2:
                    parts.append(f'width={self._width} (must be even)')
                if need_even_h and self._height % 2:
                    parts.append(f'height={self._height} (must be even)')
                raise ValueError(
                    f'H.264 {fmt} requires even {" and ".join(parts)} — '
                    f'cannot represent exact dimensions {self._width}x{self._height}')
            self._sps_crop_right = dw // cu_x
            self._sps_crop_bottom = dh // cu_y
        else:
            self._sps_crop_right = 0
            self._sps_crop_bottom = 0
        init_params.frameRateNum = self._fps_frac.numerator
        init_params.frameRateDen = self._fps_frac.denominator
        init_params.enableEncodeAsync = 0
        init_params.enablePTD = 1
        init_params.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY
        init_params.encodeConfig = ctypes.pointer(config)

        status = self._api.nvEncInitializeEncoder(self._nvenc_encoder, byref(init_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to initialize encoder'))

        # Compute exact bitstream buffer count from finalized config.
        # NVENC holds at most (frameIntervalP - 1 + lookaheadDepth) frames
        # before producing output. We need one extra since we submit before
        # reading, so the maximum in-flight count is frameIntervalP + lookahead.
        enable_lookahead = (config.rcParams.rcFlags >> 5) & 1
        lookahead = config.rcParams.lookaheadDepth if enable_lookahead else 0
        self._num_buffers = config.frameIntervalP + lookahead

        for i in range(self._num_buffers):
            bs_params = NV_ENC_CREATE_BITSTREAM_BUFFER()
            bs_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
            status = self._api.nvEncCreateBitstreamBuffer(self._nvenc_encoder, byref(bs_params))
            if status != NV_ENC_SUCCESS:
                raise NvencError(
                    nvenc_status_message(status, f'Failed to create bitstream buffer {i}'))
            self._bitstream_buffers.append(bs_params.bitstreamBuffer)

        # Determine buffer format
        if self._downsample_to == '422':
            buffer_format = NV_ENC_BUFFER_FORMAT_NV16
        elif subsampling == NVJPEG_CSS_420 or self._downsample_to == '420':
            buffer_format = NV_ENC_BUFFER_FORMAT_IYUV
        elif subsampling == NVJPEG_CSS_444:
            buffer_format = NV_ENC_BUFFER_FORMAT_YUV444
        else:
            raise NvencError(f'Unsupported chroma subsampling: {subsampling}')

        # Register all YUV buffers with NVENC
        for i, devptr in enumerate(self._yuv_buffers):
            reg = NV_ENC_REGISTER_RESOURCE()
            reg.version = NV_ENC_REGISTER_RESOURCE_VER
            reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
            reg.width = self._encode_width
            reg.height = self._encode_height
            reg.pitch = self._yuv_pitch
            reg.resourceToRegister = devptr
            reg.bufferUsage = NV_ENC_INPUT_IMAGE
            reg.bufferFormat = buffer_format

            status = self._api.nvEncRegisterResource(self._nvenc_encoder, byref(reg))
            if status != NV_ENC_SUCCESS:
                raise NvencError(
                    nvenc_status_message(status, f'Failed to register YUV resource {i}'))
            self._registered_resources.append(reg.registeredResource)

    def _open(self) -> None:
        """Open output container."""
        # Start DTS negative to ensure dts <= pts; negative_cts_offsets avoids edit list
        self._next_dts = -self._encoder_config.bframes

        muxer_options = {'movflags': 'negative_cts_offsets', 'use_editlist': '0'}
        if self._temp_file is not None:
            self._output_container = av.open(
                os.fspath(self._temp_file.temp_path), 'w', format=self._format,
                options=muxer_options)
        else:
            self._output_container = av.open(
                self._file_output, 'w', format=self._format, options=muxer_options)

        self._video_stream = self._output_container.add_stream('h264', rate=self._fps_frac)
        self._video_stream.width = self._encode_width
        self._video_stream.height = self._encode_height
        if self._downsample_to == '422':
            self._video_stream.pix_fmt = 'yuv422p'
        elif self._subsampling == NVJPEG_CSS_444 and not self._downsample_to:
            self._video_stream.pix_fmt = 'yuv444p'
        else:
            self._video_stream.pix_fmt = 'yuv420p'
        self._video_stream.codec_context.options = {'strict': 'experimental'}

        # Audio setup
        if self._audio_source_path is not None:
            try:
                self._audio_input_container = av.open(str(self._audio_source_path))
                if self._audio_input_container.streams.audio:
                    src_audio = self._audio_input_container.streams.audio[0]
                    self._audio_stream = self._output_container.add_stream(template=src_audio)
                    self._audio_time_base = src_audio.time_base
                    self._audio_pkts = (
                        pkt for pkt in self._audio_input_container.demux(src_audio)
                        if pkt.dts is not None
                    )
            except Exception:
                if self._audio_input_container:
                    self._audio_input_container.close()
                self._audio_input_container = None

    def write_jpeg(self, jpeg_data: bytes) -> None:
        """Decode JPEG and encode to video using pipelined GPU processing.

        Pipeline overlaps decode of frame N with encode of frame N-1:
        - Sync to ensure previous decode is done
        - Start decode of frame N (async)
        - Encode frame N-1 (while GPU decodes frame N in parallel)
        """
        if self._closed:
            raise RuntimeError('Writer is closed')

        # Initialize on first frame
        if self._jpeg_decoder is None:
            self._init_cuda()
            self._jpeg_decoder = NvjpegPhasedDecoder(gpu=None)
            width, height, subsampling = self._jpeg_decoder.parse(jpeg_data)
            self._alloc_buffers(width, height, subsampling)
            self._init_nvenc(width, height, subsampling)
            self._open()
            self._jpeg_decoder.decode_host()
        else:
            # Sync BEFORE decode_host: previous frame's decode_transfer is async
            # and reads from nvJPEG's internal pinned buffer. decode_host overwrites
            # that buffer, so we must wait for the transfer to complete first.
            driver.cuStreamSynchronize(self._decode_stream)
            self._jpeg_decoder.parse(jpeg_data)
            self._jpeg_decoder.decode_host()

        buf_idx = self._current_buffer
        prev_buf = (buf_idx - 1) % self._num_yuv_buffers

        # Start async GPU decode into current buffer (returns immediately)
        self._decode_gpu_into_buffer(buf_idx)

        # Encode previous frame while GPU decodes current frame
        if self._frame_idx > 0:
            self._encode_buffer(prev_buf)

        # Advance to next buffer (round-robin)
        self._current_buffer = (buf_idx + 1) % self._num_yuv_buffers
        self._frame_idx += 1

    def _decode_gpu_into_buffer(self, buf_idx: int) -> None:
        """Transfer + device decode into specified buffer (async on decode stream)."""
        devptr = self._yuv_buffers[buf_idx]
        stream = int(self._decode_stream)

        # Transfer from pinned buffer to device (async)
        self._jpeg_decoder.decode_transfer(stream)

        # Compute plane pointers based on subsampling
        if self._subsampling == NVJPEG_CSS_420:
            # I420 layout: Y, then U (half size), then V (half size)
            y_ptr = devptr
            u_ptr = y_ptr + self._yuv_pitch * self._height
            v_ptr = u_ptr + self._uv_pitch * (self._height // 2)
            self._jpeg_decoder.decode_device(
                y_ptr, u_ptr, v_ptr,
                self._yuv_pitch, self._uv_pitch, self._uv_pitch,
                stream,
            )
        elif self._downsample_to:
            # Decode 4:4:4 JPEG, then downsample U/V chroma.
            # Y goes directly into encode buffer; U/V go to scratch, then resize.
            y_ptr = devptr
            scratch_plane_size = self._yuv_pitch * self._height
            scratch_u = self._uv_scratch
            scratch_v = scratch_u + scratch_plane_size
            self._jpeg_decoder.decode_device(
                y_ptr, scratch_u, scratch_v,
                self._yuv_pitch, self._yuv_pitch, self._yuv_pitch,
                stream,
            )
            self._npp_ctx.hStream = stream
            from .npp_bindings import resize_plane_8u
            chroma_w = self._width // 2
            if self._downsample_to == '420':
                # 4:2:0: half width, half height → IYUV planar layout
                chroma_h = self._height // 2
                dst_u = y_ptr + self._yuv_pitch * self._height
                dst_v = dst_u + self._uv_pitch * chroma_h
                resize_plane_8u(
                    scratch_u, self._yuv_pitch, self._width, self._height,
                    dst_u, self._uv_pitch, chroma_w, chroma_h,
                    ctx=self._npp_ctx,
                )
                resize_plane_8u(
                    scratch_v, self._yuv_pitch, self._width, self._height,
                    dst_v, self._uv_pitch, chroma_w, chroma_h,
                    ctx=self._npp_ctx,
                )
            else:  # '422'
                # 4:2:2: half width, full height → NV16 semi-planar layout
                chroma_h = self._height
                hp = self._scratch_half_pitch
                # Resize into separate area after full-res scratch
                resized_base = self._uv_scratch + scratch_plane_size * 2
                resized_u = resized_base
                resized_v = resized_base + hp * chroma_h
                resize_plane_8u(
                    scratch_u, self._yuv_pitch, self._width, self._height,
                    resized_u, hp, chroma_w, chroma_h,
                    ctx=self._npp_ctx,
                )
                resize_plane_8u(
                    scratch_v, self._yuv_pitch, self._width, self._height,
                    resized_v, hp, chroma_w, chroma_h,
                    ctx=self._npp_ctx,
                )
                # Interleave U and V into NV16 UV plane
                uv_ptr = y_ptr + self._yuv_pitch * self._height
                from .npp_bindings import interleave_uv
                interleave_uv(
                    resized_u, hp,
                    resized_v, hp,
                    uv_ptr, self._uv_pitch,
                    chroma_w, chroma_h,
                    stream=stream,
                )
        else:  # NVJPEG_CSS_444 (native)
            # YUV444 layout: Y, U, V all same size
            plane_size = self._yuv_pitch * self._height
            y_ptr = devptr
            u_ptr = y_ptr + plane_size
            v_ptr = u_ptr + plane_size
            self._jpeg_decoder.decode_device(
                y_ptr, u_ptr, v_ptr,
                self._yuv_pitch, self._yuv_pitch, self._yuv_pitch,
                stream,
            )

    def _encode_buffer(self, buf_idx: int) -> None:
        """Encode frame. When output is ready, read all pending buffers in submission order."""
        bs_idx, need_more = self._submit_frame_for_encoding(buf_idx)

        if not need_more:
            # Output is ready - read all buffers in submission order up to and including this one
            while self._next_read_buffer <= bs_idx:
                self._read_bitstream_buffer(self._next_read_buffer)
                self._next_read_buffer += 1

    def _submit_frame_for_encoding(self, buf_idx: int) -> tuple[int, bool]:
        """Submit YUV buffer to NVENC. Returns (bitstream_buffer_idx, need_more_input)."""
        # Assign a bitstream buffer (use frame index for simpler tracking)
        bs_idx = self._next_submit_buffer
        self._next_submit_buffer += 1
        output_buffer = self._bitstream_buffers[bs_idx % self._num_buffers]

        # Map the resource for the specified buffer
        map_params = NV_ENC_MAP_INPUT_RESOURCE()
        map_params.version = NV_ENC_MAP_INPUT_RESOURCE_VER
        map_params.registeredResource = self._registered_resources[buf_idx]

        status = self._api.nvEncMapInputResource(self._nvenc_encoder, byref(map_params))
        if status != NV_ENC_SUCCESS:
            raise NvencError(nvenc_status_message(status, 'Failed to map resource'))

        mapped_resource = map_params.mappedResource
        mapped_format = map_params.mappedBufferFmt

        encode_frame_idx = self._frame_idx - 1  # We're encoding the previous frame

        try:
            pic_params = NV_ENC_PIC_PARAMS()
            pic_params.version = NV_ENC_PIC_PARAMS_VER
            pic_params.inputWidth = self._encode_width
            pic_params.inputHeight = self._encode_height
            pic_params.inputPitch = self._yuv_pitch
            pic_params.inputBuffer = mapped_resource
            pic_params.outputBitstream = output_buffer
            pic_params.bufferFmt = mapped_format
            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
            pic_params.frameIdx = encode_frame_idx
            pic_params.inputTimeStamp = encode_frame_idx

            status = self._api.nvEncEncodePicture(self._nvenc_encoder, byref(pic_params))
            if status != NV_ENC_SUCCESS and status != NV_ENC_ERR_NEED_MORE_INPUT:
                raise NvencError(
                    nvenc_status_message(status, f'Failed to encode frame {encode_frame_idx}'))
            need_more_input = (status == NV_ENC_ERR_NEED_MORE_INPUT)
        finally:
            self._api.nvEncUnmapInputResource(self._nvenc_encoder, mapped_resource)

        return bs_idx, need_more_input

    def _read_bitstream_buffer(self, bs_idx: int) -> None:
        """Read encoded data from bitstream buffer and mux immediately."""
        output_buffer = self._bitstream_buffers[bs_idx % self._num_buffers]

        lock_bs = NV_ENC_LOCK_BITSTREAM()
        lock_bs.version = NV_ENC_LOCK_BITSTREAM_VER
        lock_bs.outputBitstream = output_buffer
        lock_bs.doNotWait = 0

        status = self._api.nvEncLockBitstream(self._nvenc_encoder, byref(lock_bs))
        if status != NV_ENC_SUCCESS:
            raise NvencError(
                f'nvEncLockBitstream failed: {nvenc_status_message(status)}'
            )

        try:
            if lock_bs.bitstreamSizeInBytes > 0:
                data = ctypes.string_at(lock_bs.bitstreamBufferPtr, lock_bs.bitstreamSizeInBytes)
                nvenc_pts = lock_bs.outputTimeStamp  # NVENC's assigned display timestamp
                is_keyframe = lock_bs.pictureType == NV_ENC_PIC_TYPE_IDR

                # Rewrite SPS NALs to add frame_cropping_rect if dimensions
                # were padded to macroblock alignment (only IDR packets have SPS)
                if is_keyframe and (self._sps_crop_right or self._sps_crop_bottom):
                    data = _patch_sps_crop(
                        data, self._sps_crop_right, self._sps_crop_bottom)

                # Compute video time for audio interleaving
                video_time = self._next_dts / self._fps_frac

                # Interleave: write audio packets up to current video time
                if (self._audio_putback is not None
                        and self._audio_putback.dts * self._audio_time_base <= video_time):
                    self._audio_putback.stream = self._audio_stream
                    self._output_container.mux(self._audio_putback)
                    self._audio_putback = None
                if self._audio_putback is None:
                    for audio_pkt in self._audio_pkts:
                        if audio_pkt.dts * self._audio_time_base > video_time:
                            self._audio_putback = audio_pkt
                            break
                        audio_pkt.stream = self._audio_stream
                        self._output_container.mux(audio_pkt)

                # Mux video packet immediately
                packet = av.Packet(data)
                packet.stream = self._video_stream
                packet.pts = nvenc_pts
                packet.dts = self._next_dts
                packet.time_base = 1 / self._fps_frac
                packet.is_keyframe = is_keyframe
                self._output_container.mux(packet)

                self._next_dts += 1
        finally:
            self._api.nvEncUnlockBitstream(self._nvenc_encoder, output_buffer)

    def _flush_encoder(self) -> None:
        """Flush remaining B-frames after EOS."""
        if self._nvenc_encoder is None:
            return

        # Send EOS to signal end of input
        pic_params = NV_ENC_PIC_PARAMS()
        pic_params.version = NV_ENC_PIC_PARAMS_VER
        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS
        self._api.nvEncEncodePicture(self._nvenc_encoder, byref(pic_params))

        # After EOS, read all remaining buffers in submission order
        while self._next_read_buffer < self._next_submit_buffer:
            self._read_bitstream_buffer(self._next_read_buffer)
            self._next_read_buffer += 1

    def close(self) -> None:
        """Flush encoder, close containers, release all resources."""
        if self._closed:
            return
        self._closed = True

        if self._output_container is not None:
            try:
                # Encode the last frame (still in the previous buffer)
                if self._frame_idx > 0:
                    driver.cuStreamSynchronize(self._decode_stream)
                    last_buf = (self._current_buffer - 1) % self._num_yuv_buffers
                    self._encode_buffer(last_buf)

                self._flush_encoder()

                if self._audio_putback is not None:
                    self._audio_putback.stream = self._audio_stream
                    self._output_container.mux(self._audio_putback)
                    self._audio_putback = None
                for audio_pkt in self._audio_pkts:
                    audio_pkt.stream = self._audio_stream
                    self._output_container.mux(audio_pkt)
            finally:
                if self._output_container:
                    self._output_container.close()
                if self._audio_input_container:
                    self._audio_input_container.close()

        # Handle temp file: finalize only if frames were actually muxed, else cleanup
        if self._temp_file is not None:
            if self._next_read_buffer > 0:
                self._temp_file.finalize()
            else:
                self._temp_file.cleanup()
            self._temp_file = None

        # Now release all GPU resources
        self._cleanup_gpu_resources()

    def _abort(self) -> None:
        """Abort write - close resources without flushing, delete temp file."""
        if self._closed:
            return
        self._closed = True

        self._cleanup_resources()
        if self._temp_file is not None:
            self._temp_file.cleanup()

    def _cleanup_gpu_resources(self) -> None:
        """Release all GPU resources (CUDA, nvJPEG, NVENC)."""
        # Clean up NVENC
        if self._nvenc_encoder:
            for reg_resource in self._registered_resources:
                self._api.nvEncUnregisterResource(self._nvenc_encoder, reg_resource)
            self._registered_resources.clear()

            for buf in self._bitstream_buffers:
                self._api.nvEncDestroyBitstreamBuffer(self._nvenc_encoder, buf)
            self._bitstream_buffers.clear()

            self._api.nvEncDestroyEncoder(self._nvenc_encoder)
            self._nvenc_encoder = None

        # Clean up CUDA stream
        if self._decode_stream:
            driver.cuStreamDestroy(self._decode_stream)
            self._decode_stream = None

        # Clean up YUV buffers
        for devptr in self._yuv_buffers:
            driver.cuMemFree(devptr)
        self._yuv_buffers.clear()

        # Clean up UV scratch buffer (chroma downsampling)
        if self._uv_scratch:
            driver.cuMemFree(self._uv_scratch)
            self._uv_scratch = 0

        # Clean up nvJPEG
        if self._jpeg_decoder:
            self._jpeg_decoder.close()
            self._jpeg_decoder = None

        # Clean up CUDA context (only if we created it)
        if self._owns_cuda_ctx and self._cuda_ctx:
            driver.cuCtxDestroy(self._cuda_ctx)
            self._cuda_ctx = None

    def _cleanup_resources(self) -> None:
        """Release all GPU and container resources."""
        self._cleanup_gpu_resources()

        # Clean up containers
        if self._output_container:
            self._output_container.close()
            self._output_container = None
        if self._audio_input_container:
            self._audio_input_container.close()
            self._audio_input_container = None

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()
