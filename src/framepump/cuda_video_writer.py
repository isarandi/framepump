"""CUDA video writer with GPU JPEG decoding and NVENC encoding.

Pipeline for 4:2:0 JPEGs: JPEG → nvJPEG → NVENC (zero-copy, IYUV format)
Pipeline for 4:4:4 JPEGs: JPEG → nvJPEG → NVENC (zero-copy, YUV444 format)

nvJPEG decodes directly into NVENC-registered device buffer - no GPU copies.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Union

import simplepyutils as spu
from cuda.bindings import driver

from ._cuda_compat import cuda_ctx_pushed, retain_primary_context
from ._h264_mux import H264PassthroughMuxer
from .encoder_config import EncoderConfig
from .nvenc._session import NvencEncodeSession
from .nvenc.bindings import (NV_ENC_BUFFER_FORMAT_IYUV, NV_ENC_BUFFER_FORMAT_NV16,
                             NV_ENC_BUFFER_FORMAT_YUV444, NV_ENC_DEVICE_TYPE_CUDA,
                             NV_ENC_H264_PROFILE_HIGH_422_GUID,
                             NV_ENC_H264_PROFILE_HIGH_444_GUID, NV_ENC_INPUT_IMAGE,
                             NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                             NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_REGISTER_RESOURCE,
                             NV_ENC_REGISTER_RESOURCE_VER)
from .nvenc.exceptions import NvencError
from .nvjpeg import NvjpegPhasedDecoder
from .nvjpeg.bindings import NVJPEG_CSS_420, NVJPEG_CSS_444
from .video_writing import AbstractVideoWriter, VideoOutput

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

    Ending a sequence to which no frame was written is a no-op: no output
    file is created.

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
            chroma: Target chroma subsampling for 4:4:4 JPEG input: '420' or
                '422' downsample the chroma planes before encoding; None or
                '444' keep the source subsampling (4:2:0 input is always
                encoded as 4:2:0, 4:4:4 input as 4:4:4).
        """
        if chroma not in (None, '420', '422', '444'):
            raise ValueError(
                f"chroma must be None, '420', '422' or '444', got {chroma!r}")
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




@dataclass(frozen=True)
class _Plane:
    """One plane inside a GPU buffer: byte offset, row pitch, allocated rows."""

    offset: int
    pitch: int
    rows: int

    @property
    def nbytes(self) -> int:
        return self.pitch * self.rows

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class _EncodeBufferLayout:
    """Byte layout of one NVENC input buffer.

    Plane offsets, pitches and the total size all follow the PADDED encode
    dimensions: NVENC locates the chroma planes from the registered pitch and
    encode height, so the decode side must place its output with this exact
    geometry. The display-sized decode extents live in `_ScratchLayout` and
    the writer's `_width`/`_height`; mixing up the padded and display
    families of numbers is what used to corrupt chroma at heights not
    divisible by 16 (e.g. 1080).
    """

    buffer_format: int
    size: int
    y: _Plane
    u: _Plane  # the interleaved UV plane for NV16
    v: _Plane | None  # None for NV16 (V is interleaved into `u`)


@dataclass(frozen=True)
class _ScratchLayout:
    """Scratch buffer layout for 4:4:4 → 4:2:0/4:2:2 chroma downsampling.

    Holds the full-resolution U/V planes decoded from the JPEG and, for the
    4:2:2 path, staging planes for the half-width resize before NV16
    interleaving. Extents here are display-sized on purpose: this buffer is
    never registered with NVENC.
    """

    size: int
    full_u: _Plane
    full_v: _Plane
    resized_u: _Plane | None
    resized_v: _Plane | None
    chroma_width: int
    chroma_height: int


def _build_encode_layout(
    enc_width: int, enc_height: int, subsampling: int, downsample_to: str | None
) -> _EncodeBufferLayout:
    """Compute the NVENC input buffer layout from the padded encode dimensions."""
    # Align pitch to 256 bytes for optimal GPU access
    pitch = ((enc_width + 255) // 256) * 256
    y = _Plane(0, pitch, enc_height)
    if subsampling == NVJPEG_CSS_420 or downsample_to == '420':
        # IYUV: UV pitch must be exactly Y pitch / 2 (NVENC infers it from Y)
        uv_pitch = pitch // 2
        u = _Plane(y.end, uv_pitch, enc_height // 2)
        v = _Plane(u.end, uv_pitch, enc_height // 2)
        return _EncodeBufferLayout(NV_ENC_BUFFER_FORMAT_IYUV, v.end, y, u, v)
    if downsample_to == '422':
        # NV16: interleaved UV plane with the same pitch and rows as Y
        uv = _Plane(y.end, pitch, enc_height)
        return _EncodeBufferLayout(NV_ENC_BUFFER_FORMAT_NV16, uv.end, y, uv, None)
    if subsampling == NVJPEG_CSS_444:
        u = _Plane(y.end, pitch, enc_height)
        v = _Plane(u.end, pitch, enc_height)
        return _EncodeBufferLayout(NV_ENC_BUFFER_FORMAT_YUV444, v.end, y, u, v)
    raise NvencError(f'Unsupported chroma subsampling: {_css_name(subsampling)}')


def _build_scratch_layout(
    width: int, height: int, y_pitch: int, downsample_to: str
) -> _ScratchLayout:
    """Compute the downsample scratch layout from the display dimensions."""
    full_u = _Plane(0, y_pitch, height)
    full_v = _Plane(full_u.end, y_pitch, height)
    chroma_width = width // 2
    if downsample_to == '422':
        half_pitch = ((chroma_width + 255) // 256) * 256
        resized_u = _Plane(full_v.end, half_pitch, height)
        resized_v = _Plane(resized_u.end, half_pitch, height)
        return _ScratchLayout(
            resized_v.end, full_u, full_v, resized_u, resized_v, chroma_width, height)
    return _ScratchLayout(full_v.end, full_u, full_v, None, None, chroma_width, height // 2)


_CSS_NAMES = {0: '4:4:4', 1: '4:2:2', 2: '4:2:0', 3: '4:4:0', 4: '4:1:1', 5: '4:1:0', 6: 'gray'}


def _css_name(subsampling: int) -> str:
    return _CSS_NAMES.get(subsampling, f'unknown ({subsampling})')


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
        self._video_output = video_output
        self._format = format

        if not isinstance(video_output, (str, Path)) and format is None:
            raise ValueError('format is required when writing to a file-like object')

        # Muxer created on first frame, once the frame geometry is known
        self._muxer: H264PassthroughMuxer | None = None

        # CUDA/NVENC state
        self._cuda_ctx = None
        self._cuda_device = None
        self._owns_cuda_ctx = False
        self._jpeg_decoder: NvjpegPhasedDecoder | None = None
        self._session: NvencEncodeSession | None = None
        self._width = 0
        self._height = 0
        self._encode_width = 0
        self._encode_height = 0
        self._sps_crop_right = 0
        self._sps_crop_bottom = 0

        # YUV input ring: decode targets that double as NVENC input buffers
        self._yuv_buffers: list[int] = []  # CUdeviceptr list
        self._registered: list = []  # NVENC registered-resource handles
        self._num_yuv_buffers = 0  # Set from the session's ring size
        self._layout: _EncodeBufferLayout | None = None  # NVENC input buffer layout
        self._scratch: _ScratchLayout | None = None  # Downsample scratch layout
        self._subsampling = None  # NVJPEG_CSS_420 or NVJPEG_CSS_444
        self._downsample_to = None  # '420' or '422' when downsampling from 4:4:4
        self._uv_scratch: int = 0  # Scratch buffer for full-res U/V before downsample
        self._npp_ctx = None  # NppStreamContext for resize calls
        self._current_buffer = 0  # Which buffer to decode into (round-robin)

        # CUDA stream for async GPU decode (transfer + device)
        self._decode_stream = None

        self._frame_idx = 0
        self._closed = False

    def _init_cuda(self) -> None:
        """Initialize CUDA context and stream for async decode.

        Reuses the caller's current context if one exists; otherwise retains
        the primary context of the configured device. The context is made
        current only for the duration of writer calls.
        """
        err, = driver.cuInit(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise NvencError(f'Failed to initialize CUDA: {err}')

        err, ctx = driver.cuCtxGetCurrent()
        if ctx is not None and int(ctx) != 0:
            self._cuda_ctx = ctx
            self._owns_cuda_ctx = False
        else:
            try:
                self._cuda_device, self._cuda_ctx = retain_primary_context(self._gpu)
            except RuntimeError as e:
                raise NvencError(str(e)) from e
            self._owns_cuda_ctx = True

        # Create stream for async decode (allows overlap with encode)
        with cuda_ctx_pushed(self._cuda_ctx):
            err, self._decode_stream = driver.cuStreamCreate(0)
        if err != driver.CUresult.CUDA_SUCCESS:
            self._release_owned_ctx()
            raise NvencError(f'Failed to create decode stream: {err}')

    def _release_owned_ctx(self) -> None:
        if self._owns_cuda_ctx and self._cuda_device is not None:
            driver.cuDevicePrimaryCtxRelease(self._cuda_device)
        self._cuda_ctx = None
        self._cuda_device = None
        self._owns_cuda_ctx = False

    def _prepare_layouts(self, width: int, height: int, subsampling: int) -> None:
        """Fix the frame geometry and compute the GPU buffer layouts."""
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

        self._layout = _build_encode_layout(
            self._encode_width, self._encode_height, subsampling, self._downsample_to)
        if self._downsample_to:
            self._scratch = _build_scratch_layout(
                width, height, self._layout.y.pitch, self._downsample_to)

    def _alloc_buffers(self) -> None:
        """Allocate the YUV input ring plus downsample scratch space.

        The ring size comes from the encode session: NVENC may hold that many
        submitted inputs before producing output, and the decode target for
        the next frame occupies exactly the session's one slot of headroom,
        so the session requirement and the decode-pipeline requirement
        coincide.
        """
        self._num_yuv_buffers = self._session.ring_size
        for i in range(self._num_yuv_buffers):
            err, devptr = driver.cuMemAlloc(self._layout.size)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to allocate YUV device buffer {i}: {err}')
            self._yuv_buffers.append(int(devptr))
            self._memset_planes(int(devptr), self._layout)

        if self._downsample_to:
            # Scratch shared across ring buffers (decode stream is serialized)
            err, devptr = driver.cuMemAlloc(self._scratch.size)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to allocate UV scratch buffer: {err}')
            self._uv_scratch = int(devptr)
            from .npp_bindings import make_npp_stream_context
            self._npp_ctx = make_npp_stream_context(self._gpu)

    @staticmethod
    def _memset_planes(base: int, layout: _EncodeBufferLayout) -> None:
        """Define every byte of the buffer: black luma, neutral chroma.

        The padding rows (display height..encode height) get encoded, so they
        must hold deterministic values rather than uninitialized memory.
        """
        for plane, value in ((layout.y, 0), (layout.u, 128), (layout.v, 128)):
            if plane is None:
                continue
            err, = driver.cuMemsetD8(base + plane.offset, value, plane.nbytes)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise NvencError(f'Failed to initialize YUV buffer: {err}')

    def _init_session(self) -> None:
        """Open the NVENC encode session, customized for JPEG-decoded YUV input."""
        if self._downsample_to == '422':
            chroma_idc = 2
        elif self._subsampling == NVJPEG_CSS_444 and not self._downsample_to:
            chroma_idc = 3
        else:
            chroma_idc = 1
        self._compute_sps_crop(chroma_idc)

        qp = self._encoder_config.crf

        def tune(config) -> None:
            # CONSTQP mode - pure quality control, no bitrate targets needed.
            # B-frames use higher QP (factor 1.25 + offset 1.25, like FFmpeg default)
            config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP
            config.rcParams.constQP.qpIntra = qp
            config.rcParams.constQP.qpInterP = qp
            config.rcParams.constQP.qpInterB = int(qp * 1.25 + 1.25 + 0.5)
            h264 = config.encodeCodecConfig.h264Config
            if chroma_idc == 2:
                config.profileGUID = NV_ENC_H264_PROFILE_HIGH_422_GUID
                h264.chromaFormatIDC = 2
            elif chroma_idc == 3:
                config.profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID
                h264.chromaFormatIDC = 3
            # Signal full-range BT.601 YUV (what JPEG uses), not TV range
            vui = h264.h264VUIParameters
            vui.videoSignalTypePresentFlag = 1
            vui.videoFullRangeFlag = 1
            vui.colourDescriptionPresentFlag = 1
            vui.colourPrimaries = 6  # SMPTE 170M (BT.601)
            vui.transferCharacteristics = 6
            vui.colourMatrix = 6

        self._session = NvencEncodeSession(
            device_type=NV_ENC_DEVICE_TYPE_CUDA,
            device=int(self._cuda_ctx),
            width=self._encode_width,
            height=self._encode_height,
            fps=self._fps_frac,
            crf=qp,
            gop=self._encoder_config.gop,
            bframes=self._encoder_config.bframes,
            dar_size=(self._width, self._height),
            tune_config=tune,
        )

    def _compute_sps_crop(self, chroma_idc: int) -> None:
        """Compute SPS crop offsets (in crop units, not pixels)."""
        dw = self._encode_width - self._width
        dh = self._encode_height - self._height
        if not (dw or dh):
            self._sps_crop_right = 0
            self._sps_crop_bottom = 0
            return

        # CropUnitX/Y depend on chroma_format_idc (progressive: frame_mbs_only=1)
        cu_x = 1 if chroma_idc == 3 else 2  # 444→1, 420/422→2
        cu_y = 2 if chroma_idc == 1 else 1  # 420→2, 422/444→1
        if dw % cu_x or dh % cu_y:
            fmt = ['', '4:2:0', '4:2:2', '4:4:4'][chroma_idc]
            parts = []
            if cu_x > 1 and self._width % 2:
                parts.append(f'width={self._width} (must be even)')
            if cu_y > 1 and self._height % 2:
                parts.append(f'height={self._height} (must be even)')
            raise ValueError(
                f'H.264 {fmt} requires even {" and ".join(parts)} — '
                f'cannot represent exact dimensions {self._width}x{self._height}')
        self._sps_crop_right = dw // cu_x
        self._sps_crop_bottom = dh // cu_y

    def _register_buffers(self) -> None:
        """Register the YUV ring with NVENC, using the decode side's layout."""
        for devptr in self._yuv_buffers:
            reg = NV_ENC_REGISTER_RESOURCE()
            reg.version = NV_ENC_REGISTER_RESOURCE_VER
            reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
            reg.width = self._encode_width
            reg.height = self._encode_height
            reg.pitch = self._layout.y.pitch
            reg.resourceToRegister = devptr
            reg.bufferUsage = NV_ENC_INPUT_IMAGE
            reg.bufferFormat = self._layout.buffer_format
            self._registered.append(self._session.register_input(reg))

    def _open(self) -> None:
        """Create the output muxer, once the frame geometry is known."""
        if self._downsample_to == '422':
            pix_fmt = 'yuv422p'
        elif self._subsampling == NVJPEG_CSS_444 and not self._downsample_to:
            pix_fmt = 'yuv444p'
        else:
            pix_fmt = 'yuv420p'

        self._muxer = H264PassthroughMuxer(
            self._video_output,
            fps=self._fps_frac,
            width=self._encode_width,
            height=self._encode_height,
            bframes=self._encoder_config.bframes,
            pix_fmt=pix_fmt,
            audio_source_path=self._audio_source_path,
            format=self._format,
            stream_options={'strict': 'experimental'},
        )

    def write_jpeg(self, jpeg_data: bytes) -> None:
        """Decode JPEG and encode to video using pipelined GPU processing.

        Pipeline overlaps decode of frame N with encode of frame N-1:
        - Sync to ensure previous decode is done
        - Start decode of frame N (async)
        - Encode frame N-1 (while GPU decodes frame N in parallel)
        """
        if self._closed:
            raise RuntimeError('Writer is closed')

        if self._cuda_ctx is None:
            self._init_cuda()

        with cuda_ctx_pushed(self._cuda_ctx):
            # Initialize on first frame
            if self._jpeg_decoder is None:
                self._jpeg_decoder = NvjpegPhasedDecoder(gpu=None)
                width, height, subsampling = self._jpeg_decoder.parse(jpeg_data)
                self._prepare_layouts(width, height, subsampling)
                self._init_session()
                self._alloc_buffers()
                self._register_buffers()
                self._open()
                self._jpeg_decoder.decode_host()
            else:
                # Wait for the previous frame's async decode to complete: its
                # buffer is submitted to NVENC below (which reads it right
                # away), and the phased decoder's slots are recycled every
                # other frame.
                driver.cuStreamSynchronize(self._decode_stream)
                width, height, subsampling = self._jpeg_decoder.parse(jpeg_data)
                self._check_frame_consistent(width, height, subsampling)
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

    def _check_frame_consistent(self, width: int, height: int, subsampling: int) -> None:
        """All frames must match the geometry the GPU buffers were sized for."""
        if (width, height) != (self._width, self._height):
            raise ValueError(
                f'JPEG dimensions {width}x{height} do not match initial frame '
                f'dimensions {self._width}x{self._height}')
        if subsampling != self._subsampling:
            raise ValueError(
                f'JPEG chroma subsampling {_css_name(subsampling)} does not match '
                f'initial frame subsampling {_css_name(self._subsampling)}')

    def _decode_gpu_into_buffer(self, buf_idx: int) -> None:
        """Transfer + device decode into specified buffer (async on decode stream)."""
        base = self._yuv_buffers[buf_idx]
        stream = int(self._decode_stream)
        layout = self._layout

        # Transfer from pinned buffer to device (async)
        self._jpeg_decoder.decode_transfer(stream)

        if self._downsample_to:
            # Decode 4:4:4 JPEG, then downsample U/V chroma.
            # Y goes directly into the encode buffer; U/V go to scratch, then resize.
            scratch = self._scratch
            self._jpeg_decoder.decode_device(
                base + layout.y.offset,
                self._uv_scratch + scratch.full_u.offset,
                self._uv_scratch + scratch.full_v.offset,
                layout.y.pitch, scratch.full_u.pitch, scratch.full_v.pitch,
                stream,
            )
            self._npp_ctx.hStream = stream
            from .npp_bindings import resize_plane_8u
            if self._downsample_to == '420':
                # Half width, half height, straight into the IYUV chroma planes
                for src, dst in ((scratch.full_u, layout.u), (scratch.full_v, layout.v)):
                    resize_plane_8u(
                        self._uv_scratch + src.offset, src.pitch, self._width, self._height,
                        base + dst.offset, dst.pitch,
                        scratch.chroma_width, scratch.chroma_height,
                        ctx=self._npp_ctx,
                    )
            else:  # '422': half width, full height, staged then interleaved into NV16
                for src, dst in (
                    (scratch.full_u, scratch.resized_u),
                    (scratch.full_v, scratch.resized_v),
                ):
                    resize_plane_8u(
                        self._uv_scratch + src.offset, src.pitch, self._width, self._height,
                        self._uv_scratch + dst.offset, dst.pitch,
                        scratch.chroma_width, scratch.chroma_height,
                        ctx=self._npp_ctx,
                    )
                from .npp_bindings import interleave_uv
                interleave_uv(
                    self._uv_scratch + scratch.resized_u.offset, scratch.resized_u.pitch,
                    self._uv_scratch + scratch.resized_v.offset, scratch.resized_v.pitch,
                    base + layout.u.offset, layout.u.pitch,
                    scratch.chroma_width, scratch.chroma_height,
                    stream=stream,
                )
        else:
            # Native 4:2:0 or 4:4:4: decode straight into the encode buffer planes
            self._jpeg_decoder.decode_device(
                base + layout.y.offset,
                base + layout.u.offset,
                base + layout.v.offset,
                layout.y.pitch, layout.u.pitch, layout.v.pitch,
                stream,
            )

    def _encode_buffer(self, buf_idx: int) -> None:
        """Submit a decoded YUV buffer to NVENC and mux whatever output is ready."""
        packets = self._session.submit(
            self._registered[buf_idx],
            width=self._encode_width,
            height=self._encode_height,
            pitch=self._layout.y.pitch,
        )
        self._mux_packets(packets)

    def _mux_packets(self, packets) -> None:
        for packet in packets:
            # Rewrite SPS NALs to add frame_cropping_rect if dimensions were
            # padded to macroblock alignment (only IDR packets carry SPS)
            if packet.is_keyframe and (self._sps_crop_right or self._sps_crop_bottom):
                packet.data = _patch_sps_crop(
                    packet.data, self._sps_crop_right, self._sps_crop_bottom)
            self._muxer.mux(packet)

    def close(self) -> None:
        """Encode the last frame, flush, finalize the output, free GPU state.

        On error, the output is discarded (no file at the final path) and the
        error propagates. If no frame was written, no file is created.
        """
        if self._closed:
            return
        self._closed = True

        try:
            with self._ctx_guard():
                if self._muxer is not None:
                    # Encode the last frame (still in the previous buffer)
                    if self._frame_idx > 0:
                        driver.cuStreamSynchronize(self._decode_stream)
                        last_buf = (self._current_buffer - 1) % self._num_yuv_buffers
                        self._encode_buffer(last_buf)
                    self._mux_packets(self._session.flush())
                    self._muxer.close()
        except BaseException:
            if self._muxer is not None:
                self._muxer.abort()
            raise
        finally:
            self._muxer = None
            self._cleanup_gpu_resources()

    def _ctx_guard(self):
        """Push the writer's context for the duration of a call, if initialized."""
        if self._cuda_ctx is not None:
            return cuda_ctx_pushed(self._cuda_ctx)
        return nullcontext()

    def _abort(self) -> None:
        """Abort the write: free GPU state, discard output, delete the temp file."""
        if self._closed:
            return
        self._closed = True

        try:
            self._cleanup_gpu_resources()
        finally:
            if self._muxer is not None:
                self._muxer.abort()
                self._muxer = None

    def _cleanup_gpu_resources(self) -> None:
        """Release all GPU resources (CUDA, nvJPEG, NVENC)."""
        if self._cuda_ctx is None:
            return  # CUDA was never initialized (no frame written)

        with cuda_ctx_pushed(self._cuda_ctx):
            # An aborted write can still have an async decode in flight;
            # cuStreamDestroy does not wait for queued work, so synchronize
            # before freeing the buffers and nvJPEG state the decode writes to.
            if self._decode_stream:
                driver.cuStreamSynchronize(self._decode_stream)

            # The session unmaps and unregisters the input buffers, so it must
            # be closed before the underlying device memory is freed.
            if self._session is not None:
                self._session.close()
                self._session = None
            self._registered.clear()

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

        self._release_owned_ctx()

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()
