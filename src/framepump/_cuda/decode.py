"""NVDEC decode plumbing: packet feeding, the frame index, decode sessions."""

from __future__ import annotations

import ctypes
from fractions import Fraction

import av
from av.bitstream import BitStreamFilterContext
import PyNvVideoCodec as nvc

from .._pyav import (
    FrameIndexPyAV,
    PyAVReader,
    VideoDecodeError,
    _discard_other_streams,
)
from .compat import cuda_ctx_pushed, retain_primary_context

# decoder support.
_NVDEC_CODECS = {
    'h264': nvc.cudaVideoCodec.H264,
    'hevc': nvc.cudaVideoCodec.HEVC,
    'av1': nvc.cudaVideoCodec.AV1,
    'vp9': nvc.cudaVideoCodec.VP9,
    'vp8': nvc.cudaVideoCodec.VP8,
    'mpeg1video': nvc.cudaVideoCodec.MPEG1,
    'mpeg2video': nvc.cudaVideoCodec.MPEG2,
    'mpeg4': nvc.cudaVideoCodec.MPEG4,
    'mjpeg': nvc.cudaVideoCodec.JPEG,
    'vc1': nvc.cudaVideoCodec.VC1,
}

# Codecs whose MP4/MKV packets need conversion to Annex-B start codes for
# the NVDEC parser
_ANNEXB_FILTERS = {'h264': 'h264_mp4toannexb', 'hevc': 'hevc_mp4toannexb'}

# Raw PyNvVideoCodec exceptions, wrapped into VideoDecodeError wherever the
# decoder is driven (unsupported profiles, reconfiguration limits, ...)
_NVC_ERRORS = (nvc.PyNvVCException, nvc.PyNvVCExceptionUnsupported)


class _PyAVPacketSource:
    """Demux packets with PyAV and present them as NVDEC PacketData.

    PyAV replaces PyNvDemuxer here: its seeking is reliable (PyNvDemuxer's
    Seek() segfaults outright on some driver/library combinations and
    mishandles edit-list files), and its bitstream filters convert AVCC
    packets to the Annex-B form the NVDEC parser expects.
    """

    def __init__(self, video_path: str, codec_name: str) -> None:
        self._container = av.open(video_path)
        self._stream = self._container.streams.video[0]
        _discard_other_streams(self._container, self._stream)
        self._codec_name = codec_name
        self._bsf = self._make_bsf()
        # The NVDEC parser copies the bitstream during Decode(), but hold the
        # most recent buffer anyway so its memory can never be reused early
        self._last_buf = None

    def _make_bsf(self):
        filter_name = _ANNEXB_FILTERS.get(self._codec_name)
        if filter_name is None:
            return None
        return BitStreamFilterContext(filter_name, self._stream)

    def seek(self, pts: int) -> None:
        """Seek to the keyframe at or before ``pts`` (stream time_base units)."""
        self._container.seek(pts, stream=self._stream, backward=True)
        # The filter buffers parameter sets; start it fresh after a jump
        self._bsf = self._make_bsf()

    def packets(self):
        """Yield NVDEC PacketData for the remaining packets.

        Packets without timestamps (raw elementary streams) are fed too —
        skipping them would starve the parser entirely; their PacketData pts
        stays 0 and callers must use positional access for such streams.
        """
        for packet in self._container.demux(self._stream):
            filtered = self._bsf.filter(packet) if self._bsf is not None else (packet,)
            for fp in filtered:
                data = bytes(fp)
                if not data:
                    continue
                buf = ctypes.create_string_buffer(data, len(data))
                self._last_buf = buf
                pd = nvc.PacketData()
                pd.bsl_data = ctypes.addressof(buf)
                pd.bsl = len(data)
                pts = fp.pts if fp.pts is not None else fp.dts
                pd.pts = pts if pts is not None else 0
                yield pd

    def close(self) -> None:
        self._container.close()


# ── Frame index ──────────────────────────────────────────────────────


class _CudaFrameIndex:
    """Frame index for NVDEC access, derived from the shared CPU packet index.

    ``FrameIndexPyAV`` stores display-order PTS as exact Fractions in
    seconds; NVDEC packet feeding and seeking work in raw integer stream
    PTS, so the values are converted back through the stream time base
    (exact — they were produced as integer PTS × time base). The Fraction
    PTS are kept for the CFR source map.
    """

    def __init__(self, video_path: str) -> None:
        reader = PyAVReader(video_path)
        try:
            time_base = reader.time_base
            base = FrameIndexPyAV(video_path, reader=reader)
            # Whether the container supports reliable seeking (raw bitstreams
            # and image-pipe formats do not); reuses the CPU reader's
            # detection and probe. Non-seekable sources decode from the start.
            self.seekable: bool = reader.seekable
        finally:
            reader.close()

        if base.pts_synthesized:
            # Timestampless streams (raw bitstreams): the synthesized PTS
            # never match decoder output, so PTS-based frame location would
            # silently return wrong frames.
            name = video_path if isinstance(video_path, str) else '<file-like>'
            raise RuntimeError(f'No valid packets found in {name}')

        self.frame_pts_frac: list[Fraction] = base.frame_pts
        self.frame_pts: list[int] = [int(p / time_base) for p in base.frame_pts]
        self.safe_seek_pts: list[int] = [int(p / time_base) for p in base.safe_seek_pts]
        self.frame_count: int = base.frame_count


# ── Decode session ───────────────────────────────────────────────────


class _NvDecSession:
    """PyAV packet source + PyNvDecoder for a single decode session.

    Created per iteration/access — not shared across clones or concurrent
    iterations.

    The decoder is bound to the device's *primary* CUDA context, retained for
    the session's lifetime. Without an explicit context, PyNvVideoCodec binds
    to whatever context state the creating thread happens to have, and the
    DLPack export then dies with CUDA_ERROR_INVALID_CONTEXT when a different
    thread (e.g. one holding a torch model) owns the process's CUDA state.
    """

    def __init__(
        self,
        video_path: str,
        gpu: int,
        codec,
        codec_name: str,
        output_color_type,
    ) -> None:
        self._path = video_path
        self._device = None
        self._ctx = None
        self._dec = None
        self._src = None
        self._src = _PyAVPacketSource(video_path, codec_name)
        try:
            self._device, self._ctx = retain_primary_context(gpu)
            self._dec = nvc.CreateDecoder(
                gpuid=gpu,
                codec=codec,
                cudacontext=int(self._ctx),
                cudastream=0,
                usedevicememory=True,
                outputColorType=output_color_type,
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )
        except _NVC_ERRORS as e:
            self.close()
            raise VideoDecodeError(video_path, 0, e) from e
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Drop the decoder and balance the primary-context retain."""
        if self._dec is not None:
            with cuda_ctx_pushed(self._ctx):
                self._dec = None
        if self._device is not None:
            from cuda.bindings import driver

            driver.cuDevicePrimaryCtxRelease(self._device)
            self._device = None
            self._ctx = None
        if self._src is not None:
            self._src.close()
            self._src = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _decode_all(self):
        """Feed every remaining packet and flush; yield (pts, frame).

        NVDEC/driver errors (unsupported profile, mid-stream reconfiguration
        beyond limits, ...) are wrapped so callers never see raw
        PyNvVideoCodec exceptions. Decode calls run under the session context
        (pushed per step, never across a yield, so no context state leaks
        into the consuming thread).
        """
        count = 0
        try:
            for pd in self._src.packets():
                with cuda_ctx_pushed(self._ctx):
                    decoded = [(f.getPTS(), f) for f in self._dec.Decode(pd)]
                for item in decoded:
                    count += 1
                    yield item
            empty = nvc.PacketData()
            while True:
                with cuda_ctx_pushed(self._ctx):
                    decoded = [(f.getPTS(), f) for f in self._dec.Decode(empty)]
                if not decoded:
                    break
                for item in decoded:
                    count += 1
                    yield item
        except _NVC_ERRORS as e:
            raise VideoDecodeError(self._path, count, e) from e

    def iter_from_start(self):
        """Decode all frames sequentially from the beginning.

        Yields (pts, frame) tuples in display order.
        """
        yield from self._decode_all()

    def iter_from_pts(self, start_pts: int):
        """Seek to start_pts and decode forward.

        Yields (pts, frame) tuples in display order, starting from the first
        frame with PTS >= start_pts.
        """
        self._src.seek(start_pts)
        reached = False
        for pts, f in self._decode_all():
            if not reached:
                if pts >= start_pts:
                    reached = True
                else:
                    continue
            yield pts, f


# ── Public class ─────────────────────────────────────────────────────



def _plane_layouts(frame, num_planes: int, min_row_bytes: int, height: int) -> list:
    """Get (device pointer, row pitch in bytes) for each plane of a decoded frame.

    Prefers the per-plane CUDA-array-interface views from ``frame.cuda()``.
    PyNvVideoCodec's views report strides in elements in current releases
    (a corrected release would report bytes); both units are accepted and the
    result is validated against the row's minimum byte width, so a wrong
    pitch raises instead of producing silently sheared output.

    Falls back to inferring the pitch from plane-pointer deltas when the
    views are unavailable — validated under the explicit assumption that the
    planes are allocated contiguously at a common pitch.
    """
    try:
        views = frame.cuda()
        cais = [view.__cuda_array_interface__ for view in views[:num_planes]]
        if len(cais) != num_planes:
            raise ValueError(f'Expected {num_planes} planes, got {len(cais)}')
    except Exception:
        return _plane_layouts_from_deltas(frame, num_planes, min_row_bytes, height)

    result = []
    for cai in cais:
        ptr = cai['data'][0]
        itemsize = int(cai['typestr'][-1])
        strides = cai.get('strides')
        if strides:
            s0 = strides[0]
            if itemsize == 1 or s0 < min_row_bytes:
                pitch = s0 if s0 >= min_row_bytes else s0 * itemsize
            else:
                # Ambiguous window (small widths): s0 could be a byte pitch,
                # or an element count that still exceeds the row byte width.
                # Corroborate with the plane-pointer delta; raise rather than
                # guess, per this function's contract.
                pitch = _disambiguate_pitch(frame, num_planes, s0, itemsize, height)
        else:
            pitch = min_row_bytes
        if pitch < min_row_bytes or pitch % itemsize:
            raise RuntimeError(
                f'Cannot determine plane pitch: stride {strides} (itemsize {itemsize}) '
                f'is inconsistent with a row width of {min_row_bytes} bytes'
            )
        result.append((ptr, pitch))
    return result


def _disambiguate_pitch(frame, num_planes: int, s0: int, itemsize: int, height: int) -> int:
    """Decide whether a reported stride is a byte pitch or an element count.

    Uses the luma→chroma plane-pointer delta (pitch * plane height for the
    contiguous allocations NVDEC produces) as the tiebreaker.
    """
    candidates = {s0, s0 * itemsize}
    if len(candidates) == 1:
        return s0
    if num_planes >= 2:
        try:
            delta = frame.GetPtrToPlane(1) - frame.GetPtrToPlane(0)
        except Exception:
            delta = None
        if delta is not None:
            matching = [c for c in sorted(candidates) if delta == c * height]
            if len(matching) == 1:
                return matching[0]
    raise RuntimeError(
        f'Cannot determine whether the reported stride {s0} (itemsize {itemsize}) '
        f'is a byte pitch or an element count; refusing to guess.'
    )


def _plane_layouts_from_deltas(frame, num_planes: int, min_row_bytes: int, height: int) -> list:
    """Infer a common plane pitch from consecutive plane-pointer deltas."""
    ptrs = [frame.GetPtrToPlane(i) for i in range(num_planes)]
    if num_planes == 1:
        return [(ptrs[0], min_row_bytes)]
    delta = ptrs[1] - ptrs[0]
    if delta <= 0 or delta % height:
        raise RuntimeError(
            f'Cannot infer plane pitch: plane delta {delta} is not a positive '
            f'multiple of the plane height {height} (planes not contiguous?)'
        )
    pitch = delta // height
    if pitch < min_row_bytes or pitch % 2:
        raise RuntimeError(
            f'Inferred plane pitch {pitch} is inconsistent with a row width of '
            f'{min_row_bytes} bytes'
        )
    return [(ptr, pitch) for ptr in ptrs]


# ── GPU RGB buffer with DLPack export ────────────────────────────────
