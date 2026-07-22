"""PyAV-based video I/O for high-performance decoding and encoding.

This module provides persistent video readers using PyAV (Python bindings to FFmpeg
libraries), eliminating the subprocess overhead of spawning FFmpeg processes.

All time calculations use fractions.Fraction for exact rational arithmetic,
avoiding floating-point precision loss in PTS/timestamp handling.
"""

from __future__ import annotations

import bisect
from collections.abc import Generator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Union

import av
import numpy as np
from numpy.typing import DTypeLike, NDArray

PathLike = Union[str, Path]


class FramePumpError(Exception):
    """Base class for all FramePump exceptions."""

    pass


class VideoDecodeError(FramePumpError):
    """Error during video decoding, typically due to corrupt/truncated data."""

    def __init__(self, path, frame_count: int, original_error: Exception):
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        self.frame_count = frame_count
        self.original_error = original_error
        name = self.path.name if isinstance(self.path, Path) else '<file-like>'
        msg = (
            f'Corrupt or truncated video: {name}\n'
            f'Decoded {frame_count} frames before encountering invalid data.\n'
            f'Original error: {type(original_error).__name__}: {original_error}'
        )
        super().__init__(msg)


class VideoEncodeError(FramePumpError):
    """Error during video encoding."""

    def __init__(
        self,
        path: PathLike,
        frame_count: int,
        original_error: Exception,
        *,
        resolution: tuple[int, int] | None = None,
        codec: str | None = None,
    ):
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        self.frame_count = frame_count
        self.original_error = original_error
        self.resolution = resolution
        self.codec = codec
        name = self.path.name if isinstance(self.path, Path) else '<file-like>'

        # Build informative message
        parts = []
        # Detect NVENC small frame error
        if (
            codec
            and 'nvenc' in codec
            and resolution
            and (resolution[0] < 150 or resolution[1] < 50)
        ):
            parts.append(
                f'NVENC frame size too small: {resolution[0]}x{resolution[1]} '
                f'(minimum ~145x49 for h264_nvenc)'
            )
        else:
            parts.append(f'Failed to encode video: {name}')
            if resolution:
                parts.append(f'Resolution: {resolution[0]}x{resolution[1]}')
        if codec:
            parts.append(f'Codec: {codec}')
        parts.append(f'Encoded {frame_count} frames before error')
        parts.append(f'Original error: {type(original_error).__name__}: {original_error}')
        super().__init__('\n'.join(parts))


class NoAudioStreamError(FramePumpError):
    """Raised when audio is expected but not found."""

    def __init__(self, path):
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        name = self.path.name if isinstance(self.path, Path) else '<file-like>'
        super().__init__(f'No audio stream found in {name}')


class NoVideoStreamError(FramePumpError):
    """Raised when a file contains no video stream (e.g. audio-only files)."""

    def __init__(self, path):
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        name = self.path.name if isinstance(self.path, Path) else '<file-like>'
        super().__init__(f'No video stream found in {name}')


class UnsupportedCodecError(FramePumpError):
    """Raised when this FFmpeg build has no decoder for the video stream."""

    def __init__(self, path):
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        name = self.path.name if isinstance(self.path, Path) else '<file-like>'
        super().__init__(
            f'No decoder available for the video stream in {name} '
            f'(codec not supported by this FFmpeg build)'
        )


class FilterConfigError(FramePumpError):
    """Raised when filter graph configuration fails."""

    pass


# Lossless repack targets for semi-planar NVDEC download formats
_SEMIPLANAR_TO_PLANAR = {
    'nv12': 'yuv420p',
    'nv16': 'yuv422p',
    'p010le': 'yuv420p10le',
    'p016le': 'yuv420p16le',
}


def _discard_other_streams(container, keep) -> None:
    """Discard all streams except ``keep`` at the demuxer level.

    Skips the cost of synchronizing streams that are never read, and makes
    truncated interleaved files recover as many video packets as the ffmpeg
    CLI does: without this, a corrupt sample in an unrelated (e.g. audio)
    stream ends demuxing early for the video stream too.
    """
    from av.stream import Discard

    for other in container.streams:
        if other.index != keep.index:
            other.discard = Discard.all


def _make_hwaccel(gpu: bool | int):
    """Build the PyAV HWAccel spec for NVDEC decoding, or None for CPU.

    Software fallback is disabled on purpose: gpu=True must either really
    decode on the GPU or fail loudly, never silently decode on the CPU.
    """
    if not gpu:
        return None
    from av.codec.hwaccel import HWAccel

    device = str(gpu) if type(gpu) is int else None  # noqa: E721 (bool excluded on purpose)
    return HWAccel(device_type='cuda', device=device, allow_software_fallback=False)


class PyAVReader:
    """Persistent video reader using PyAV.

    Keeps the container open for fast seeking and decoding. Use Fraction
    arithmetic for all time calculations to avoid floating-point precision loss.

    Example:
        >>> reader = PyAVReader('video.mp4')
        >>> print(reader.fps, reader.duration, reader.resolution)
        >>> for frame in reader.decode_frames():
        ...     process(frame)
        >>> reader.close()
    """

    def __init__(
        self,
        source,
        gpu: bool | int = False,
    ) -> None:
        """Open video file for reading.

        Args:
            source: Path to video file (str or Path), or a seekable file-like
                object (must support read, seek, tell).
            gpu: False for CPU decoding, True for GPU (CUDA) on default device,
                or an int to select a specific GPU device ordinal.
        """
        self._is_fileobj = hasattr(source, 'read')

        if self._is_fileobj:
            if gpu:
                raise ValueError('GPU decoding requires a filesystem path, not a file-like object')
            self.path = None
            self._source = source
            try:
                self._container = av.open(source, metadata_errors='surrogateescape')
            except av.error.FFmpegError as e:
                raise VideoDecodeError('<file-like>', 0, e) from e
        else:
            self.path = Path(source)
            self._source = source
            if not self.path.exists():
                raise FileNotFoundError(f'Video file not found: {source}')
            try:
                self._container = av.open(
                    str(source),
                    hwaccel=_make_hwaccel(gpu),
                    metadata_errors='surrogateescape',
                )
            except RuntimeError as e:
                # PyAV raises a plain RuntimeError when no stream is
                # NVDEC-compatible and software fallback is disabled
                raise FramePumpError(
                    f'GPU decode is not supported for this codec: {source}. '
                    f'Use gpu=False for software decoding. ({e})'
                ) from e
            except av.FFmpegError as e:
                if gpu and isinstance(e, av.error.ExternalError):
                    raise ValueError(
                        f'GPU decoding failed to initialize on device {gpu!r} '
                        f'(is the device ordinal valid?): {e}'
                    ) from e
                raise VideoDecodeError(source, 0, e) from e

        self._gpu = gpu

        if not self._container.streams.video:
            raise NoVideoStreamError(source)
        self._stream = self._container.streams.video[0]
        _discard_other_streams(self._container, self._stream)

        # codec_context is None when this FFmpeg build has no decoder for the
        # stream's codec (e.g. JPEG-XL, VVC on older builds)
        if self._stream.codec_context is None:
            raise UnsupportedCodecError(source)

        if not self._stream.width or not self._stream.height:
            # e.g. animated WebP on FFmpeg builds without an anim decoder:
            # the stream probes as 0x0 and decoding fails with cryptic errors
            raise VideoDecodeError(
                source,
                0,
                RuntimeError('Video stream has no valid dimensions (unsupported format?)'),
            )

        # Enable multi-threaded decoding (~5x speedup).
        # Some formats/codecs don't support threading safely.
        _THREADING_UNSAFE_CODECS = {'vp4'}
        codec_name = self._stream.codec_context.codec.name
        self.codec_name = codec_name
        self._use_threading = (
            self._container.format.name not in ('pmp',)
            and codec_name not in _THREADING_UNSAFE_CODECS
        )
        if self._use_threading:
            self._stream.thread_type = 'AUTO'
        self._threading_active = self._use_threading

        # Cache metadata as Fractions for exact arithmetic
        self._fps_frac: Fraction | None = None
        self._duration_frac: Fraction | None = None
        self._time_base: Fraction = Fraction(
            self._stream.time_base.numerator, self._stream.time_base.denominator
        )

        # Test if seeking is supported (cached).
        self._seekable: bool | None = None
        self._current_frame_idx: int = 0  # Track position for non-seekable streams

        # Sticky: set when decode_raw observes display-order PTS regressing
        self.pts_regression_seen: bool = False
        # False until anything demuxes, decodes or seeks: a fresh container is
        # already at the start, so seeking it to 0 is at best pointless and at
        # worst destructive (palette-carrying codecs lose their palette on
        # seek, some raw demuxers report spurious EOF errors afterwards)
        self._consumed: bool = False

    @property
    def seekable(self) -> bool:
        """Whether seeking is supported for this container (cached)."""
        if self._seekable is None:
            format_name = self._container.format.name
            # Attached picture streams (cover art) crash on seek - detect via average_rate=None
            if self._stream.average_rate is None:
                self._seekable = False
            # Formats where seek corrupts decode state (must reopen instead)
            # - image2: still images (JPG, PNG, etc.)
            # - dirac: raw dirac bitstream
            # - *_pipe: image pipes (bmp_pipe, png_pipe, etc.) - seek(0) works but seek(N) fails
            # - cavsvideo: CAVS codec fails to decode after seeking
            elif (
                'image' in format_name
                or format_name in ('dirac', 'cavsvideo', 'rm')
                or '_pipe' in format_name
            ):
                self._seekable = False
            else:
                try:
                    self._container.seek(0, stream=self._stream)
                    # Basic seek works, now probe for broken index
                    self._seekable = self._probe_seek_works()
                except av.error.FFmpegError:
                    self._seekable = False
                finally:
                    # The probe seeks (and decodes) around; a later seek(0)
                    # does not reliably restore the start on fuzzy-seeking
                    # containers (MPEG-PS lands mid-file, losing half the
                    # packets for whoever demuxes next) and stateful decoders
                    # lose palette/history. Reopen for a clean slate.
                    self._reopen()
        return self._seekable

    def _probe_seek_works(self) -> bool:
        """Probe whether backward seek actually lands at or before target.

        Some containers (broken MXF, certain MPEG-TS) have corrupt or missing
        seek indices. FFmpeg's backward seek should land AT or BEFORE the target,
        but on these files it lands AFTER. This detects such cases.
        """
        start_time = self._stream.start_time or 0
        duration = self._stream.duration
        if duration is None or duration <= 0:
            # Can't probe without duration, assume seekable
            return True

        # Seek to middle of video with backward flag
        test_pts = start_time + duration // 2
        try:
            self._container.seek(test_pts, stream=self._stream, backward=True)
            frame = next(self._container.decode(self._stream))
            # Backward seek should land AT or BEFORE target
            # If it lands AFTER, the seek index is broken
            return frame.pts is not None and frame.pts <= test_pts
        except (av.error.FFmpegError, StopIteration):
            # Seek/decode failed, treat as non-seekable
            return False

    def _reopen(self, *, use_threading: bool | None = None) -> None:
        """Reopen container (for non-seekable streams).

        Args:
            use_threading: Enable multi-threaded decoding. None (default)
                inherits the setting from __init__. False disables for corrupt
                files where threading causes decode failures.
        """
        if use_threading is None:
            use_threading = self._use_threading
        self._threading_active = use_threading
        self._container.close()
        if self._is_fileobj:
            self._source.seek(0)
            try:
                self._container = av.open(self._source, metadata_errors='surrogateescape')
            except av.error.FFmpegError as e:
                raise VideoDecodeError('<file-like>', 0, e) from e
        else:
            try:
                self._container = av.open(
                    str(self.path),
                    hwaccel=_make_hwaccel(self._gpu),
                    metadata_errors='surrogateescape',
                )
            except av.error.FFmpegError as e:
                raise VideoDecodeError(self.path, 0, e) from e
        self._stream = self._container.streams.video[0]
        _discard_other_streams(self._container, self._stream)
        if use_threading:
            self._stream.thread_type = 'AUTO'
        self._current_frame_idx = 0
        self._consumed = False

    def seek_to_frame(self, frame_idx: int) -> None:
        """Seek to a frame index.

        For seekable streams: uses fast seeking.
        For non-seekable: reopens if needed, then skips frames.
        """
        if self.seekable:
            # Normal seeking not possible by frame index alone
            # This method is for non-seekable streams primarily
            raise NotImplementedError('Use seek() with PTS for seekable streams')

        # Non-seekable: reopen if target is before current position
        if frame_idx < self._current_frame_idx:
            self._reopen(use_threading=False)

        # Skip forward to target frame (demux without decode for speed)
        while self._current_frame_idx < frame_idx:
            for packet in self._container.demux(self._stream):
                if packet.dts is not None or packet.pts is not None:
                    self._current_frame_idx += 1
                    if self._current_frame_idx >= frame_idx:
                        break
            else:
                # End of stream
                break

    # --- Metadata Properties (Fraction-first) ---

    @property
    def fps_fraction(self) -> Fraction:
        """Video frame rate as exact Fraction."""
        if self._fps_frac is None:
            rate = self._stream.guessed_rate or self._stream.average_rate
            if rate is not None:
                self._fps_frac = Fraction(rate.numerator, rate.denominator)
            else:
                # Fallback: estimate from duration and frame count
                self._fps_frac = Fraction(30, 1)
        return self._fps_frac

    @property
    def fps(self) -> float:
        """Video frame rate as float."""
        return float(self.fps_fraction)

    @property
    def duration_fraction(self) -> Fraction:
        """Video duration as exact Fraction (in seconds)."""
        if self._duration_frac is None:
            if self._container.duration is not None:
                # Container duration is in AV_TIME_BASE (1/1000000)
                self._duration_frac = Fraction(self._container.duration, av.time_base)
            elif self._stream.duration is not None:
                # Stream duration in stream time_base
                self._duration_frac = Fraction(self._stream.duration) * self._time_base
            else:
                # Fallback to 0
                self._duration_frac = Fraction(0)
        return self._duration_frac

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return float(self.duration_fraction)

    @property
    def time_base(self) -> Fraction:
        """Stream time base as Fraction."""
        return self._time_base

    @property
    def resolution(self) -> tuple[int, int]:
        """Video resolution as (width, height)."""
        return (self._stream.width, self._stream.height)

    @property
    def frame_count_estimate(self) -> int:
        """Estimated frame count (may not be exact for VFR videos)."""
        # Try stream's frame count first
        if self._stream.frames > 0:
            return self._stream.frames

        # Fallback: estimate from duration * fps
        return int(self.duration_fraction * self.fps_fraction)

    def has_audio(self) -> bool:
        """Check if video has audio stream."""
        return len(self._container.streams.audio) > 0

    # --- Seeking and Decoding ---

    def seek(self, pts: int | Fraction, *, any_frame: bool = False) -> None:
        """Seek to a position in the video.

        Args:
            pts: Target PTS in stream time_base units, or Fraction of seconds.
            any_frame: If True, seek to nearest frame (faster). If False, seek
                to nearest keyframe before target (safer, default).
        """
        if isinstance(pts, Fraction):
            # Convert seconds to stream PTS units
            pts_int = int(pts / self._time_base)
        else:
            pts_int = pts

        self._consumed = True
        self._container.seek(pts_int, stream=self._stream, any_frame=any_frame)

    def seek_to_time(self, time_seconds: float | Fraction) -> None:
        """Seek to a time position in seconds.

        For non-seekable streams, reopens the container if seeking to 0,
        otherwise raises an error.
        """
        if isinstance(time_seconds, float):
            time_seconds = Fraction(time_seconds).limit_denominator(1000000)

        if self.seekable and time_seconds == 0 and not self._consumed:
            # Fresh container: already at the start; see _consumed above
            return

        if not self.seekable:
            if time_seconds == 0:
                # Corrupt data in non-seekable files can fail with threaded
                # decoding; a fresh single-threaded container needs no reset
                if not self._consumed and not self._threading_active:
                    return
                self._reopen(use_threading=False)
                return
            else:
                raise RuntimeError(
                    f'Cannot seek to {time_seconds}s in non-seekable stream. '
                    f'Only seeking to 0 (reopen) is supported.'
                )
        self.seek(time_seconds)

    def decode_raw(self) -> Generator[av.VideoFrame, None, None]:
        """Decode raw frames from current position, mapping decoder errors.

        ``InvalidDataError`` becomes ``VideoDecodeError``; I/O errors at end of
        stream (errno 5) are treated as EOF (malformed EOF markers in some
        containers).
        """
        count = 0
        prev_pts = None
        self._consumed = True
        try:
            for frame in self._container.decode(self._stream):
                if frame.pts is not None:
                    if prev_pts is not None and frame.pts < prev_pts:
                        # Decoders emit display order, so PTS should never
                        # regress; a regression means emission order and
                        # sorted-PTS order disagree, which breaks PTS-based
                        # frame location. Callers consult this flag.
                        self.pts_regression_seen = True
                    prev_pts = frame.pts
                yield frame
                count += 1
        except av.error.EOFError:
            # AVERROR_EOF surfacing as an exception: treat as end of stream
            return
        except av.FFmpegError as e:
            # Treat I/O errors as end of stream (malformed EOF in some containers)
            if getattr(e, 'errno', None) == 5:
                return
            # Wrap every FFmpeg-level failure (invalid data, unknown decoder
            # errors, unimplemented features, DRM permission errors, ...) so
            # callers see a FramePumpError instead of raw av internals
            raise VideoDecodeError(self.path, count, e) from e
        except OSError as e:
            if e.errno != 5:
                raise

    def frame_converter(
        self, output_shape: tuple[int, int] | None, target_format: str
    ) -> '_FrameConverter':
        """Create a converter that runs decoded frames through the filter graph.

        The graph is built lazily from the first frame's actual properties
        (see _build_filter_graph) and libav-level conversion errors are
        wrapped into VideoDecodeError.
        """
        return _FrameConverter(self, output_shape, target_format)

    def decode_frames(
        self,
        max_frames: int | None = None,
        output_shape: tuple[int, int] | None = None,
        dtype: DTypeLike = np.uint8,
        target_format: str | None = None,
    ) -> Generator[NDArray, None, None]:
        """Decode frames from current position.

        Args:
            max_frames: Stop after this many frames (None for all).
            output_shape: Resize to (height, width). None to keep original.
            dtype: Output dtype (np.uint8 or np.uint16).

        Yields:
            numpy arrays of shape (height, width, 3) with RGB pixel data.

        Note:
            I/O errors at end of stream are treated as EOF (malformed EOF markers).
        """
        if dtype not in (np.uint8, np.uint16):
            raise ValueError(f'Unsupported dtype: {dtype}')

        if target_format is None:
            # Choose pixel format based on dtype
            target_format = 'rgb48' if dtype == np.uint16 else 'rgb24'

        converter = self.frame_converter(output_shape, target_format)

        count = 0
        for frame in self.decode_raw():
            filtered_frame = converter.convert(frame)

            # Convert to numpy
            arr = filtered_frame.to_ndarray()

            yield arr

            count += 1
            if max_frames is not None and count >= max_frames:
                break

    def _build_filter_graph(
        self, output_shape: tuple[int, int] | None, target_format: str, frame: av.VideoFrame
    ) -> av.filter.Graph:
        """Build a filter graph for format/resize conversion.

        Configured from the first decoded frame's actual properties (with
        hwaccel the stream-level pix_fmt is 'cuda', not the format frames
        actually arrive in). Uses FFmpeg's filter system for exact
        compatibility with subprocess output.
        """
        graph = av.filter.Graph()
        buffer_in = graph.add_buffer(
            width=frame.width,
            height=frame.height,
            format=frame.format.name,
            time_base=self._stream.time_base,
        )
        buffer_out = graph.add('buffersink')

        last_filter = buffer_in

        # NVDEC downloads arrive semi-planar (nv12/p010le). Repack losslessly
        # to the planar equivalent first, so RGB conversion runs through the
        # exact same swscale path as CPU decoding: direct nv12->rgb uses a
        # different chroma interpolation and differs on most pixels.
        planar = _SEMIPLANAR_TO_PLANAR.get(frame.format.name)
        if planar is not None:
            repack = graph.add('format', f'pix_fmts={planar}')
            last_filter.link_to(repack)
            last_filter = repack

        # Work around libswscale's SSSE3 pmulhw truncation bug: the fused
        # chroma-upsample + color-convert path truncates instead of rounding,
        # causing ~2% darkening on full-range subsampled content.
        # accurate_rnd bypasses the SSSE3 path; full_chroma_int fixes chroma
        # interpolation. Needed for subsampled full-range content: yuvj
        # formats from CPU decoding, or full-range nv12 from NVDEC (mjpeg).
        pix_fmt = frame.format.name
        needs_sws_fix = pix_fmt in ('yuvj420p', 'yuvj422p') or (
            pix_fmt in ('nv12', 'nv16') and frame.color_range == 2  # JPEG/full range
        )

        # Add scale filter if resize needed or if we need SWS flags
        if output_shape is not None:
            height, width = output_shape
            flags = ':flags=accurate_rnd+full_chroma_int' if needs_sws_fix else ''
            scale_filter = graph.add('scale', f'{width}:{height}{flags}')
            last_filter.link_to(scale_filter)
            last_filter = scale_filter
        elif needs_sws_fix:
            scale_filter = graph.add('scale', 'w=iw:h=ih:flags=accurate_rnd+full_chroma_int')
            last_filter.link_to(scale_filter)
            last_filter = scale_filter

        # Add format filter for pixel format conversion
        format_filter = graph.add('format', f'pix_fmts={target_format}')
        last_filter.link_to(format_filter)
        format_filter.link_to(buffer_out)

        try:
            graph.configure()
        except av.error.NotImplementedError as e:
            input_fmt = frame.format.name
            raise FilterConfigError(
                f'Failed to configure filter graph (input: {input_fmt}, output: {target_format}). '
                f'The pixel format may not be supported for conversion.'
            ) from e
        return graph

    def get_frame(
        self,
        pts: int | Fraction,
        output_shape: tuple[int, int] | None = None,
        dtype: DTypeLike = np.uint8,
    ) -> NDArray:
        """Get single frame at specific PTS.

        Args:
            pts: Target PTS in stream time_base units, or Fraction of seconds.
            output_shape: Resize to (height, width).
            dtype: Output dtype.

        Returns:
            Frame as numpy array.
        """
        self.seek(pts)
        gen = self.decode_frames(max_frames=1, output_shape=output_shape, dtype=dtype)
        try:
            return next(gen)
        except StopIteration:
            raise RuntimeError(f'Failed to decode frame at pts={pts}')

    # --- Packet-Level Access (for index building) ---

    def _reset_to_start(self) -> None:
        """Reset to start of stream (seek if possible, otherwise reopen)."""
        if not self._consumed:
            self._consumed = True  # the caller is about to demux/decode
            return
        if self.seekable:
            self._container.seek(0, stream=self._stream)
            self._consumed = True
        else:
            # Non-seekable files may have corrupt data that fails with threading
            self._reopen(use_threading=False)
            self._consumed = True

    def count_packets(self) -> int:
        """Count packets (for non-seekable streams that need frame count only).

        Note:
            I/O errors at end of stream are treated as EOF.
        """
        self._reset_to_start()
        count = 0
        try:
            for packet in self._container.demux(self._stream):
                if packet.dts is not None or packet.pts is not None:
                    count += 1
        except OSError as e:
            if e.errno != 5:  # Not an I/O error
                raise
        return count

    def iter_packets(self) -> Generator[PacketInfo, None, None]:
        """Iterate over video packets for index building.

        Yields:
            PacketInfo with PTS, DTS, keyframe status (all in Fraction seconds).

        Note:
            I/O errors at end of stream are treated as EOF (some containers
            have malformed EOF markers that ffmpeg handles gracefully).
        """
        self._reset_to_start()

        try:
            for packet in self._container.demux(self._stream):
                # Skip empty flush packets
                if packet.dts is None and packet.pts is None:
                    continue

                pts_frac = (
                    Fraction(packet.pts) * self._time_base if packet.pts is not None else None
                )
                dts_frac = (
                    Fraction(packet.dts) * self._time_base if packet.dts is not None else None
                )

                yield PacketInfo(
                    pts=pts_frac,
                    dts=dts_frac,
                    is_keyframe=packet.is_keyframe,
                )
        except av.error.InvalidDataError:
            # Truncated/corrupt data at EOF - treat as end of stream
            return
        except OSError as e:
            # Treat certain errors as end of stream (truncated/malformed files)
            # errno 1 = EPERM (truncated packets in some formats)
            # errno 5 = EIO (malformed EOF markers)
            # errno 11 = EAGAIN (incomplete data)
            if e.errno in (1, 5, 11):
                return
            raise

    def iter_frame_pts(self) -> Generator[Fraction, None, None]:
        """Iterate decoded frames and yield their PTS (for edit-list videos).

        This is slower than iter_packets() but handles edit lists correctly.

        Yields:
            Frame PTS as Fraction (seconds).

        Raises:
            VideoDecodeError: If decoding fails due to corrupt/truncated data.

        Note:
            I/O errors at end of stream are treated as EOF (malformed EOF markers).
        """
        # Re-seek to beginning
        if self._seekable:
            self._container.seek(0, stream=self._stream)
        else:
            # Reopen container for non-seekable streams (without threading for corrupt files)
            self._reopen(use_threading=False)

        frame_count = 0
        try:
            for frame in self._container.decode(self._stream):
                if frame.pts is not None:
                    yield Fraction(frame.pts) * self._time_base
                frame_count += 1
        except av.error.InvalidDataError as e:
            raise VideoDecodeError(self.path, frame_count, e) from e
        except OSError as e:
            # Treat I/O errors as end of stream (malformed EOF in some containers)
            if e.errno != 5:
                raise

    # --- Context Manager ---

    def close(self) -> None:
        """Close the container."""
        self._container.close()

    def __enter__(self) -> 'PyAVReader':
        return self

    def __exit__(self, *args) -> None:
        self.close()


@dataclass(slots=True)
class PacketInfo:
    """Information about a video packet."""

    pts: Fraction | None
    dts: Fraction | None
    is_keyframe: bool


class _FrameConverter:
    """Runs decoded frames through a lazily built format/resize filter graph.

    The graph is created from the first frame's actual properties, not the
    stream metadata: with hwaccel the stream reports pix_fmt='cuda' while
    frames arrive as downloaded nv12/p010le software frames, and even in CPU
    decoding the frames are the ground truth.
    """

    def __init__(
        self, reader: PyAVReader, output_shape: tuple[int, int] | None, target_format: str
    ) -> None:
        self._reader = reader
        self._output_shape = output_shape
        self._target_format = target_format
        self._graph: av.filter.Graph | None = None
        self._count = 0

    def convert(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Convert one decoded frame, building the graph on first use."""
        # Decoders can emit the illegal 'reserved' (0) value for color
        # metadata, which FFmpeg >= 8 swscale rejects (ENOTSUP) instead of
        # assuming defaults; both 0 and 2 mean 'unknown' in practice.
        if frame.color_primaries == 0:
            frame.color_primaries = 2  # AVCOL_PRI_UNSPECIFIED
        if frame.color_trc == 0:
            frame.color_trc = 2  # AVCOL_TRC_UNSPECIFIED

        try:
            if self._graph is None:
                self._graph = self._reader._build_filter_graph(
                    self._output_shape, self._target_format, frame
                )
            self._graph.push(frame)
            filtered = self._graph.pull()
        except av.FFmpegError as e:
            # e.g. colorspaces swscale cannot convert to RGB (YCgCo)
            path = self._reader.path if self._reader.path is not None else '<file-like>'
            raise VideoDecodeError(path, self._count, e) from e
        self._count += 1
        return filtered


class FrameIndexPyAV:
    """Frame index built using PyAV packet iteration.

    Stores PTS values as Fraction for exact arithmetic. Compatible with
    existing FrameIndex interface but uses PyAV instead of ffprobe subprocess.
    """

    video_path: Path
    frame_pts: list[Fraction]  # PTS in Fraction (seconds)
    safe_seek_pts: list[Fraction]  # Safe seek points in Fraction
    frame_count: int
    had_duplicate_pts: bool  # packet PTS collapsed in the index (count suspect)

    def __init__(self, video_path, reader: PyAVReader | None = None) -> None:
        """Build index from video file using PyAV.

        Args:
            video_path: Path to the video file, or a file-like object.
            reader: Optional existing PyAVReader to use (avoids reopening).
        """
        self.video_path = Path(video_path) if isinstance(video_path, (str, Path)) else video_path

        # Use provided reader or create temporary one
        own_reader = reader is None
        if own_reader:
            reader = PyAVReader(video_path)

        try:
            if not reader.seekable:
                # Non-seekable: trivial index (count only, always seek to 0)
                self.frame_pts, self.safe_seek_pts = self._build_sequential_index(reader)
                self.had_duplicate_pts = False
            else:
                self.frame_pts, self.safe_seek_pts, self.had_duplicate_pts = (
                    self._build_from_packets(reader)
                )
        finally:
            if own_reader:
                reader.close()

        if not self.frame_pts:
            raise IndexBuildError('No valid frames found')

        self.frame_count = len(self.frame_pts)

    @staticmethod
    def _build_from_packets(reader: PyAVReader) -> tuple[list[Fraction], list[Fraction], bool]:
        """Build index from packet metadata (fast, no decoding)."""
        file_order_pts: list[Fraction] = []
        running_max_at: list[Fraction] = []

        running_max = Fraction(-1, 1)

        for pkt in reader.iter_packets():
            pts = pkt.pts
            if pts is None:
                pts = pkt.dts
            # Negative-PTS packets are discard-flagged preroll (edit-list trims)
            # that libavcodec also drops, so index and decode agree by construction.
            if pts is None or pts < 0:
                continue

            file_order_pts.append(pts)
            running_max = max(running_max, pts)
            running_max_at.append(running_max)

        # Sort by PTS for display order, remove duplicates. Duplicates mean
        # packet count != frame count; the flag lets the caller distrust the
        # index and rebuild it from decoder output.
        frame_pts = sorted(set(file_order_pts))
        had_duplicates = len(frame_pts) < len(file_order_pts)

        # Build safe seek points using binary search
        # Safe seek point = last packet in file order where running_max <= target
        # This ensures we've received all packets needed to decode target frame
        safe_seek_pts: list[Fraction] = []
        for target in frame_pts:
            idx = bisect.bisect_right(running_max_at, target) - 1
            if idx >= 0:
                safe_seek_pts.append(file_order_pts[idx])
            else:
                # No packet found with running_max <= target (B-frames at start)
                # Seek to whichever is earlier: first packet or position 0
                safe_seek_pts.append(min(file_order_pts[0], Fraction(0)))

        return frame_pts, safe_seek_pts, had_duplicates

    @staticmethod
    def _build_sequential_index(reader: PyAVReader) -> tuple[list[Fraction], list[Fraction]]:
        """Build index for non-seekable streams.

        Same as _build_from_packets but all safe_seek_pts are 0
        (always reopen and decode from start).

        For timestampless streams (raw h264, etc.), falls back to decoding
        to count frames and generates synthetic PTS based on fps.
        """
        file_order_pts: list[Fraction] = []

        for pkt in reader.iter_packets():
            pts = pkt.pts
            if pts is None:
                pts = pkt.dts
            # Negative PTS = discard-flagged preroll; see _build_from_packets.
            if pts is None or pts < 0:
                continue
            file_order_pts.append(pts)

        if file_order_pts:
            # Normal case: have PTS values
            frame_pts = sorted(set(file_order_pts))
            safe_seek_pts = [Fraction(0)] * len(frame_pts)
            return frame_pts, safe_seek_pts

        # No PTS values (raw bitstreams) - must decode to count frames
        # This is slower but necessary for timestampless streams
        reader._reset_to_start()
        frame_count = 0
        try:
            for _ in reader._container.decode(reader._stream):
                frame_count += 1
        except av.FFmpegError:
            # Truncated/corrupt tails and decoder-side failures (invalid data,
            # unimplemented features like raw VVC): the frames counted so far
            # match what decoding delivers before it raises. A file where
            # nothing decodes raises the typed IndexBuildError downstream.
            pass
        except OSError as e:
            if e.errno != 5:  # I/O error at EOF is ok
                raise

        if frame_count == 0:
            return [], []

        # Generate synthetic PTS based on fps (frames are in display order)
        fps = reader.fps_fraction
        frame_pts = [Fraction(i, 1) / fps for i in range(frame_count)]
        safe_seek_pts = [Fraction(0)] * frame_count

        return frame_pts, safe_seek_pts

    def get_seek_params(self, frame_idx: int) -> tuple[float, float]:
        """Get seek parameters for a frame.

        Returns:
            Tuple of (input_seek_pts, output_trim_time) as floats for FFmpeg compat.
        """
        target_pts = self.frame_pts[frame_idx]
        safe_pts = self.safe_seek_pts[frame_idx]
        trim = target_pts - safe_pts
        return float(safe_pts), float(trim)

    def get_frame_pts(self, frame_idx: int) -> float:
        """Get the PTS for a specific frame (as float for FFmpeg compat)."""
        return float(self.frame_pts[frame_idx])

    def get_frame_pts_fraction(self, frame_idx: int) -> Fraction:
        """Get the PTS for a specific frame as exact Fraction."""
        return self.frame_pts[frame_idx]

    def __repr__(self) -> str:
        name = self.video_path.name if isinstance(self.video_path, Path) else '<file-like>'
        return f'FrameIndexPyAV({name!r}, frames={self.frame_count})'


class IndexBuildError(FramePumpError):
    """Raised when index building fails."""

    pass
