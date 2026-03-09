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

    def __init__(self, path: PathLike, frame_count: int, original_error: Exception):
        self.path = Path(path)
        self.frame_count = frame_count
        self.original_error = original_error
        msg = (
            f'Corrupt or truncated video: {self.path.name}\n'
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
        self.path = Path(path)
        self.frame_count = frame_count
        self.original_error = original_error
        self.resolution = resolution
        self.codec = codec

        # Build informative message
        parts = []
        # Detect NVENC small frame error
        if codec and 'nvenc' in codec and resolution and (resolution[0] < 150 or resolution[1] < 50):
            parts.append(
                f'NVENC frame size too small: {resolution[0]}x{resolution[1]} '
                f'(minimum ~145x49 for h264_nvenc)'
            )
        else:
            parts.append(f'Failed to encode video: {self.path.name}')
            if resolution:
                parts.append(f'Resolution: {resolution[0]}x{resolution[1]}')
        if codec:
            parts.append(f'Codec: {codec}')
        parts.append(f'Encoded {frame_count} frames before error')
        parts.append(f'Original error: {type(original_error).__name__}: {original_error}')
        super().__init__('\n'.join(parts))


class NoAudioStreamError(FramePumpError):
    """Raised when audio is expected but not found."""

    def __init__(self, path: PathLike):
        self.path = Path(path)
        super().__init__(f'No audio stream found in {self.path.name}')


class FilterConfigError(FramePumpError):
    """Raised when filter graph configuration fails."""

    pass


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
        path: PathLike,
        gpu: bool | int = False,
    ) -> None:
        """Open video file for reading.

        Args:
            path: Path to video file.
            gpu: False for CPU decoding, True for GPU (CUDA) on default device,
                or an int to select a specific GPU device ordinal.
        """
        self.path = Path(path)
        self._gpu = gpu

        if not self.path.exists():
            raise FileNotFoundError(f'Video file not found: {path}')

        # When GPU is requested, probe the file first (without hwaccel) to check
        # whether CUDA decode is safe. Certain container formats (FLV) and codecs
        # without CUVID support cause unrecoverable segfaults inside FFmpeg when
        # opened with hwaccel=cuda.
        if gpu:
            self._probe_gpu_safety(path)

        # Open container with optional GPU acceleration
        options = {}
        if gpu:
            options['hwaccel'] = 'cuda'
            if type(gpu) is int:
                options['hwaccel_device'] = str(gpu)

        try:
            self._container = av.open(
                str(path), options=options, metadata_errors='surrogateescape')
        except av.error.FFmpegError as e:
            raise VideoDecodeError(path, 0, e) from e
        if not self._container.streams.video:
            raise ValueError(f'No video stream found in {path}')
        self._stream = self._container.streams.video[0]

        # Enable multi-threaded decoding (~5x speedup).
        # Some formats/codecs don't support threading safely.
        _THREADING_UNSAFE_CODECS = {'vp4'}
        codec_name = self._stream.codec_context.codec.name
        self._use_threading = (
            self._container.format.name not in ('pmp',)
            and codec_name not in _THREADING_UNSAFE_CODECS
        )
        if self._use_threading:
            self._stream.thread_type = 'AUTO'

        # Cache metadata as Fractions for exact arithmetic
        self._fps_frac: Fraction | None = None
        self._duration_frac: Fraction | None = None
        self._time_base: Fraction = Fraction(
            self._stream.time_base.numerator, self._stream.time_base.denominator
        )

        # Test if seeking is supported (cached)
        self._seekable: bool | None = None
        self._current_frame_idx: int = 0  # Track position for non-seekable streams

    # Codecs with NVIDIA CUVID hardware decoder support.
    _CUVID_CODECS = {
        'h264', 'hevc', 'mpeg1video', 'mpeg2video', 'mpeg4',
        'av1', 'vp8', 'vp9', 'vc1', 'mjpeg',
    }

    # Container formats where hwaccel=cuda causes segfaults in av.open().
    _GPU_UNSAFE_FORMATS = {'flv'}

    @staticmethod
    def _probe_gpu_safety(path: PathLike) -> None:
        """Check if GPU decode is safe, to avoid segfaults in FFmpeg's CUDA path.

        Opens the file without hwaccel to inspect the container format and codec.
        Raises VideoDecodeError if GPU decode would be unsafe.
        """
        try:
            probe = av.open(str(path), metadata_errors='surrogateescape')
        except av.error.FFmpegError as e:
            raise VideoDecodeError(path, 0, e) from e
        try:
            if not probe.streams.video:
                raise ValueError(f'No video stream found in {path}')
            fmt = probe.format.name
            codec = probe.streams.video[0].codec_context.codec.name

            # FLV containers segfault in FFmpeg when opened with hwaccel=cuda
            fmt_parts = set(fmt.split(','))
            if fmt_parts & PyAVReader._GPU_UNSAFE_FORMATS:
                raise FramePumpError(
                    f'GPU decode is not supported for the {fmt!r} container format '
                    f'(causes a crash in FFmpeg). Use gpu=False.'
                )

            if codec not in PyAVReader._CUVID_CODECS:
                raise FramePumpError(
                    f'GPU decode is not supported for the {codec!r} codec. '
                    f'Supported codecs: {", ".join(sorted(PyAVReader._CUVID_CODECS))}. '
                    f'Use gpu=False.'
                )
        finally:
            probe.close()

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
            elif 'image' in format_name or format_name in ('dirac', 'cavsvideo', 'rm') or '_pipe' in format_name:
                self._seekable = False
            else:
                try:
                    self._container.seek(0, stream=self._stream)
                    # Basic seek works, now probe for broken index
                    self._seekable = self._probe_seek_works()
                except av.error.FFmpegError:
                    self._seekable = False
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
        self._container.close()
        options = {}
        if self._gpu:
            options['hwaccel'] = 'cuda'
            if type(self._gpu) is int:
                options['hwaccel_device'] = str(self._gpu)
        try:
            self._container = av.open(
                str(self.path), options=options, metadata_errors='surrogateescape')
        except av.error.FFmpegError as e:
            raise VideoDecodeError(self.path, 0, e) from e
        self._stream = self._container.streams.video[0]
        if use_threading:
            self._stream.thread_type = 'AUTO'
        self._current_frame_idx = 0

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

        self._container.seek(pts_int, stream=self._stream, any_frame=any_frame)

    def seek_to_time(self, time_seconds: float | Fraction) -> None:
        """Seek to a time position in seconds.

        For non-seekable streams, reopens the container if seeking to 0,
        otherwise raises an error.
        """
        if isinstance(time_seconds, float):
            time_seconds = Fraction(time_seconds).limit_denominator(1000000)

        if not self.seekable:
            if time_seconds == 0:
                self._reopen(use_threading=False)
                return
            else:
                raise RuntimeError(
                    f'Cannot seek to {time_seconds}s in non-seekable stream. '
                    f'Only seeking to 0 (reopen) is supported.'
                )
        self.seek(time_seconds)

    def decode_frames(
        self,
        max_frames: int | None = None,
        output_shape: tuple[int, int] | None = None,
        dtype: DTypeLike = np.uint8,
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

        # Choose pixel format based on dtype
        target_format = 'rgb48' if dtype == np.uint16 else 'rgb24'

        # Build filter graph for exact FFmpeg compatibility
        graph = self._build_filter_graph(output_shape, target_format)

        count = 0
        try:
            for frame in self._container.decode(self._stream):
                # Push frame through filter graph
                graph.push(frame)
                filtered_frame = graph.pull()

                # Convert to numpy
                arr = filtered_frame.to_ndarray()

                yield arr

                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        except av.error.InvalidDataError as e:
            raise VideoDecodeError(self.path, count, e) from e
        except OSError as e:
            # Treat I/O errors as end of stream (malformed EOF in some containers)
            if e.errno != 5:
                raise

    def _build_filter_graph(
        self, output_shape: tuple[int, int] | None, target_format: str
    ) -> av.filter.Graph:
        """Build a filter graph for format/resize conversion.

        Uses FFmpeg's filter system for exact compatibility with subprocess output.
        """
        graph = av.filter.Graph()
        buffer_in = graph.add_buffer(template=self._stream)
        buffer_out = graph.add('buffersink')

        last_filter = buffer_in

        # Work around libswscale's SSSE3 pmulhw truncation bug: the fused
        # chroma-upsample + color-convert path truncates instead of rounding,
        # causing ~2% darkening on full-range subsampled content.
        # accurate_rnd bypasses the SSSE3 path; full_chroma_int fixes chroma
        # interpolation. Only needed for subsampled full-range (yuvj420p/yuvj422p).
        pix_fmt = self._stream.codec_context.pix_fmt or ''
        needs_sws_fix = pix_fmt in ('yuvj420p', 'yuvj422p')

        # Add scale filter if resize needed or if we need SWS flags
        if output_shape is not None:
            height, width = output_shape
            flags = ':flags=accurate_rnd+full_chroma_int' if needs_sws_fix else ''
            scale_filter = graph.add('scale', f'{width}:{height}{flags}')
            last_filter.link_to(scale_filter)
            last_filter = scale_filter
        elif needs_sws_fix:
            scale_filter = graph.add(
                'scale', 'w=iw:h=ih:flags=accurate_rnd+full_chroma_int')
            last_filter.link_to(scale_filter)
            last_filter = scale_filter

        # Add format filter for pixel format conversion
        format_filter = graph.add('format', f'pix_fmts={target_format}')
        last_filter.link_to(format_filter)
        format_filter.link_to(buffer_out)

        try:
            graph.configure()
        except av.error.NotImplementedError as e:
            input_fmt = self._stream.codec_context.pix_fmt
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
        if self.seekable:
            self._container.seek(0, stream=self._stream)
        else:
            # Non-seekable files may have corrupt data that fails with threading
            self._reopen(use_threading=False)

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


class FrameIndexPyAV:
    """Frame index built using PyAV packet iteration.

    Stores PTS values as Fraction for exact arithmetic. Compatible with
    existing FrameIndex interface but uses PyAV instead of ffprobe subprocess.
    """

    video_path: Path
    frame_pts: list[Fraction]  # PTS in Fraction (seconds)
    safe_seek_pts: list[Fraction]  # Safe seek points in Fraction
    frame_count: int

    def __init__(self, video_path: PathLike, reader: PyAVReader | None = None) -> None:
        """Build index from video file using PyAV.

        Args:
            video_path: Path to the video file.
            reader: Optional existing PyAVReader to use (avoids reopening).
        """
        self.video_path = Path(video_path)

        # Use provided reader or create temporary one
        own_reader = reader is None
        if own_reader:
            reader = PyAVReader(video_path)

        try:
            if not reader.seekable:
                # Non-seekable: trivial index (count only, always seek to 0)
                self.frame_pts, self.safe_seek_pts = self._build_sequential_index(reader)
            else:
                self.frame_pts, self.safe_seek_pts = self._build_from_packets(reader)
        finally:
            if own_reader:
                reader.close()

        if not self.frame_pts:
            raise IndexBuildError('No valid frames found')

        self.frame_count = len(self.frame_pts)

    @staticmethod
    def _build_from_packets(reader: PyAVReader) -> tuple[list[Fraction], list[Fraction]]:
        """Build index from packet metadata (fast, no decoding)."""
        file_order_pts: list[Fraction] = []
        running_max_at: list[Fraction] = []

        running_max = Fraction(-1, 1)

        for pkt in reader.iter_packets():
            pts = pkt.pts
            if pts is None:
                pts = pkt.dts
            if pts is None or pts < 0:
                continue

            file_order_pts.append(pts)
            running_max = max(running_max, pts)
            running_max_at.append(running_max)

        # Sort by PTS for display order, remove duplicates
        frame_pts = sorted(set(file_order_pts))

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

        return frame_pts, safe_seek_pts

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
        return f'FrameIndexPyAV({self.video_path.name!r}, frames={self.frame_count})'


class IndexBuildError(FramePumpError):
    """Raised when index building fails."""

    pass
