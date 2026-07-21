from __future__ import annotations

import io
import operator
from bisect import bisect_left
from collections import deque
from collections.abc import Generator
from fractions import Fraction
from pathlib import Path
from typing import Union, overload

import more_itertools
import numpy as np
import simplepyutils as spu
from numpy.typing import DTypeLike, NDArray

from ._pyav import FrameIndexPyAV, FramePumpError, PyAVReader, VideoDecodeError

PathLike = Union[str, Path]

__all__ = [
    'VideoFrames',
    'get_fps',
    'get_duration',
    'num_frames',
    'video_extents',
    'has_audio',
]


# Codecs whose containers are known to mark non-decodable packets as keyframes
# (screen codecs) or whose GOP structure defeats keyframe seeking (open-GOP
# MPEG-1/2, packed-B MPEG-4). Seeking is content-verified for these at
# construction and disabled when it does not reproduce sequential decode.
_SEEK_UNRELIABLE_CODECS = frozenset({'mpeg1video', 'mpeg2video', 'mpeg4', 'fic', 'vmnc'})


class VideoFrames:
    """Lazy, sliceable video frame iterator.

    Frames are only decoded when iterated. Slicing and resizing are lazy operations
    that return new VideoFrames instances without loading pixel data.

    Example:
        >>> frames = VideoFrames('video.mp4')
        >>> for frame in frames[::2][:100].resized((128, 128)):
        ...     process(frame)

    Args:
        video_path: Path to video file.
        dtype: Output dtype (uint8, uint16, float16, float32, float64).
        gpu: False for CPU decoding, True for GPU (CUDA) on default device,
            or an int to select a specific GPU device ordinal.
        constant_framerate: False for VFR (native timestamps), True for CFR at
            original fps, or a number for CFR at that specific fps.
    """

    def __init__(
        self,
        video_path,
        *,
        dtype: DTypeLike = np.uint8,
        gpu: bool | int = False,
        constant_framerate: Union[bool, float] = False,
        seekable: bool | None = None,
    ) -> None:
        """Open a video file for lazy frame access.

        Args:
            video_path: Path to video file (str or Path), or a seekable file-like
                object (must support read, seek, tell). File-like objects cannot
                be used with ``gpu=True``. ``BytesIO`` sources support any number
                of concurrently active iterators (each gets an independent view);
                other file-like objects allow only one active iterator at a time,
                since all iterators share the object's read position.
            seekable: Whether the video stream supports seeking. ``None``
                (default) auto-detects by probing. ``True``/``False`` skips the
                probe, which saves one seek+decode per reader creation.

        See class docstring for full parameter descriptions.
        """
        try:
            dtype = np.dtype(dtype).type
        except TypeError as e:
            raise ValueError(f'Unsupported dtype: {dtype!r}') from e
        if dtype not in (np.uint8, np.uint16, np.float16, np.float32, np.float64):
            raise ValueError(f'Unsupported dtype: {np.dtype(dtype).name}')

        self._is_fileobj = hasattr(video_path, 'read')
        self.path = video_path

        # Create persistent PyAV reader for metadata and decoding
        self._reader = PyAVReader(video_path, gpu=gpu)
        try:
            # Get metadata from reader (no subprocess calls)
            width, height = self._reader.resolution
            self.original_imshape: tuple[int, int] = (height, width)
            self.original_fps = self._reader.fps
            self.resized_imshape: tuple[int, int] | None = None
            self.repeat_count = 1

            self.dtype = dtype
            self.gpu = gpu

            # Parse constant_framerate: False, True, or a number (target fps)
            if isinstance(constant_framerate, bool):
                self.constant_framerate = constant_framerate
                self.target_fps = self.original_fps
            else:
                self.constant_framerate = True
                self.target_fps = float(constant_framerate)

            # Apply the caller's seekability before building the index, so the
            # index style (seek-based vs sequential) always agrees with what
            # iteration readers are allowed to do, and an explicit value
            # actually skips the probe.
            if seekable is not None:
                self._reader._seekable = seekable
            self._seekable = self._reader.seekable
            self._codec_name: str = self._reader.codec_name

            # Build frame index upfront
            self._index = FrameIndexPyAV(self.path, reader=self._reader)
        finally:
            # The reader gets reopened lazily in __iter__
            self._reader.close()
            self._reader = None

        # When True, decoded frame PTS cannot be trusted for locating frames
        # (set when the index had to synthesize timestamps); frame-count
        # matching is used instead.
        self._pts_unreliable = False

        # Some codecs/containers mark packets as keyframes that are not truly
        # independently decodable (screen codecs, open-GOP MPEG, packed-B
        # MPEG-4), so seeking would silently return wrong pixels. For those,
        # verify that seeking reproduces sequential decode and fall back to
        # sequential-only access (correct, slower) when it does not.
        if (
            self._index.frame_count > 1
            and self._codec_name in _SEEK_UNRELIABLE_CODECS
            and not self._seek_reproduces_sequential()
        ):
            self._seekable = False
            # The packet-based index may also count packets the decoder never
            # turns into frames (the same brokenness that defeats seeking), so
            # rebuild it from what the decoder actually produces.
            self._rebuild_index_from_decode()

        # In CFR mode, all behavior (count, indexing, iteration, seeking) derives
        # from this single output-index -> source-index map.
        if self.constant_framerate:
            self._cfr_source_map: list[int] | None = self._build_cfr_source_map()
            n_frames = len(self._cfr_source_map)
        else:
            self._cfr_source_map = None
            n_frames = self._index.frame_count

        # Store frame range - slicing applies directly to this
        self._frame_range: range = range(n_frames)

    def __iter__(self) -> Generator[NDArray, None, None]:
        internal_dtype = np.uint8 if self.dtype == np.uint8 else np.uint16

        frame_range = self._frame_range
        if len(frame_range) == 0:
            return

        # Create a fresh reader for this iteration
        reader = self._create_reader()
        try:
            raw_frames = self._iter_decoded(reader, frame_range, internal_dtype)
            frames = map(self._maybe_to_float, raw_frames)
            if self.repeat_count != 1:
                frames = spu.repeat_n(frames, self.repeat_count)
            count = 0
            for frame in frames:
                yield frame
                count += 1
            if count == 0:
                # The index recorded frames but the decoder produced none
                # (e.g. an unsupported bitstream feature the decoder skips
                # silently) — silence here would look like a valid empty view.
                raise VideoDecodeError(
                    self.path,
                    0,
                    RuntimeError(
                        f'Decoder produced no frames, but the video index '
                        f'recorded {len(frame_range)} for this range'
                    ),
                )
        finally:
            reader.close()

    @overload
    def __getitem__(self, item: int) -> NDArray: ...

    @overload
    def __getitem__(self, item: slice) -> VideoFrames: ...

    def __getitem__(self, item: int | slice) -> NDArray | VideoFrames:
        if isinstance(item, int):
            # Handle negative indices
            length = len(self)
            if item < 0:
                item = length + item
            if item < 0 or item >= length:
                raise IndexError(f'Frame index {item} out of range for video with {length} frames')

            # The bounds check above uses the repeat-inclusive length, so after
            # dividing out the repeat factor the index is always within range.
            abs_idx = self._frame_range[item // self.repeat_count]
            source_idx = self._abs_to_source(abs_idx)
            return self._maybe_to_float(self._decode_frame_at_source(source_idx))
        elif isinstance(item, slice):
            if self.repeat_count != 1:
                raise NotImplementedError(
                    'Slicing after repeat_each_frame() is not supported. '
                    'Apply slicing before repeat_each_frame(), e.g. '
                    'frames[::2].repeat_each_frame(3) instead of '
                    'frames.repeat_each_frame(3)[::2].'
                )

            if item.step is not None and item.step < 0:
                raise ValueError('Negative step not supported. Use list(frames)[::-1] instead.')

            if item.step == 0:
                raise ValueError('slice step cannot be zero')

            # Apply slice to frame range
            result = self._clone()
            result._frame_range = self._frame_range[item]
            return result
        else:
            raise TypeError('VideoFrames indices must be integers or slices.')

    def __len__(self) -> int:
        return len(self._frame_range) * self.repeat_count

    def __repr__(self) -> str:
        h, w = self.imshape
        label = self.path if isinstance(self.path, (str, Path)) else '<file-like>'
        return f"VideoFrames('{label}', {w}x{h}, {self.fps:.4g} fps, {len(self)} frames)"

    @property
    def imshape(self) -> tuple[int, int]:
        """Frame dimensions as (height, width) in pixels."""
        return self.resized_imshape if self.resized_imshape is not None else self.original_imshape

    @property
    def fps(self) -> float:
        """Effective frame rate, accounting for slicing and frame repetition."""
        return self.target_fps / self._frame_range.step * self.repeat_count

    def resized(self, shape: tuple[int, int]) -> 'VideoFrames':
        """Return a new VideoFrames that decodes frames at the given resolution.

        Args:
            shape: Target size as (height, width), following numpy/image convention.
                Note: this is the opposite order of ``video_extents()``, which
                returns (width, height).
        """
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(x, int) for x in shape)
        ):
            raise TypeError(f'shape must be a (height, width) tuple of two ints, got {shape!r}')
        result = self._clone()
        result.resized_imshape = shape
        return result

    def repeat_each_frame(self, n: int) -> 'VideoFrames':
        try:
            n = operator.index(n)
        except TypeError:
            raise TypeError(
                f'The repeat count must be an integer, got {type(n).__name__}'
            ) from None
        if n < 1:
            raise ValueError('The repeat count must be at least 1.')
        result = self._clone()
        result.repeat_count *= n
        return result

    def close(self) -> None:
        """Close the video reader. Call when done with this VideoFrames."""
        if self._reader is not None:
            self._reader.close()

    def __enter__(self) -> 'VideoFrames':
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _clone(self) -> 'VideoFrames':
        result = VideoFrames.__new__(VideoFrames)
        result.path = self.path
        result.original_imshape = self.original_imshape
        result.resized_imshape = self.resized_imshape
        result._frame_range = self._frame_range
        result.original_fps = self.original_fps
        result.repeat_count = self.repeat_count
        result.dtype = self.dtype
        result.gpu = self.gpu
        result.constant_framerate = self.constant_framerate
        result.target_fps = self.target_fps
        # Share index and CFR map with clones (read-only, thread-safe)
        result._index = self._index
        result._cfr_source_map = self._cfr_source_map
        result._is_fileobj = self._is_fileobj
        result._seekable = self._seekable
        result._pts_unreliable = self._pts_unreliable
        result._reader = None  # Each clone gets its own reader on iteration
        return result

    def _create_reader(self) -> PyAVReader:
        """Create a new reader for iteration."""
        if self._is_fileobj:
            if hasattr(self.path, 'getbuffer'):
                # BytesIO: give each reader an independent view so concurrently
                # active iterators can't disturb each other's read position
                # (costs one in-memory copy per reader).
                source = io.BytesIO(self.path.getbuffer())
            else:
                # Generic file-like object: all readers share its read position,
                # so only one iterator may be active at a time (see __init__).
                self.path.seek(0)
                source = self.path
        else:
            source = self.path
        reader = PyAVReader(source, gpu=self.gpu)
        reader._seekable = self._seekable
        return reader

    def _iter_decoded(
        self,
        reader: PyAVReader,
        frame_range: range,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Decode the sliced frames in internal dtype (no repetition, no conversion)."""
        slice_start = frame_range.start
        slice_stop = frame_range.stop
        slice_step = frame_range.step

        # Large step: more efficient to seek to each frame individually
        # Threshold is lower with PyAV since seeking is fast (~10ms vs ~100ms)
        if slice_step > 30:
            return self._iter_with_individual_seeks(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )

        # Use index-based seeking if we have an offset
        if slice_start > 0:
            return self._iter_with_seek(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )

        return self._iter_sequential(reader, slice_stop, slice_step, internal_dtype)

    def _iter_sequential(
        self,
        reader: PyAVReader,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Standard iteration from the beginning of the video (no seeking)."""
        reader.seek_to_time(Fraction(0))
        frames = self._decode_frames_cfr_aware(reader, internal_dtype)
        if slice_step == 1:
            return more_itertools.islice_extended(frames, slice_stop)
        return more_itertools.islice_extended(frames, 0, slice_stop, slice_step)

    def _iter_with_individual_seeks(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Iterate by seeking to each frame individually (efficient for large steps)."""
        for abs_idx in range(slice_start, slice_stop, slice_step):
            source_idx = self._abs_to_source(abs_idx)
            yield self._decode_frame_at_source(source_idx, internal_dtype, reader=reader)

    def _iter_with_seek(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Iterate with seeking to slice start."""
        if self.constant_framerate:
            yield from self._iter_with_seek_cfr(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )
            return

        max_frames = slice_stop - slice_start
        if max_frames <= 0:
            return

        # Get target PTS and seek to safe position (keyframe before target)
        target_pts_frac = self._index.get_frame_pts_fraction(slice_start)
        safe_pts_frac = self._index.safe_seek_pts[slice_start]
        reader.seek_to_time(safe_pts_frac)

        # Build filter graph
        target_format = 'rgb48' if internal_dtype == np.uint16 else 'rgb24'
        graph = reader._build_filter_graph(self.resized_imshape, target_format)

        # Skip frames until we reach the target PTS (mimics FFmpeg's -ss behavior)
        target_pts_float = float(target_pts_frac)
        time_base = reader.time_base
        reached_target = False

        frame_count = 0
        skip_count = 0
        for frame in reader.decode_raw():
            # Check if we've reached the target frame. Match by PTS if available,
            # otherwise by decoded-frame count (timestampless streams, e.g. raw
            # H.264 elementary streams, where the index has synthetic PTS and
            # decoding starts from frame 0).
            if not reached_target:
                usable_pts = frame.pts is not None and not self._pts_unreliable
                frame_pts = Fraction(frame.pts) * time_base if usable_pts else None
                if not (
                    (frame_pts is not None and float(frame_pts) >= target_pts_float - 1e-6)
                    or skip_count == slice_start
                ):
                    skip_count += 1
                    continue
                reached_target = True

            # Process frame through filter graph
            graph.push(frame)
            filtered_frame = graph.pull()

            if frame_count % slice_step == 0:
                yield filtered_frame.to_ndarray()

            frame_count += 1
            if frame_count >= max_frames:
                break

    def _iter_with_seek_cfr(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Iterate with seeking in CFR mode: walk the source map from the slice start."""
        source_map = self._cfr_source_map
        first_source = source_map[slice_start]
        safe_pts_frac = self._index.safe_seek_pts[first_source]
        target_pts_frac = self._index.frame_pts[first_source]
        reader.seek_to_time(safe_pts_frac)

        target_format = 'rgb48' if internal_dtype == np.uint16 else 'rgb24'
        graph = reader._build_filter_graph(self.resized_imshape, target_format)

        target_pts_float = float(target_pts_frac)
        time_base = reader.time_base

        # The map is non-decreasing, so the first output slot fed by first_source
        # is its leftmost occurrence. That can precede slice_start when the source
        # frame is duplicated across several output slots; emission is gated on
        # slice_start below.
        output_idx = bisect_left(source_map, first_source)
        source_idx = 0
        reached_target = False
        prev_frame_arr = None

        skip_count = 0
        for frame in reader.decode_raw():
            if not reached_target:
                # Match by PTS if available, otherwise by decoded-frame count
                # (timestampless streams, e.g. raw H.264 elementary streams).
                usable_pts = frame.pts is not None and not self._pts_unreliable
                frame_pts = Fraction(frame.pts) * time_base if usable_pts else None
                if not (
                    (frame_pts is not None and float(frame_pts) >= target_pts_float - 1e-6)
                    or skip_count == first_source
                ):
                    skip_count += 1
                    continue
                reached_target = True
                source_idx = first_source

            graph.push(frame)
            filtered_frame = graph.pull()
            frame_arr = filtered_frame.to_ndarray()

            # Output this frame for all CFR output indices that map to this source
            while output_idx < len(source_map) and source_map[output_idx] == source_idx:
                if slice_start <= output_idx < slice_stop:
                    if (output_idx - slice_start) % slice_step == 0:
                        yield frame_arr
                output_idx += 1
                if output_idx >= slice_stop:
                    return

            prev_frame_arr = frame_arr
            source_idx += 1

        # Handle remaining output frames (EOF duplication, or a truncated stream
        # that decoded fewer frames than the index recorded)
        while output_idx < len(source_map) and output_idx < slice_stop:
            if (
                prev_frame_arr is not None
                and slice_start <= output_idx
                and (output_idx - slice_start) % slice_step == 0
            ):
                yield prev_frame_arr
            output_idx += 1

    def _decode_frames_cfr_aware(
        self, reader: PyAVReader, dtype: DTypeLike
    ) -> Generator[NDArray, None, None]:
        """Decode frames from the start, with CFR simulation if enabled."""
        if not self.constant_framerate:
            # VFR mode: pass through directly
            yield from reader.decode_frames(
                output_shape=self.resized_imshape,
                dtype=dtype,
            )
            return

        # CFR mode: walk the source map from the beginning
        source_map = self._cfr_source_map
        target_format = 'rgb48' if dtype == np.uint16 else 'rgb24'

        # Build filter graph for exact FFmpeg compatibility
        graph = reader._build_filter_graph(self.resized_imshape, target_format)

        source_idx = 0
        output_idx = 0
        prev_frame_arr = None

        for frame in reader.decode_raw():
            # Process through filter graph for exact color conversion
            graph.push(frame)
            filtered_frame = graph.pull()
            frame_arr = filtered_frame.to_ndarray()

            # Output this frame for all output indices that map to this source index
            while output_idx < len(source_map) and source_map[output_idx] == source_idx:
                yield frame_arr
                output_idx += 1

            prev_frame_arr = frame_arr
            source_idx += 1

        # Handle any remaining output frames (EOF duplication)
        while output_idx < len(source_map):
            if prev_frame_arr is not None:
                yield prev_frame_arr
            output_idx += 1

    def _abs_to_source(self, abs_idx: int) -> int:
        """Map an absolute output-frame index to the source frame that fills it."""
        if self.constant_framerate:
            return self._cfr_source_map[abs_idx]
        return abs_idx

    def _decode_frame_at_source(
        self,
        source_idx: int,
        internal_dtype: DTypeLike | None = None,
        reader: PyAVReader | None = None,
    ) -> NDArray:
        """Seek to and decode a single source frame (in internal dtype, unconverted).

        Args:
            source_idx: Source frame index in the original video.
            internal_dtype: Internal dtype for decoding (uint8 or uint16).
            reader: Optional reader to use. If None, creates a temporary one.
        """
        if internal_dtype is None:
            internal_dtype = np.uint8 if self.dtype == np.uint8 else np.uint16

        # Get safe seek point and target PTS
        safe_pts_frac = self._index.safe_seek_pts[source_idx]
        target_pts_frac = self._index.frame_pts[source_idx]

        # Use provided reader or create temporary one
        own_reader = reader is None
        if own_reader:
            reader = self._create_reader()

        try:
            # Seek to safe point (keyframe before target)
            reader.seek_to_time(safe_pts_frac)

            # Build filter graph for exact FFmpeg compatibility
            target_format = 'rgb48' if internal_dtype == np.uint16 else 'rgb24'
            graph = reader._build_filter_graph(self.resized_imshape, target_format)

            # Decode frames until we reach the target PTS
            target_pts_float = float(target_pts_frac)
            time_base = reader.time_base

            frame_count = 0
            for frame in reader.decode_raw():
                usable_pts = frame.pts is not None and not self._pts_unreliable
                frame_pts = Fraction(frame.pts) * time_base if usable_pts else None
                # Match by PTS if available, otherwise by frame count (for attached pictures etc.)
                if (
                    frame_pts is not None and float(frame_pts) >= target_pts_float - 1e-6
                ) or frame_count == source_idx:
                    graph.push(frame)
                    filtered_frame = graph.pull()
                    return filtered_frame.to_ndarray()
                frame_count += 1

            raise VideoDecodeError(
                self.path, source_idx, RuntimeError(f'Failed to decode frame {source_idx}')
            )
        finally:
            if own_reader:
                reader.close()

    def _seek_reproduces_sequential(self) -> bool:
        """Check whether seek-based access decodes the same pixels as iteration.

        Compares a few early frames plus the frames around the first real seek
        point (the first index whose safe seek target differs from the start),
        decoded via the seek path, against sequential decode from the start.
        Bounded by roughly one GOP of sequential decoding.
        """
        n = self._index.frame_count
        first_safe = self._index.safe_seek_pts[0]
        j0 = next((i for i in range(1, n) if self._index.safe_seek_pts[i] != first_safe), 1)
        probe = sorted({i for i in (1, 2, j0, j0 + 1, j0 + 2) if 0 < i < n})
        if not probe:
            return True

        try:
            reader = self._create_reader()
            try:
                # Same starting procedure as sequential iteration
                reader.seek_to_time(Fraction(0))
                sequential = {}
                for i, arr in enumerate(reader.decode_frames(max_frames=probe[-1] + 1)):
                    if i in probe:
                        sequential[i] = arr
            finally:
                reader.close()

            for i in probe:
                if i not in sequential:
                    # The decoder yielded fewer frames than the index recorded
                    return False
                if not np.array_equal(sequential[i], self._decode_frame_at_source(i, np.uint8)):
                    return False
        except FramePumpError:
            return False
        return True

    def _rebuild_index_from_decode(self) -> None:
        """Rebuild the frame index from actual decoder output (one full decode).

        Used when packet-based indexing is untrustworthy: the count and PTS
        list then reflect the frames the decoder really produces, so ``len()``,
        indexing and iteration agree even when some packets are undecodable.
        """
        pts_list: list[Fraction | None] = []
        reader = self._create_reader()
        try:
            # Start decoding exactly like sequential iteration does (reopens
            # non-seekable containers), so the rebuilt index reflects it
            reader.seek_to_time(Fraction(0))
            time_base = reader.time_base
            try:
                for frame in reader.decode_raw():
                    pts_list.append(
                        Fraction(frame.pts) * time_base if frame.pts is not None else None
                    )
            except VideoDecodeError:
                # Keep the frames decoded before the error (truncated-file
                # semantics, same as sequential iteration would surface them)
                pass
        finally:
            reader.close()

        n = len(pts_list)
        monotonic = n > 0 and all(p is not None for p in pts_list)
        if monotonic:
            monotonic = all(b >= a for a, b in zip(pts_list, pts_list[1:]))
        if not monotonic:
            # Unusable timestamps: synthesize evenly spaced PTS and switch
            # frame matching to counting, since decoded PTS can't locate frames
            fps = (
                Fraction(self.original_fps).limit_denominator(1000000)
                if self.original_fps
                else Fraction(25)
            )
            start = pts_list[0] if n and pts_list[0] is not None else Fraction(0)
            pts_list = [start + Fraction(i) / fps for i in range(n)]
            self._pts_unreliable = True

        self._index.frame_pts = pts_list
        self._index.safe_seek_pts = [Fraction(0)] * n
        self._index.frame_count = n

    def _build_cfr_source_map(self) -> list[int]:
        """Build the mapping from CFR output frame index to source frame index.

        Simulates FFmpeg's vsync=1 algorithm to determine which source frame is
        displayed at each output position. This map is the single source of truth
        for CFR mode: frame count is its length, and indexing, iteration, and
        seeking all read from it, so they cannot disagree with each other.

        The arithmetic deliberately uses floats: FFmpeg computes vsync in doubles,
        so float — not exact rational — arithmetic reproduces its output. Like
        ffmpeg, sync values stay unrounded (only the frame counts are rounded)
        and each source frame carries its real duration rescaled to the output
        timebase — both matter when the target fps differs from the source fps.
        PTS are taken relative to the first frame (as ffmpeg does via the stream
        start time), so streams that start late (e.g. MPEG-TS) produce no
        phantom leading frames.
        """
        fps = self.target_fps
        frame_pts = self._index.frame_pts
        start_pts = frame_pts[0] if frame_pts else Fraction(0)

        # Convert PTS to output timebase (frame units), mirroring ffmpeg's
        # adjust_frame_pts_to_encoder_tb: rescale the exact rational PTS into
        # the output timebase with 16 extra precision bits (round half away
        # from zero, like av_rescale_q's AV_ROUND_NEAR_INF for non-negative
        # values); non-integer results are then biased by 2^-17 away from zero
        # to avoid exact midpoints in the frame-count rounding below (integers
        # are left exact so on-boundary frames round half-to-even like ffmpeg).
        fps_frac = Fraction(fps)
        raw_ipts = [(pts - start_pts) * fps_frac for pts in frame_pts]
        scale = 1 << 16
        eps = 1.0 / (1 << 17)
        sync_ipts_list = []
        for v in raw_ipts:
            q = int(v * scale + Fraction(1, 2)) / scale
            sync_ipts_list.append(q if q == int(q) else q + eps)

        # Per-frame duration in output timebase: delta to the next PTS; the last
        # frame reuses the previous delta (single frame: one output slot).
        durations = [float(b - a) for a, b in zip(raw_ipts, raw_ipts[1:])]
        durations.append(durations[-1] if durations else 1.0)

        next_pts = 0
        source_map = []  # source_map[output_idx] = source_idx
        frames_prev_hist = deque([0, 0, 0], maxlen=3)  # History for EOF median

        for source_idx, sync_ipts in enumerate(sync_ipts_list):
            delta0 = sync_ipts - next_pts
            delta = delta0 + durations[source_idx]

            # ffmpeg clips frames that arrive slightly early but still within
            # their duration ("Clipping frame in rate conversion"): the drift
            # is zeroed while delta keeps its value.
            if delta0 < 0 < delta:
                delta0 = 0

            nb_frames = 1
            nb_frames_prev = 0

            if delta < -1.1:
                nb_frames = 0
            elif delta > 1.1:
                nb_frames = round(delta)
                if delta0 > 1.1:
                    nb_frames_prev = round(delta0 - 0.6)

            # Output nb_frames_prev copies of PREVIOUS source frame
            for _ in range(nb_frames_prev):
                if source_idx > 0:
                    source_map.append(source_idx - 1)
                next_pts += 1

            # Output (nb_frames - nb_frames_prev) copies of CURRENT source frame
            for _ in range(nb_frames - nb_frames_prev):
                source_map.append(source_idx)
                next_pts += 1

            frames_prev_hist.appendleft(nb_frames_prev)

        # EOF handling: output median of last 3 nb_frames_prev values
        eof_frames = sorted(frames_prev_hist)[1]
        for _ in range(eof_frames):
            source_map.append(len(frame_pts) - 1)

        return source_map

    def _maybe_to_float(self, value: NDArray) -> NDArray:
        if self.dtype == np.uint8 or self.dtype == np.uint16:
            return value

        maxval = np.iinfo(value.dtype).max
        if self.dtype == np.float16:
            # float16 cannot represent the division in-type (uint16 values above
            # ~65519 overflow to inf), so normalize in float32 first.
            return (value.astype(np.float32) / maxval).astype(np.float16)
        return value.astype(self.dtype) / maxval


def num_frames(path: PathLike, exact: bool = False, absolutely_exact: bool = False) -> int:
    """Count frames in a video.

    Args:
        path: Path to video file.
        exact: Use frame index for exact count (builds packet index).
        absolutely_exact: Count by iterating all frames (slowest but most accurate).

    Returns:
        Number of frames in the video.
    """
    if absolutely_exact:
        # Count by actually iterating all frames
        with VideoFrames(path) as frames:
            return more_itertools.ilen(frames)

    if exact:
        # Use frame index for exact count
        index = FrameIndexPyAV(path)
        return index.frame_count

    return int(round(get_duration(path) * get_fps(path)))


def get_fps(video_path: PathLike) -> float:
    """Get video frame rate using PyAV."""
    with PyAVReader(video_path) as reader:
        return reader.fps


def get_duration(video_path: PathLike) -> float:
    """Get video duration in seconds using PyAV."""
    with PyAVReader(video_path) as reader:
        return reader.duration


def video_extents(filepath: PathLike) -> NDArray:
    """Returns the video (width, height) as a numpy array, without loading the pixel data.

    Note: this returns (width, height), which is the opposite of
    ``VideoFrames.imshape`` and numpy array shapes that use (height, width).
    """
    with PyAVReader(filepath) as reader:
        return np.array(reader.resolution)


def has_audio(video_path: PathLike) -> bool:
    """Check if video has an audio stream using PyAV."""
    with PyAVReader(video_path) as reader:
        return reader.has_audio()
