from __future__ import annotations

from collections import deque
from collections.abc import Generator
from fractions import Fraction
from pathlib import Path
from typing import Union, overload

import more_itertools
import numpy as np
import simplepyutils as spu
from numpy.typing import DTypeLike, NDArray

from ._pyav import FrameIndexPyAV, PyAVReader, VideoDecodeError

PathLike = Union[str, Path]

__all__ = [
    'VideoFrames',
    'get_fps',
    'get_duration',
    'num_frames',
    'video_extents',
    'has_audio',
]


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
        video_path: PathLike,
        dtype: DTypeLike = np.uint8,
        gpu: bool | int = False,
        constant_framerate: Union[bool, float] = False,
    ) -> None:
        """Open a video file for lazy frame access.

        See class docstring for full parameter descriptions.
        """
        self.path = video_path

        # Create persistent PyAV reader for metadata and decoding
        self._reader = PyAVReader(video_path, gpu=gpu)

        # Get metadata from reader (no subprocess calls)
        width, height = self._reader.resolution
        self.original_imshape: tuple[int, int] = (height, width)
        self.original_fps = self._reader.fps
        self._original_fps_frac = self._reader.fps_fraction
        self.resized_imshape: tuple[int, int] | None = None
        self.repeat_count = 1

        if dtype not in (np.uint8, np.uint16, np.float16, np.float32, np.float64):
            raise ValueError(f'Unsupported dtype: {dtype}')

        self.dtype = dtype
        self.gpu = gpu

        # Parse constant_framerate: False, True, or a number (target fps)
        if isinstance(constant_framerate, bool):
            self.constant_framerate = constant_framerate
            self.target_fps = self.original_fps
            self._target_fps_frac = self._original_fps_frac
        else:
            self.constant_framerate = True
            self.target_fps = float(constant_framerate)
            self._target_fps_frac = Fraction(constant_framerate).limit_denominator(100000)

        # Build frame index upfront
        self._index = FrameIndexPyAV(self.path, reader=self._reader)

        # Close the reader — it gets reopened lazily in __iter__
        self._reader.close()
        self._reader = None

        # Compute total frame count
        if self.constant_framerate:
            n_frames = self._count_cfr_frames()
        else:
            n_frames = self._index.frame_count

        # Store frame range - slicing applies directly to this
        self._frame_range: range = range(n_frames)

    def __iter__(self) -> Generator[NDArray, None, None]:
        internal_dtype = np.uint8 if self.dtype == np.uint8 else np.uint16

        frame_range = self._frame_range
        if len(frame_range) == 0:
            return

        slice_start = frame_range.start
        slice_stop = frame_range.stop
        slice_step = frame_range.step

        # Create a fresh reader for this iteration
        reader = self._create_reader()
        try:
            # Large step: more efficient to seek to each frame individually
            # Threshold is lower with PyAV since seeking is fast (~10ms vs ~100ms)
            if slice_step > 30:
                yield from self._iter_with_individual_seeks(
                    reader, slice_start, slice_stop, slice_step, internal_dtype
                )
                return

            # Use index-based seeking if we have an offset
            if slice_start > 0:
                yield from self._iter_with_seek(
                    reader, slice_start, slice_stop, slice_step, internal_dtype
                )
                return

            # Standard iteration (no seeking) - use PyAV directly
            reader.seek_to_time(Fraction(0))
            frames = self._decode_frames_cfr_aware(reader, internal_dtype)

            # Apply step and stop
            if slice_step == 1:
                sliced_frames = more_itertools.islice_extended(frames, slice_stop)
            else:
                sliced_frames = more_itertools.islice_extended(frames, 0, slice_stop, slice_step)

            cast_frames = map(self._maybe_to_float, sliced_frames)
            if self.repeat_count == 1:
                yield from cast_frames
            else:
                yield from spu.repeat_n(cast_frames, self.repeat_count)
        finally:
            reader.close()

    @overload
    def __getitem__(self, item: int) -> NDArray:
        ...

    @overload
    def __getitem__(self, item: slice) -> VideoFrames:
        ...

    def __getitem__(self, item: int | slice) -> NDArray | VideoFrames:
        if isinstance(item, int):
            # Handle negative indices
            length = len(self)
            if item < 0:
                item = length + item
            if item < 0 or item >= length:
                raise IndexError(f'Frame index {item} out of range for video with {length} frames')

            # Get absolute frame index from the range
            abs_idx = self._frame_range[item]
            return self._get_frame_indexed(abs_idx)
        elif isinstance(item, slice):
            if self.repeat_count != 1:
                raise NotImplementedError(
                    'Slicing after repeat_each_frame() is not supported. '
                    'Apply slicing before repeat_each_frame(), e.g. '
                    'frames[::2].repeat_each_frame(3) instead of '
                    'frames.repeat_each_frame(3)[::2].')

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
        return f"VideoFrames('{self.path}', {w}x{h}, {self.fps:.4g} fps, {len(self)} frames)"

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
        if (not isinstance(shape, tuple) or len(shape) != 2
                or not all(isinstance(x, int) for x in shape)):
            raise TypeError(
                f'shape must be a (height, width) tuple of two ints, got {shape!r}')
        result = self._clone()
        result.resized_imshape = shape
        return result

    def repeat_each_frame(self, n: int) -> 'VideoFrames':
        if n < 1:
            raise ValueError('The repeat count must be at least 1.')
        result = self._clone()
        result.repeat_count *= n
        return result

    def _clone(self) -> 'VideoFrames':
        result = VideoFrames.__new__(VideoFrames)
        result.path = self.path
        result.original_imshape = self.original_imshape
        result.resized_imshape = self.resized_imshape
        result._frame_range = self._frame_range
        result.original_fps = self.original_fps
        result._original_fps_frac = self._original_fps_frac
        result.repeat_count = self.repeat_count
        result.dtype = self.dtype
        result.gpu = self.gpu
        result.constant_framerate = self.constant_framerate
        result.target_fps = self.target_fps
        result._target_fps_frac = self._target_fps_frac
        # Share index with clones (read-only, thread-safe)
        result._index = self._index
        result._reader = None  # Each clone gets its own reader on iteration
        return result

    def _create_reader(self) -> PyAVReader:
        """Create a new reader for iteration."""
        return PyAVReader(self.path, gpu=self.gpu)

    def close(self) -> None:
        """Close the video reader. Call when done with this VideoFrames."""
        if self._reader is not None:
            self._reader.close()

    def __enter__(self) -> 'VideoFrames':
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _iter_with_individual_seeks(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Iterate by seeking to each frame individually (efficient for large steps)."""
        for idx in range(slice_start, slice_stop, slice_step):
            frame = self._get_frame_indexed(idx, internal_dtype, reader=reader)
            if self.repeat_count == 1:
                yield frame
            else:
                for _ in range(self.repeat_count):
                    yield frame

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
        for frame in reader._container.decode(reader._stream):
            # Check if we've reached the target frame
            if not reached_target:
                frame_pts = Fraction(frame.pts) * time_base if frame.pts is not None else None
                if frame_pts is None or float(frame_pts) < target_pts_float - 1e-6:
                    continue  # Skip this frame
                reached_target = True

            # Process frame through filter graph
            graph.push(frame)
            filtered_frame = graph.pull()
            arr = filtered_frame.to_ndarray()

            if frame_count % slice_step == 0:
                converted = self._maybe_to_float(arr)
                if self.repeat_count == 1:
                    yield converted
                else:
                    for _ in range(self.repeat_count):
                        yield converted

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
        """Iterate with seeking in CFR mode.

        Maps output indices through the CFR source map and seeks to the
        first needed source frame.
        """
        source_map = self._build_cfr_source_map()

        # Find the range of source frames we need
        first_source = source_map[slice_start] if slice_start < len(source_map) else 0
        safe_pts_frac = self._index.safe_seek_pts[first_source]
        target_pts_frac = self._index.frame_pts[first_source]
        reader.seek_to_time(safe_pts_frac)

        target_format = 'rgb48' if internal_dtype == np.uint16 else 'rgb24'
        graph = reader._build_filter_graph(self.resized_imshape, target_format)

        target_pts_float = float(target_pts_frac)
        time_base = reader.time_base

        source_idx = 0
        output_idx = 0
        reached_target = False
        prev_frame_arr = None

        for frame in reader._container.decode(reader._stream):
            if not reached_target:
                frame_pts = Fraction(frame.pts) * time_base if frame.pts is not None else None
                if frame_pts is None or float(frame_pts) < target_pts_float - 1e-6:
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
                        converted = self._maybe_to_float(frame_arr)
                        if self.repeat_count == 1:
                            yield converted
                        else:
                            for _ in range(self.repeat_count):
                                yield converted
                output_idx += 1
                if output_idx >= slice_stop:
                    return

            prev_frame_arr = frame_arr
            source_idx += 1

        # Handle remaining output frames (EOF duplication)
        while output_idx < len(source_map) and output_idx < slice_stop:
            if prev_frame_arr is not None and (output_idx - slice_start) % slice_step == 0:
                converted = self._maybe_to_float(prev_frame_arr)
                if self.repeat_count == 1:
                    yield converted
                else:
                    for _ in range(self.repeat_count):
                        yield converted
            output_idx += 1

    def _get_frame_indexed(
        self,
        abs_idx: int,
        internal_dtype: DTypeLike | None = None,
        reader: PyAVReader | None = None,
    ) -> NDArray:
        """Get frame using index for fast seeking.

        Args:
            abs_idx: Absolute frame index in the original video.
            internal_dtype: Internal dtype for decoding (uint8 or uint16).
            reader: Optional reader to use. If None, creates a temporary one.
        """
        if internal_dtype is None:
            internal_dtype = np.uint8 if self.dtype == np.uint8 else np.uint16

        if self.constant_framerate:
            # CFR mode: find source frame using FFmpeg's vsync algorithm
            source_idx = self._find_source_frame_for_cfr_output(abs_idx)
        else:
            source_idx = abs_idx

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
            for frame in reader._container.decode(reader._stream):
                frame_pts = Fraction(frame.pts) * time_base if frame.pts is not None else None
                # Match by PTS if available, otherwise by frame count (for attached pictures etc.)
                if (frame_pts is not None and float(
                    frame_pts) >= target_pts_float - 1e-6) or frame_count == source_idx:
                    graph.push(frame)
                    filtered_frame = graph.pull()
                    return self._maybe_to_float(filtered_frame.to_ndarray())
                frame_count += 1

            raise VideoDecodeError(
                self.path, abs_idx,
                RuntimeError(f'Failed to decode frame {abs_idx}'))
        finally:
            if own_reader:
                reader.close()

    def _find_source_frame_for_cfr_output(self, output_idx: int) -> int:
        """Find which source frame is displayed at CFR output index."""
        source_map = self._build_cfr_source_map()
        if output_idx < len(source_map):
            return source_map[output_idx]
        return len(self._index.frame_pts) - 1

    def _build_cfr_source_map(self) -> list[int]:
        """Build mapping from CFR output frame index to source frame index.

        Simulates FFmpeg's vsync=1 algorithm to determine which source frame
        is displayed at each output position.
        """
        if hasattr(self, '_cfr_source_map'):
            return self._cfr_source_map

        fps = self.target_fps
        frame_pts = self._index.frame_pts

        # Convert PTS to output timebase (integer frame units)
        sync_ipts_list = [round(pts * fps) for pts in frame_pts]
        duration = 1  # Each source frame has duration=1 in output timebase

        next_pts = 0
        source_map = []  # source_map[output_idx] = source_idx
        frames_prev_hist = deque([0, 0, 0], maxlen=3)  # History for EOF median

        for source_idx, sync_ipts in enumerate(sync_ipts_list):
            delta0 = sync_ipts - next_pts
            delta = delta0 + duration

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

        self._cfr_source_map = source_map
        return source_map

    def _decode_frames_cfr_aware(
        self, reader: PyAVReader, dtype: DTypeLike, max_frames: int | None = None
    ) -> Generator[NDArray, None, None]:
        """Decode frames with CFR simulation if enabled.

        If constant_framerate is True, simulates FFmpeg's vsync=1 algorithm
        to duplicate/drop frames as needed.
        """
        if not self.constant_framerate:
            # VFR mode: pass through directly
            yield from reader.decode_frames(
                max_frames=max_frames,
                output_shape=self.resized_imshape,
                dtype=dtype,
            )
            return

        # CFR mode: simulate vsync=1 algorithm
        source_map = self._build_cfr_source_map()
        target_format = 'rgb48' if dtype == np.uint16 else 'rgb24'

        # Build filter graph for exact FFmpeg compatibility
        graph = reader._build_filter_graph(self.resized_imshape, target_format)

        source_idx = 0
        output_idx = 0
        prev_frame_arr = None

        for frame in reader._container.decode(reader._stream):
            # Process through filter graph for exact color conversion
            graph.push(frame)
            filtered_frame = graph.pull()
            frame_arr = filtered_frame.to_ndarray()

            # Output this frame for all output indices that map to this source index
            while output_idx < len(source_map) and source_map[output_idx] == source_idx:
                yield frame_arr
                output_idx += 1
                if max_frames is not None and output_idx >= max_frames:
                    return

            prev_frame_arr = frame_arr
            source_idx += 1

        # Handle any remaining output frames (EOF duplication)
        while output_idx < len(source_map):
            if prev_frame_arr is not None:
                yield prev_frame_arr
            output_idx += 1
            if max_frames is not None and output_idx >= max_frames:
                return

    def _count_cfr_frames(self) -> int:
        """Count frames that will be output in CFR mode (vsync=1 simulation)."""
        fps_frac = self._target_fps_frac
        frame_pts_list = self._index.frame_pts  # List of Fraction

        # Convert PTS to output timebase (integer frame units)
        sync_ipts_list = [round(float(pts * fps_frac)) for pts in frame_pts_list]
        duration = 1  # Each source frame has duration=1 in output timebase

        next_pts = 0
        total_frames = 0
        frames_prev_hist = [0, 0, 0]

        for sync_ipts in sync_ipts_list:
            delta0 = sync_ipts - next_pts
            delta = delta0 + duration

            nb_frames = 1
            nb_frames_prev = 0

            if delta < -1.1:
                nb_frames = 0
            elif delta > 1.1:
                nb_frames = round(delta)
                if delta0 > 1.1:
                    nb_frames_prev = round(delta0 - 0.6)

            total_frames += nb_frames
            next_pts += nb_frames
            frames_prev_hist = [nb_frames_prev] + frames_prev_hist[:2]

        # At EOF, FFmpeg outputs median of last 3 nb_frames_prev values
        eof_frames = sorted(frames_prev_hist)[1]
        total_frames += eof_frames

        return total_frames

    def _maybe_to_float(self, value: NDArray) -> NDArray:
        if self.dtype == np.uint8 or self.dtype == np.uint16:
            return value

        if value.dtype == np.uint16 and self.dtype == np.float16:
            x = value.clip(0, 65504).astype(np.float16)
            x /= 65504.0
            return x

        maxval = np.iinfo(value.dtype).max
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


