from __future__ import annotations

import io
import itertools
import operator
import threading
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
from ._selection import FrameSelection

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
# MPEG-1/2, packed-B MPEG-4). Seeking is content-verified for these when the
# index is built and disabled when it does not reproduce sequential decode.
_SEEK_UNRELIABLE_CODECS = frozenset(
    {
        'mpeg1video',
        'mpeg2video',
        'mpeg4',
        'fic',
        'vmnc',
        'cavs',
        'ansi',
        'rv10',
        'rv20',
        'rv30',
        'rv40',
        'cinepak',
        'cscd',
    }
)


class _CountingIterator:
    """Iterator wrapper counting consumed items and noticing exhaustion."""

    __slots__ = ('_it', 'count', 'exhausted')

    def __init__(self, iterable) -> None:
        self._it = iter(iterable)
        self.count = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._it)
        except StopIteration:
            self.exhausted = True
            raise
        self.count += 1
        return item


class _SeekPathUnsound(Exception):
    """Internal: a PTS regression was observed during a seek-based access
    before the target frame was delivered; the caller retries sequentially."""


class _RetrySequential(Exception):
    """Internal: decoding after a real seek failed before anything was
    delivered; the caller disables seeking for the file and retries."""


# Deepest frame the seek-reliability probe references when a file claims
# dense keyframes; bounds the probe's sequential reference decode.
_PROBE_DEEP_MAX = 128

# Streaming (decode-from-start-and-skip, no index) is used for forward
# selections whose start is at most this many frames; larger starts build the
# index and seek, which amortizes better than decoding and discarding.
_STREAM_MAX_SKIP = 256

# Reverse iteration buffers one chunk of decoded frames at a time; the chunk
# frame count is derived from this budget (clamped to [4, 64] frames).
_REVERSE_CHUNK_BYTES = 256 * 1024 * 1024


class _LazyIndexState:
    """Frame-index state shared between a VideoFrames and all its views.

    Views are created by slicing; they must observe the same index, CFR map
    and seek-reliability verdicts no matter which view triggered the build.
    """

    __slots__ = (
        'lock',
        'index',
        'cfr_source_map',
        'seek_disabled',
        'pts_unreliable',
        'emission_disorder',
        'observed_seq_count',
        'seekable_probed',
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index: FrameIndexPyAV | None = None
        self.cfr_source_map: list[int] | None = None
        # True when the seek-reliability probe failed: access is sequential-only
        self.seek_disabled = False
        # True when decoded frame PTS cannot be trusted for locating frames
        # (the index had to synthesize timestamps); frame counting is used
        self.pts_unreliable = False
        # True when a display-order PTS regression was observed in decoder
        # output: sorted-PTS frame location is unsound for this video
        self.emission_disorder = False
        # Total frame count observed by an index-less streaming pass that
        # exhausted the decoder (ground truth to reconcile the index against)
        self.observed_seq_count: int | None = None
        # Cached container-seekability auto-probe verdict (None = not probed)
        self.seekable_probed: bool | None = None


class VideoFrames:
    """Lazy, sliceable video frame iterator.

    Frames are only decoded when iterated. Slicing and resizing are lazy operations
    that return new VideoFrames instances without loading pixel data.

    Output is always numpy arrays in CPU memory. ``gpu=True`` changes only
    where the *decoding* happens (NVDEC instead of libavcodec); each frame is
    still downloaded, and the pixels are bit-identical either way. For frames
    that stay in GPU memory, use ``VideoFramesCuda`` instead.

    Example:
        >>> frames = VideoFrames('video.mp4')
        >>> for frame in frames[::2][:100].resized((128, 128)):
        ...     process(frame)

    Args:
        video_path: Path to video file.
        dtype: Output dtype (uint8, uint16, float16, float32, float64).
        gpu: False for CPU decoding, True for NVDEC hardware decoding on the
            default GPU, or an int to select a specific GPU device ordinal.
            Frames are decoded on the GPU and downloaded to numpy arrays;
            output is bit-identical to CPU decoding. Codecs NVDEC cannot
            handle raise an error instead of silently falling back (use
            VideoFramesCuda for frames that stay in GPU memory).
        constant_framerate: False for VFR (native timestamps), True for CFR at
            original fps, or a number for CFR at that specific fps.
        seekable: Override seek-support detection: False forces sequential
            decode-from-start access (for streams where seeking is unreliable
            or unsupported), True skips the automatic probing and assumes
            seeking works. None (default) auto-detects.
        gray: Decode to single-channel grayscale: frames are (height, width)
            instead of (height, width, 3). With dtype=np.uint16 the decode is
            bit-exact for gray16le sources (e.g. FFV1 depth videos written by
            DepthVideoWriter). Not supported together with gpu decoding.
    """

    def __init__(
        self,
        video_path,
        *,
        dtype: DTypeLike = np.uint8,
        gpu: bool | int = False,
        constant_framerate: Union[bool, float] = False,
        seekable: bool | None = None,
        gray: bool = False,
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

        if gray and gpu:
            raise ValueError('gray=True is not supported together with gpu decoding')
        self.gray = gray

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

            # The caller's explicit seekability (None = auto-probe lazily on
            # first reader use; probing seeks and decodes a test frame, too
            # expensive for construction)
            self._seekable: bool | None = seekable
            self._codec_name: str = self._reader.codec_name
        finally:
            # Readers are created on demand for iteration and index building
            self._reader.close()
            self._reader = None

        # The frame index (full packet scan of the file) is built lazily: it
        # is only needed for length-dependent access (len(), integer indexing,
        # negative slice components, reverse iteration) and for CFR mode.
        # Plain forward iteration and prefix-style slicing stream without it.
        # The state is shared with all views cloned from this instance.
        self._lazy = _LazyIndexState()

        # Which frames this view selects; symbolic until the count is known
        self._selection = FrameSelection.identity()

    def __iter__(self) -> Generator[NDArray, None, None]:
        """Decode and yield the selected frames in order.

        Each frame is a numpy array of shape (height, width, 3), or
        (height, width) in grayscale mode, with the configured dtype.
        """
        internal_dtype = np.uint8 if self.dtype == np.uint8 else np.uint16

        # Stream without the index when the selection is a plain forward
        # slice with a small start (CFR always needs the index: all CFR
        # behavior derives from the single source map). If the index already
        # exists, the seek-based paths are at least as good — use them.
        if not self._selection.is_resolved and not self.constant_framerate:
            streamable = self._selection.streamable_slice
            if (
                streamable is not None
                and (streamable.start or 0) <= _STREAM_MAX_SKIP
                and self._lazy.index is None
            ):
                yield from self._iter_streamed(streamable, internal_dtype)
                return

        frame_range = self._resolved_range()
        if len(frame_range) == 0:
            return

        # Create a fresh reader for this iteration
        reader = self._create_reader()
        try:
            raw_frames = self._iter_decoded(reader, frame_range, internal_dtype)
            count = 0
            for frame in self._convert_and_repeat(raw_frames):
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
            if reader.pts_regression_seen and not self._lazy.seek_disabled:
                # Observed in-band and remembered, so future seek-based access
                # is served from decoder output instead of sorted PTS
                self._flag_emission_disorder()
        finally:
            reader.close()

    def _iter_streamed(
        self, streamable: slice, internal_dtype: DTypeLike
    ) -> Generator[NDArray, None, None]:
        """Decode from the start and skip, without building the frame index."""
        start = streamable.start or 0
        step = streamable.step or 1

        count = 0
        reader = self._create_reader()
        try:
            reader.seek_to_time(Fraction(0))
            raw = reader.decode_frames(
                output_shape=self.resized_imshape,
                dtype=internal_dtype,
                target_format=self._pix_target(internal_dtype),
            )
            # Count every decoded frame (not just yielded ones): when this
            # pass exhausts the decoder, the count is the file's true frame
            # count and is used to reconcile the packet index (packets and
            # frames are not always 1:1 - multi-frame packets, flush frames,
            # no-op packets). Costs one counter increment per frame.
            decoded = _CountingIterator(raw)
            sliced = itertools.islice(decoded, start, streamable.stop, step)
            for frame in self._convert_and_repeat(sliced):
                yield frame
                count += 1
            if decoded.exhausted:
                self._lazy.observed_seq_count = decoded.count
            if reader.pts_regression_seen and not self._lazy.seek_disabled:
                self._flag_emission_disorder()
        finally:
            reader.close()

        if count == 0 and len(self) > 0:
            # Same guarantee as the resolved path: an empty decode of a
            # nonempty selection must not look like a valid empty view.
            # Building the index here is cheap — the stream ended immediately.
            raise VideoDecodeError(
                self.path,
                0,
                RuntimeError(
                    f'Decoder produced no frames, but the video index '
                    f'recorded {len(self)} for this range'
                ),
            )

    def _convert_and_repeat(self, raw_frames):
        frames = map(self._maybe_to_float, raw_frames)
        if self.repeat_count == 1:
            return frames
        return spu.repeat_n(frames, self.repeat_count)

    def _pix_target(self, internal_dtype: DTypeLike) -> str:
        """Decode target pixel format: RGB by default, grayscale for gray=True."""
        if self.gray:
            return 'gray16le' if internal_dtype == np.uint16 else 'gray'
        return 'rgb48' if internal_dtype == np.uint16 else 'rgb24'

    def _resolved_range(self) -> range:
        """The concrete source-index range, resolving the selection if needed."""
        if not self._selection.is_resolved:
            self._selection = self._selection.resolve(self._n_frames_total())
        return self._selection.range

    def _n_frames_total(self) -> int:
        if self.constant_framerate:
            return len(self._cfr_source_map)
        return self._index.frame_count

    @property
    def _index(self) -> FrameIndexPyAV:
        if self._lazy.index is None:
            self._materialize_index()
        return self._lazy.index

    @property
    def _cfr_source_map(self) -> list[int] | None:
        if self.constant_framerate and self._lazy.index is None:
            self._materialize_index()
        return self._lazy.cfr_source_map

    @property
    def _pts_unreliable(self) -> bool:
        return self._lazy.pts_unreliable

    def _materialize_index(self) -> None:
        """Build the frame index (full packet scan), shared with all views.

        Also runs the seek-reliability verification for suspect codecs and
        builds the CFR source map, so every consumer of the index sees the
        same verdicts regardless of which view triggered the build.
        """
        with self._lazy.lock:
            if self._lazy.index is not None:
                return

            reader = self._create_reader()
            try:
                self._lazy.index = FrameIndexPyAV(self.path, reader=reader)
            finally:
                reader.close()

            # Some codecs/containers mark packets as keyframes that are not
            # truly independently decodable (screen codecs, open-GOP MPEG,
            # packed-B MPEG-4), so seeking would silently return wrong
            # pixels. Verify that seeking reproduces sequential decode and
            # fall back to sequential-only access when it does not.
            if (
                self._lazy.index.frame_count > 1
                and self._codec_name in _SEEK_UNRELIABLE_CODECS
                and not self._seek_reproduces_sequential()
            ):
                self._lazy.seek_disabled = True
                # The packet-based index may also count packets the decoder
                # never turns into frames (the same brokenness that defeats
                # seeking), so rebuild it from what the decoder produces.
                self._rebuild_index_from_decode()

            # A PTS regression observed during earlier (index-less) streaming
            # means sorted-PTS location is unsound: use decoder output instead
            if self._lazy.emission_disorder:
                self._lazy.seek_disabled = True
                self._rebuild_index_from_decode()

            # Duplicate packet timestamps collapse in the sorted-unique index,
            # so the packet count is provably not the frame count; and an
            # earlier full streaming pass that saw a different total is ground
            # truth. Either way the packet index is wrong: rebuild it from
            # decoder output (only such broken files pay for the decode).
            elif self._lazy.index.had_duplicate_pts or (
                self._lazy.observed_seq_count is not None
                and self._lazy.observed_seq_count > 0
                and self._lazy.observed_seq_count != self._lazy.index.frame_count
            ):
                self._lazy.seek_disabled = True
                self._rebuild_index_from_decode()

            # In CFR mode, all behavior (count, indexing, iteration, seeking)
            # derives from this single output-index -> source-index map.
            if self.constant_framerate:
                self._lazy.cfr_source_map = self._build_cfr_source_map()

    def _degrade_to_sequential(self) -> None:
        """Seeking or PTS-based frame location proved unsound for this video.

        Disables seeking, rebuilds the index from actual decoder output
        (count-based access then matches iteration, the ground truth), and
        refreshes the CFR map. The verdict is shared by all views; healthy
        videos never reach this path, so they pay nothing.
        """
        self._lazy.seek_disabled = True
        if self._lazy.index is not None:
            self._rebuild_index_from_decode()
            if self.constant_framerate:
                self._lazy.cfr_source_map = self._build_cfr_source_map()

    def _flag_emission_disorder(self) -> None:
        self._lazy.emission_disorder = True
        self._degrade_to_sequential()

    @overload
    def __getitem__(self, item: int) -> NDArray: ...

    @overload
    def __getitem__(self, item: slice) -> VideoFrames: ...

    def __getitem__(self, item: int | slice) -> NDArray | VideoFrames:
        """Access a single frame by index or create a sliced lazy view.

        Args:
            item: Frame index (negative indices count from the end) or slice.

        Returns:
            The decoded frame as a numpy array for an integer index, or a
            new lazy VideoFrames view for a slice (no decoding happens).
        """
        if isinstance(item, int):
            # Handle negative indices
            length = len(self)
            if item < 0:
                item = length + item
            if item < 0 or item >= length:
                total = self._n_frames_total()
                if len(self._resolved_range()) != total or self.repeat_count != 1:
                    detail = f'view with {length} frames (source video has {total})'
                else:
                    detail = f'video with {length} frames'
                raise IndexError(f'Frame index {item} out of range for {detail}')

            # The bounds check above uses the repeat-inclusive length, so after
            # dividing out the repeat factor the index is always within range.
            abs_idx = self._resolved_range()[item // self.repeat_count]
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

            if item.step == 0:
                raise ValueError('slice step cannot be zero')

            result = self._clone()
            result._selection = self._selection.sliced(item)
            return result
        else:
            raise TypeError('VideoFrames indices must be integers or slices.')

    def __len__(self) -> int:
        """Exact number of frames in this view.

        Builds the frame index on first use, which scans the file's packets.
        """
        return len(self._resolved_range()) * self.repeat_count

    def __repr__(self) -> str:
        h, w = self.imshape
        label = self.path if isinstance(self.path, (str, Path)) else '<file-like>'
        # Never trigger the index scan just for a repr
        length = f'{len(self)} frames' if self._selection.is_resolved else 'lazy'
        return f"VideoFrames('{label}', {w}x{h}, {self.fps:.4g} fps, {length})"

    @property
    def imshape(self) -> tuple[int, int]:
        """Frame dimensions as (height, width) in pixels."""
        return self.resized_imshape if self.resized_imshape is not None else self.original_imshape

    @property
    def fps(self) -> float:
        """Effective frame rate, accounting for slicing and frame repetition.

        Uses the selection's effective stride, which is known even before the
        frame count is — reading fps never triggers the index scan.
        """
        return self.target_fps / abs(self._selection.step_product) * self.repeat_count

    def resized(self, shape: tuple[int, int]) -> 'VideoFrames':
        """Return a new VideoFrames that decodes frames at the given resolution.

        Args:
            shape: Target size as (height, width), following numpy/image convention.
                Note: this is the opposite order of ``video_extents()``, which
                returns (width, height). The frame is stretched to exactly this
                size; aspect ratio is not preserved.
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
        """Return a new VideoFrames that yields each selected frame ``n`` times.

        The effective ``fps`` scales by ``n`` accordingly. Apply slicing
        before this, not after: slicing a repeated view raises
        NotImplementedError.

        Args:
            n: Repeat count, at least 1.
        """
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
        result._selection = self._selection
        result.original_fps = self.original_fps
        result.repeat_count = self.repeat_count
        result.dtype = self.dtype
        result.gpu = self.gpu
        result.gray = self.gray
        result.constant_framerate = self.constant_framerate
        result.target_fps = self.target_fps
        # Index state (index, CFR map, seek verdicts) is shared: whichever
        # view builds it, all views observe the same result
        result._lazy = self._lazy
        result._is_fileobj = self._is_fileobj
        result._seekable = self._seekable
        result._codec_name = self._codec_name
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
        seekable = self._seekable if self._seekable is not None else self._lazy.seekable_probed
        if seekable is None:
            # First reader for this video: let it auto-probe (seek + decode
            # one frame) and cache the verdict so later readers skip it.
            # The probe pollutes the decoder state — codecs with delayed
            # frames can emit the probe's frame after a later seek(0) — so
            # reopen the container before handing the reader out.
            seekable = reader.seekable
            self._lazy.seekable_probed = seekable
            reader._reopen()
        reader._seekable = seekable and not self._lazy.seek_disabled
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

        # Large step (either direction): more efficient to seek to each frame
        # individually. Threshold is lower with PyAV since seeking is fast
        # (~10ms vs ~100ms). range() iterates backward natively for step < 0.
        if abs(slice_step) > 30:
            return self._iter_with_individual_seeks(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )

        # Small negative step: chunked reverse (buffer forward windows,
        # yield them reversed)
        if slice_step < 0:
            return self._iter_reversed(reader, frame_range, internal_dtype)

        # Use index-based seeking if we have an offset
        if slice_start > 0:
            return self._iter_with_seek(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )

        return self._iter_sequential(reader, slice_stop, slice_step, internal_dtype)

    def _iter_reversed(
        self,
        reader: PyAVReader,
        frame_range: range,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        """Iterate a negative-step range via backward chunks decoded forward.

        Walks the selected frames from the end in chunks; each chunk is a
        forward sub-range decoded through the normal forward machinery (so
        seeking, CFR mapping and all fallbacks behave identically), buffered,
        and yielded in reverse. Chunk starts prefer frames that are their own
        safe seek point (keyframes), so each chunk's seek is cheap; the chunk
        length is bounded by a memory budget since a chunk is held in memory.
        """
        fwd = frame_range[::-1]
        min_chunk, max_chunk, fallback_chunk = self._reverse_chunk_bounds(internal_dtype)

        pos = len(fwd)
        while pos > 0:
            lo = self._pick_reverse_chunk_start(fwd, pos, min_chunk, max_chunk, fallback_chunk)
            buf = list(self._iter_decoded(reader, fwd[lo:pos], internal_dtype))
            yield from reversed(buf)
            pos = lo

    def _reverse_chunk_bounds(self, internal_dtype: DTypeLike) -> tuple[int, int, int]:
        """(min, max, fallback) chunk lengths for reverse iteration.

        Derived from a byte budget so 4K chunks stay reasonable; clamped to
        the empirically tuned 32-64 frame window from the original
        implementation where memory allows.
        """
        h, w = self.imshape
        itemsize = 1 if internal_dtype == np.uint8 else 2
        frame_bytes = max(h * w * 3 * itemsize, 1)
        max_chunk = max(4, min(64, _REVERSE_CHUNK_BYTES // frame_bytes))
        min_chunk = max(1, max_chunk // 2)
        return min_chunk, max_chunk, (min_chunk + max_chunk) // 2

    def _pick_reverse_chunk_start(
        self, fwd: range, hi_pos: int, min_chunk: int, max_chunk: int, fallback_chunk: int
    ) -> int:
        """Position in ``fwd`` where the next (backward) chunk should start.

        Searches the window [hi_pos - max_chunk, hi_pos - min_chunk] backward
        for a selected frame that is its own safe seek point; falls back to a
        fixed chunk length when none is found.
        """
        lo_limit = max(hi_pos - max_chunk, 0)
        for p in range(max(hi_pos - min_chunk, 0), lo_limit - 1, -1):
            if p == 0 or self._is_safe_seek_frame(fwd[p]):
                return p
        return max(hi_pos - fallback_chunk, 0)

    def _is_safe_seek_frame(self, abs_idx: int) -> bool:
        """Whether this output frame's source is its own safe seek point."""
        src = self._abs_to_source(abs_idx)
        return self._index.frame_pts[src] == self._index.safe_seek_pts[src]

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
        watch_disorder = not self._lazy.seek_disabled
        try:
            yield from self._iter_with_seek_attempt(
                reader,
                slice_start,
                slice_stop,
                slice_step,
                internal_dtype,
                target_pts_frac,
                safe_pts_frac,
                watch_disorder,
            )
            return
        except _SeekPathUnsound:
            self._flag_emission_disorder()
        except _RetrySequential:
            self._degrade_to_sequential()
        # Retry against the rebuilt, decoder-ordered index (safe seeks are
        # now all zero and matching is count-based, mirroring iteration).
        # Only reachable before the first yield, so no frames are duplicated.
        reader._reopen()
        yield from self._iter_with_seek(
            reader, slice_start, slice_stop, slice_step, internal_dtype
        )

    def _iter_with_seek_attempt(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
        target_pts_frac,
        safe_pts_frac,
        watch_disorder: bool,
    ) -> Generator[NDArray, None, None]:
        max_frames = slice_stop - slice_start
        reader.seek_to_time(safe_pts_frac)

        # Build filter graph
        target_format = self._pix_target(internal_dtype)
        converter = reader.frame_converter(self.resized_imshape, target_format)

        # Skip frames until we reach the target PTS (mimics FFmpeg's -ss behavior)
        target_pts_float = float(target_pts_frac)
        time_base = reader.time_base
        reached_target = False
        # See _decode_at_source_once: accepting the target earlier than this
        # many emissions means out-of-order decoder output
        min_skips = slice_start - bisect_left(self._index.frame_pts, safe_pts_frac)

        frame_count = 0
        skip_count = 0
        frames = reader.decode_raw()
        while True:
            # The skip phase (nothing yielded yet) is safely retryable: a PTS
            # regression or a decode failure after a real seek both trigger a
            # sequential retry in the caller.
            try:
                frame = next(frames)
            except StopIteration:
                break
            except VideoDecodeError:
                if (
                    not reached_target
                    and not self._lazy.seek_disabled
                    and float(safe_pts_frac) > 0
                ):
                    raise _RetrySequential() from None
                raise
            # Check if we've reached the target frame. Match by PTS if available,
            # otherwise by decoded-frame count (timestampless streams, e.g. raw
            # H.264 elementary streams, where the index has synthetic PTS and
            # decoding starts from frame 0).
            if not reached_target:
                if watch_disorder and reader.pts_regression_seen:
                    raise _SeekPathUnsound()
                usable_pts = frame.pts is not None and not self._pts_unreliable
                frame_pts = Fraction(frame.pts) * time_base if usable_pts else None
                if not (
                    (frame_pts is not None and float(frame_pts) >= target_pts_float - 1e-6)
                    or skip_count == slice_start
                ):
                    skip_count += 1
                    continue
                if watch_disorder and frame_pts is not None and skip_count < min_skips:
                    raise _SeekPathUnsound()
                reached_target = True

            # Process frame through filter graph
            filtered_frame = converter.convert(frame)

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
        try:
            yield from self._iter_with_seek_cfr_attempt(
                reader, slice_start, slice_stop, slice_step, internal_dtype
            )
            return
        except _SeekPathUnsound:
            self._flag_emission_disorder()
        except _RetrySequential:
            self._degrade_to_sequential()
        # Retry against the rebuilt, decoder-ordered index and refreshed CFR
        # map. Only reachable before the first yield: no frames are duplicated.
        reader._reopen()
        yield from self._iter_with_seek_cfr(
            reader, slice_start, slice_stop, slice_step, internal_dtype
        )

    def _iter_with_seek_cfr_attempt(
        self,
        reader: PyAVReader,
        slice_start: int,
        slice_stop: int,
        slice_step: int,
        internal_dtype: DTypeLike,
    ) -> Generator[NDArray, None, None]:
        source_map = self._cfr_source_map
        first_source = source_map[slice_start]
        safe_pts_frac = self._index.safe_seek_pts[first_source]
        target_pts_frac = self._index.frame_pts[first_source]
        watch_disorder = not self._lazy.seek_disabled
        # See _decode_at_source_once: accepting the target earlier than this
        # many emissions means out-of-order decoder output
        min_skips = first_source - bisect_left(self._index.frame_pts, safe_pts_frac)
        reader.seek_to_time(safe_pts_frac)

        target_format = self._pix_target(internal_dtype)
        converter = reader.frame_converter(self.resized_imshape, target_format)

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
        frames = reader.decode_raw()
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except VideoDecodeError:
                if (
                    not reached_target
                    and not self._lazy.seek_disabled
                    and float(safe_pts_frac) > 0
                ):
                    raise _RetrySequential() from None
                raise
            if not reached_target:
                if watch_disorder and reader.pts_regression_seen:
                    raise _SeekPathUnsound()
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
                if watch_disorder and frame_pts is not None and skip_count < min_skips:
                    raise _SeekPathUnsound()
                reached_target = True
                source_idx = first_source

            filtered_frame = converter.convert(frame)
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
                target_format=self._pix_target(dtype),
            )
            return

        # CFR mode: walk the source map from the beginning
        source_map = self._cfr_source_map
        target_format = self._pix_target(dtype)

        # Build filter graph for exact FFmpeg compatibility
        converter = reader.frame_converter(self.resized_imshape, target_format)

        source_idx = 0
        output_idx = 0
        prev_frame_arr = None

        for frame in reader.decode_raw():
            # Process through filter graph for exact color conversion
            filtered_frame = converter.convert(frame)
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

        if source_idx >= self._index.frame_count:
            # Reachable when a view resolved its length against the packet
            # index before a disorder trigger shrank it to decoder output
            raise IndexError(
                f'Frame {source_idx} is out of range: {self.path} decodes to '
                f'{self._index.frame_count} frames'
            )

        # Get safe seek point and target PTS
        safe_pts_frac = self._index.safe_seek_pts[source_idx]
        target_pts_frac = self._index.frame_pts[source_idx]

        # Use provided reader or create temporary one
        own_reader = reader is None
        if own_reader:
            reader = self._create_reader()

        try:
            try:
                return self._decode_at_source_once(
                    reader, source_idx, safe_pts_frac, target_pts_frac, internal_dtype
                )
            except _SeekPathUnsound:
                # PTS regression observed before delivery: the sorted-PTS
                # match would return the wrong frame. Degrade (shared verdict)
                # and retry against the rebuilt, decoder-ordered index.
                self._flag_emission_disorder()
            except VideoDecodeError:
                if self._lazy.seek_disabled or float(safe_pts_frac) <= 0:
                    raise
                # Decoding after a real seek failed (some containers cannot
                # resume mid-stream even though sequential decode works);
                # remember and retry from the start.
                self._degrade_to_sequential()
            reader._reopen()
            source_idx = min(source_idx, self._index.frame_count - 1)
            return self._decode_at_source_once(
                reader,
                source_idx,
                self._index.safe_seek_pts[source_idx],
                self._index.frame_pts[source_idx],
                internal_dtype,
            )
        finally:
            if own_reader:
                reader.close()

    def _decode_at_source_once(
        self,
        reader: PyAVReader,
        source_idx: int,
        safe_pts_frac: Fraction,
        target_pts_frac: Fraction,
        internal_dtype: DTypeLike,
    ) -> NDArray:
        """One seek-and-match attempt; raises _SeekPathUnsound on a PTS
        regression observed before the target frame was delivered."""
        # Seek to safe point (keyframe before target)
        reader.seek_to_time(safe_pts_frac)

        # Build filter graph for exact FFmpeg compatibility
        target_format = self._pix_target(internal_dtype)
        converter = reader.frame_converter(self.resized_imshape, target_format)

        # Decode frames until we reach the target PTS
        target_pts_float = float(target_pts_frac)
        time_base = reader.time_base
        watch_disorder = not self._lazy.seek_disabled
        # After a seek to the safe point, the target must arrive at exactly
        # source_idx - safe_pos emissions. Arriving EARLIER means the decoder
        # emitted a future frame ahead of order (leading extra frames only
        # ever inflate the count, so this cannot false-trigger on open GOPs).
        min_skips = source_idx - bisect_left(self._index.frame_pts, safe_pts_frac)

        frame_count = 0
        for frame in reader.decode_raw():
            if watch_disorder and reader.pts_regression_seen:
                raise _SeekPathUnsound()
            usable_pts = frame.pts is not None and not self._pts_unreliable
            frame_pts = Fraction(frame.pts) * time_base if usable_pts else None
            # Match by PTS if available, otherwise by frame count (for attached pictures etc.)
            if (
                frame_pts is not None and float(frame_pts) >= target_pts_float - 1e-6
            ) or frame_count == source_idx:
                if watch_disorder and frame_pts is not None and frame_count < min_skips:
                    raise _SeekPathUnsound()
                filtered_frame = converter.convert(frame)
                return filtered_frame.to_ndarray()
            frame_count += 1

        raise VideoDecodeError(
            self.path, source_idx, RuntimeError(f'Failed to decode frame {source_idx}')
        )

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
        probe = {i for i in (1, 2, j0, j0 + 1, j0 + 2) if 0 < i < n}
        # When nearly every packet claims to be an independently decodable
        # keyframe (screen/animation codecs with false keyframe flags), the
        # shallow positions above all decode from nearby "keyframes" and match
        # even though deep seeks drift. Sample two deeper positions, bounded so
        # the sequential reference decode stays cheap. Sparse-GOP files (real
        # MPEG structure) skip this, keeping their probe cost at ~one GOP.
        if 2 * len(set(self._index.safe_seek_pts)) > n:
            deep = min(n - 1, _PROBE_DEEP_MAX)
            probe |= {deep // 2, deep}
        probe = sorted(i for i in probe if 0 < i < n)
        if not probe:
            return True

        try:
            reader = self._create_reader()
            try:
                # Same starting procedure as sequential iteration
                reader.seek_to_time(Fraction(0))
                sequential = {}
                for i, arr in enumerate(
                    reader.decode_frames(
                        max_frames=probe[-1] + 1, target_format=self._pix_target(np.uint8)
                    )
                ):
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
            self._lazy.pts_unreliable = True

        if n == 0:
            # Nothing decodes: keep the packet-based index so len() stays > 0
            # and empty iteration raises loudly instead of looking like a
            # legitimately empty video
            return
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

    By default this returns a fast **estimate** (container duration x fps),
    which can be wrong for variable-frame-rate videos and arbitrarily wrong
    when the container's duration metadata is bogus. Pass ``exact=True`` (or
    use ``len(VideoFrames(path))``, which is always exact) when the count
    matters.

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
