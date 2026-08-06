"""Video writing using PyAV for encoding.

This module provides the VideoWriter class for writing video files using PyAV.
It supports CPU encoding (libx264) and GPU encoding (h264_nvenc).
"""

from __future__ import annotations

import itertools
import os
import queue
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Generic, TypeVar, Union

import av
import av.stream
import numpy as np
import simplepyutils as spu
from numpy.typing import NDArray

from ._h264_mux import _FORMAT_ALIASES
from ._pyav import FrameIndexPyAV, PyAVReader, VideoEncodeError
from ._temp_file import TempFile
from .encoder_config import EncoderConfig

PathLike = Union[str, Path]
VideoOutput = Union[str, Path, BinaryIO]
T = TypeVar('T')


class AbstractVideoWriter(ABC, Generic[T]):
    """Abstract base class for video writers.

    Defines the common interface for VideoWriter (threaded, CPU/GPU encoding)
    and GLVideoWriter (synchronous, NVENC from GL textures).

    Generic over the data type: NDArray for VideoWriter, moderngl.Texture for GLVideoWriter.
    """

    _accepts_new_frames: bool

    @abstractmethod
    def start_sequence(
        self,
        video_output: VideoOutput,
        fps: float | Fraction | None = None,
        audio_source_path: PathLike | None = None,
        gpu: bool | int = False,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
    ) -> SequenceContext:
        """Start a new video sequence.

        Returns a ``SequenceContext`` usable as a context manager: it ends the
        sequence on clean exit and aborts it if the body raised.

        Args:
            video_path: Output path (str/Path) or file-like object (BinaryIO).
            fps: Frame rate for the video.
            audio_source_path: Optional path to copy audio from. A source
                without any audio stream is fine: the output is then video-only.
            gpu: False for CPU encoding, True for GPU (NVENC) on default device,
                or an int to select a specific GPU device ordinal.
            format: Container format (e.g., 'mp4'). Required for file-like objects,
                inferred from path extension otherwise.
        """
        ...

    @abstractmethod
    def append_data(self, data: T) -> None:
        """Append a frame to the current sequence."""
        ...

    @abstractmethod
    def end_sequence(self) -> None:
        """End the current video sequence.

        Implementations may accept additional keyword-only options (e.g.
        ``VideoWriter``'s ``block=``); the abstract interface stays minimal
        because such options are meaningless for synchronous writers.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        ...

    @property
    @abstractmethod
    def accepts_new_frames(self) -> bool:
        """Return whether a sequence is currently active."""
        ...


class Message:
    """Base class for queue messages."""

    pass


class SequenceContext(AbstractContextManager['SequenceContext']):
    """Context for a video sequence being written.

    On clean exit the sequence is finalized; if the body raised, the sequence
    is aborted instead, so no partial file appears at the final path. The
    writer remains usable for a new ``start_sequence`` either way.
    """

    def __init__(self, multiwriter) -> None:
        self.multiwriter = multiwriter

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any, **kwargs: Any) -> None:
        if exc_type is None:
            self.multiwriter.end_sequence()
        else:
            self.multiwriter._abort()


class _WriterState(Enum):
    """Lifecycle of the writer/worker pair.

    FAILED is sticky: producer calls keep raising until start_sequence()
    deliberately restarts the worker (which resets all lifecycle state).
    There is no separate CLOSED state because a closed writer is reusable
    by design; IDLE covers both never-started and cleanly-closed.
    """

    IDLE = 'idle'
    RUNNING = 'running'
    FAILED = 'failed'


class VideoWriter(AbstractVideoWriter[NDArray], AbstractContextManager['VideoWriter']):
    """Threaded video writer with queue-based frame buffering using PyAV.

    Uses a background thread to write frames, allowing the main thread to continue
    processing while frames are being encoded. Supports multiple video sequences.
    This way, the main thread is not blocked by video encoding and can move on to the next video
    before the previous one has finished encoding, which is useful for processing many small
    videos.

    Example:
        >>> with VideoWriter('output.mp4', fps=30) as writer:
        ...     for frame in frames:
        ...         writer.append_data(frame)

    Args:
        video_path: Output path for the first video sequence (optional).
        fps: Frame rate for the first video sequence (required if video_path is provided).
        audio_source_path: Path to copy audio from, for the first video sequence.
            A source without any audio stream results in video-only output.
        queue_size: Max frames to buffer before blocking on `append_data`.
        gpu: False for CPU encoding (libx264/libx265), True for GPU (NVENC)
            on the default device, or an int GPU device ordinal. NVENC is
            8-bit only: uint16/float frames require gpu=False.
        encoder_config: :class:`~framepump.EncoderConfig` with crf, preset,
            bframes, gop and codec settings (defaults are used if omitted).
    """

    def __init__(
        self,
        video_path: PathLike | None = None,
        fps: float | None = None,
        audio_source_path: PathLike | None = None,
        queue_size: int = 32,
        gpu: bool | int = False,
        encoder_config: EncoderConfig | None = None,
    ) -> None:
        """Create a new VideoWriter.

        See class docstring for full parameter descriptions.
        """
        if queue_size < 1:
            raise ValueError(f'queue_size must be >= 1, got {queue_size}')
        self._queue: queue.Queue[Message] = queue.Queue(queue_size)
        self._feedback_queue: queue.Queue[FeedbackMessage] = queue.Queue()
        self._thread: threading.Thread | None = None
        # Kept even after close() nulls _thread, so a restart can never race
        # a previous worker that is still consuming the queue.
        self._last_thread: threading.Thread | None = None
        self._accepts_new_frames: bool = False
        self._state: _WriterState = _WriterState.IDLE
        self._state_lock = threading.Lock()
        self._worker_error: Exception | None = None
        self._failed_video_path: PathLike | None = None  # For error reporting
        self._error_reported: bool = False
        self._shutdown_event: threading.Event = threading.Event()  # For immediate shutdown
        self._default_fps = fps
        self._default_gpu = gpu
        self._default_encoder_config = encoder_config

        if video_path is not None:
            if fps is None:
                raise ValueError('fps must be provided if video_path is provided')
            self.start_sequence(
                video_path,
                fps,
                audio_source_path=audio_source_path,
                gpu=gpu,
                encoder_config=encoder_config,
            )

    @property
    def accepts_new_frames(self) -> bool:
        """Whether new frames are accepted for writing, i.e., a sequence has been started but
        not ended yet.
        """
        return self._accepts_new_frames

    def start_sequence(
        self,
        video_output: VideoOutput,
        fps: float | Fraction | None = None,
        audio_source_path: PathLike | None = None,
        audio_stream_index: int = 0,
        gpu: bool | int | None = None,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
    ) -> SequenceContext:
        """Start a new video sequence.

        Returns a ``SequenceContext`` usable as a context manager: it ends the
        sequence on clean exit and aborts it if the body raised.

        Args:
            video_output: Output path (str/Path) or file-like object (BinaryIO).
            fps: Frame rate for the video.
            audio_source_path: Optional path to copy audio from. A source
                without any audio stream is fine: the output is then video-only.
            audio_stream_index: Which audio stream to use (default 0).
            gpu: False for CPU encoding, True for GPU (NVENC) on default device,
                or an int to select a specific GPU device ordinal.
            encoder_config: Encoder configuration (crf, preset, bframes, gop, codec).
            format: Container format (e.g., 'mp4'). Required for file-like objects.
        """
        if self._state is _WriterState.FAILED and not self._error_reported:
            self._raise_worker_failure()
        self._ensure_worker_running()

        if fps is None:
            if self._default_fps is None:
                raise ValueError('fps must be provided if not set in constructor')
            fps = self._default_fps

        if gpu is None:
            gpu = self._default_gpu

        if encoder_config is None:
            encoder_config = self._default_encoder_config

        self._put_checked(
            StartSequence(
                video_output,
                fps,
                audio_source_path=audio_source_path,
                audio_stream_index=audio_stream_index,
                gpu=gpu,
                encoder_config=encoder_config,
                format=format,
            )
        )
        self._accepts_new_frames = True
        return SequenceContext(self)

    def _abort(self) -> None:
        """Discard the current sequence (no file at the final path).

        The writer stays usable for a new ``start_sequence``. Used by
        SequenceContext and __exit__ when the body raised.
        """
        if not self._accepts_new_frames:
            return
        self._consume_stale_feedback()
        msg = AbortSequence()
        self._put_checked(msg)
        self._accepts_new_frames = False
        self._wait_for_confirmation(msg)

    def append_data(self, data: NDArray) -> None:
        """Append a frame to the current video sequence.

        Args:
            data: Frame as numpy array (H, W, 3). Supported dtypes:
                - uint8: Standard 8-bit encoding
                - uint16: High precision 10-bit encoding
                - float16/float32/float64: Auto-converted to uint16 ([0,1] -> [0,65535])
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f'append_data expects a numpy ndarray, got {type(data).__name__}')
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before appending data')
        self._put_checked(AppendFrame(data))

    def end_sequence(self, block=True) -> None:
        """Request to end the current video sequence (once all pending frames have been processed).

        Args:
            block: If True, block until the current sequence has been saved to video.
        """
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before ending the sequence')

        self._consume_stale_feedback()
        msg = EndSequence()
        self._put_checked(msg)
        self._accepts_new_frames = False

        if not block:
            return
        self._wait_for_confirmation(msg)

    def _wait_for_confirmation(self, msg: Message) -> None:
        while True:
            try:
                feedback = self._feedback_queue.get(timeout=0.5)
            except queue.Empty:
                self._check_worker_failure()
                continue
            if isinstance(feedback, ExceptionRaised):
                self._note_worker_failure(feedback.error, feedback.video_path)
                self._raise_worker_failure()
            if isinstance(feedback, EndSequenceDone) and feedback.initial_msg is msg:
                # This is the confirmation we were waiting for
                return
            # Confirmations of earlier non-blocking end_sequence calls: drop

    def close(self) -> None:
        """Close the writer, waiting for pending frames to be written.

        If the worker failed and the error was not yet raised by another
        call, it is raised here; an error that was already reported is not
        raised a second time (so ``try/finally: close()`` never masks it).
        """
        try:
            thread = self._thread
            if thread is not None and thread.is_alive() and self._state is _WriterState.RUNNING:
                try:
                    self._put_checked(Quit())
                    # Timeout proportional to queue size (assume ~0.5s per frame for encoding)
                    timeout = max(10.0, self._queue.maxsize * 2.0)
                    thread.join(timeout=timeout)
                except BaseException:
                    # KeyboardInterrupt during the drain: make the worker stop
                    # so it cannot keep consuming the queue after _thread is
                    # nulled (and race a restarted worker).
                    self._shutdown_event.set()
                    raise
                if thread.is_alive():
                    warnings.warn(
                        f'VideoWriter did not finish within {timeout:.0f}s timeout, '
                        'forcing shutdown. Some frames may not have been written.',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Graceful quit didn't work, force shutdown
                    self._shutdown_event.set()
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        raise RuntimeError('VideoWriter thread did not exit in time')
            elif thread is not None:
                thread.join(timeout=3.0)
                if self._state is _WriterState.RUNNING:
                    # Worker died without going through its error handler
                    # (e.g. killed by a non-Exception BaseException)
                    self._note_worker_failure(None, None)
        finally:
            self._thread = None
            self._accepts_new_frames = False
            self._consume_stale_feedback()
        if self._state is _WriterState.FAILED:
            if not self._error_reported:
                self._raise_worker_failure()
        else:
            self._state = _WriterState.IDLE

    def shutdown(self) -> None:
        """Immediately stop the background thread without waiting for pending work.

        Warning: any frames still queued will be discarded and the current
        output file is aborted (deleted), not finalized. Use ``close()`` to
        wait for all pending frames to be written.
        """
        thread = self._thread
        if thread is not None:
            n_pending = self._queue.qsize()
            if n_pending > 0:
                warnings.warn(
                    f'shutdown() called with ~{n_pending} pending queue items. '
                    f'Queued frames will be discarded. Use close() to wait for '
                    f'all frames to be written.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._shutdown_event.set()
            thread.join(timeout=3.0)
            self._thread = None
        self._accepts_new_frames = False
        self._consume_stale_feedback()
        if self._state is not _WriterState.FAILED:
            self._state = _WriterState.IDLE

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any, **kwargs: Any) -> None:
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            # On Ctrl+C, don't wait for pending work
            self.shutdown()
        elif exc_type is not None:
            # An exception escaped the with-block: discard the in-flight
            # sequence instead of promoting a partial file to the final path
            # (matching GLVideoWriter and NvJpegVideoWriter). Secondary
            # errors must not mask the original exception.
            try:
                self._abort()
            except Exception:
                pass
            try:
                self.close()
            except Exception:
                pass
        else:
            self.close()

    def __del__(self) -> None:
        try:
            thread = getattr(self, '_thread', None)
            if thread is not None and thread.is_alive():
                warnings.warn(
                    'VideoWriter was garbage-collected without close(); pending frames '
                    'are discarded and the current output file is not finalized.',
                    ResourceWarning,
                    stacklevel=2,
                )
                self._shutdown_event.set()
        except Exception:
            # Interpreter shutdown can leave globals in a torn-down state
            pass

    def _put_checked(self, msg: Message) -> None:
        """Put a message on the work queue without ever blocking indefinitely.

        Bounded wait that re-checks worker health each round: if the worker
        failed (or died silently), this raises instead of deadlocking on a
        queue that nobody consumes anymore.
        """
        while True:
            self._check_worker_failure()
            try:
                self._queue.put(msg, timeout=0.2)
                return
            except queue.Full:
                continue

    def _check_worker_failure(self) -> None:
        if self._state is _WriterState.FAILED:
            self._raise_worker_failure()
        if self._state is _WriterState.RUNNING and (
            self._thread is None or not self._thread.is_alive()
        ):
            # Worker died without going through its error handler
            self._note_worker_failure(None, None)
            self._raise_worker_failure()

    def _raise_worker_failure(self) -> None:
        with self._state_lock:
            exc = self._worker_error
            self._error_reported = True
        if exc is None:
            raise RuntimeError('VideoWriter thread died unexpectedly')
        # Re-raise the worker's exception as-is: callers can catch the type
        # that append/open actually raised (e.g. ValueError for a bad frame),
        # and the traceback shows the original failure site in the worker.
        raise exc

    def _note_worker_failure(self, error: Exception | None, video_path) -> None:
        with self._state_lock:
            if self._worker_error is None:
                self._worker_error = error
                self._failed_video_path = video_path
            self._state = _WriterState.FAILED

    def _consume_stale_feedback(self) -> None:
        """Drop confirmations nobody waited for; record any worker failure."""
        while True:
            try:
                feedback = self._feedback_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(feedback, ExceptionRaised):
                self._note_worker_failure(feedback.error, feedback.video_path)

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _ensure_worker_running(self) -> None:
        """(Re)start the worker thread; owns resetting all lifecycle state."""
        if (
            self._state is _WriterState.RUNNING
            and self._thread is not None
            and self._thread.is_alive()
        ):
            return
        if self._last_thread is not None:
            self._last_thread.join(timeout=5.0)
            if self._last_thread.is_alive():
                raise RuntimeError('VideoWriter worker thread is stuck; cannot restart')
        self._shutdown_event.clear()
        self._drain_queue(self._queue)
        self._drain_queue(self._feedback_queue)
        with self._state_lock:
            self._worker_error = None
            self._failed_video_path = None
            self._error_reported = False
            self._state = _WriterState.RUNNING
        self._thread = threading.Thread(target=self._main_video_writer, daemon=True)
        self._last_thread = self._thread
        self._thread.start()

    def _main_video_writer(self) -> None:
        # Main loop of the background thread. Exactly two kinds of exit:
        # clean (Quit / graceful close -> finalize the output file) and
        # error/shutdown (-> abort: a partial file is never promoted to the
        # final path).
        writer: SequenceWriter | None = None
        try:
            while not self._shutdown_event.is_set():
                try:
                    msg = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(msg, AppendFrame):
                    if writer is None:
                        raise ValueError('No active sequence to append frame to')
                    writer.write_frame(msg.frame)
                elif isinstance(msg, StartSequence):
                    if writer is not None:
                        # We allow directly sending StartSequence without EndSequence
                        # so we should close the previous writer first
                        writer.close()
                        writer = None
                    if isinstance(msg.video_output, (str, Path)):
                        spu.ensure_parent_dir_exists(msg.video_output)
                    writer = self._sequence_writer_cls(
                        msg.video_output,
                        fps=msg.fps,
                        audio_source_path=msg.audio_source_path,
                        audio_stream_index=msg.audio_stream_index,
                        gpu=msg.gpu,
                        encoder_config=msg.encoder_config,
                        format=msg.format,
                    )
                elif isinstance(msg, AbortSequence):
                    if writer is not None:
                        writer._abort()
                        writer = None
                    self._feedback_queue.put(EndSequenceDone(msg))
                elif isinstance(msg, EndSequence):
                    if writer is not None:
                        writer.close()
                        writer = None
                    self._feedback_queue.put(EndSequenceDone(msg))
                elif isinstance(msg, Quit):
                    if writer is not None:
                        writer.close()
                        writer = None
                    return
                else:
                    raise ValueError(f'Unexpected message type: {type(msg)}')

            # Shutdown requested: abort the current sequence without finalizing
            if writer is not None:
                writer._abort()
                writer = None
        except Exception as e:
            failed_path = writer.output_path if writer is not None else None
            if writer is not None:
                try:
                    writer._abort()
                except Exception:
                    pass
            self._note_worker_failure(e, failed_path)
            self._feedback_queue.put(ExceptionRaised(e, failed_path))


class SequenceWriter(AbstractContextManager['SequenceWriter']):
    """Writes a single video sequence with optional audio interleaving.

    Usage:
        with SequenceWriter(path, fps=30) as writer:
            for frame in frames:
                writer.write_frame(frame)

        # Or write to a file-like object:
        buffer = io.BytesIO()
        with SequenceWriter(buffer, fps=30, format='mp4') as writer:
            for frame in frames:
                writer.write_frame(frame)
        video_bytes = buffer.getvalue()
    """

    def __init__(
        self,
        video_output: VideoOutput,
        fps: float | Fraction,
        audio_source_path: PathLike | None = None,
        audio_stream_index: int = 0,
        gpu: bool | int = False,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
    ) -> None:
        self._fps_frac = (
            fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        )
        self._encoder_config = encoder_config if encoder_config is not None else EncoderConfig()
        self._audio_source_path = audio_source_path
        self._audio_stream_index = audio_stream_index
        self._gpu = gpu

        # Determine output mode: file-like or path
        if isinstance(video_output, (str, Path)):
            self._temp_file = TempFile(video_output)
            self._file_output = None
            fmt = format or Path(video_output).suffix.lstrip('.')
            self._format = _FORMAT_ALIASES.get(fmt, fmt)
        else:
            # File-like object
            self._temp_file = None
            self._file_output = video_output
            if format is None:
                raise ValueError('format is required when writing to a file-like object')
            self._format = format

        # State will be initialized on first frame
        self._output_container: av.container.OutputContainer | None = None
        self._audio_input_container: av.container.InputContainer | None = None
        self._video_stream: av.stream.Stream | None = None
        self._audio_stream: av.stream.Stream | None = None
        self._audio_time_base: Fraction = Fraction(1)
        self._audio_pkts: Iterator[av.Packet] = iter([])
        self._input_format: str = 'rgb24'
        self._pts: int = 0
        self._closed: bool = False
        self._frame_dtype: Any = None
        self._frame_shape: tuple[int, ...] = ()

    @property
    def output_path(self) -> Path | None:
        """Output path if writing to a file, None if writing to file-like object."""
        return self._temp_file.final_path if self._temp_file is not None else None

    def write_frame(self, frame: NDArray) -> None:
        """Write a frame to the video."""
        if self._closed:
            raise RuntimeError('Writer is closed, cannot write more frames.')

        frame = self._prepare_frame(frame)

        if self._output_container is None:
            self._open(frame)
        else:
            if frame.dtype != self._frame_dtype:
                raise ValueError(
                    f'Frame dtype {frame.dtype} does not match initial frame dtype '
                    f'{self._frame_dtype}'
                )
            if frame.shape != self._frame_shape:
                raise ValueError(
                    f'Frame shape {frame.shape} does not match initial frame shape '
                    f'{self._frame_shape}'
                )

        video_time = self._pts / self._fps_frac

        # Interleave: write audio packets up to current video time
        for audio_pkt in self._audio_pkts:
            if audio_pkt.dts * self._audio_time_base > video_time:
                # Put back the packet for next round
                self._audio_pkts = itertools.chain([audio_pkt], self._audio_pkts)
                break
            audio_pkt.stream = self._audio_stream
            self._output_container.mux(audio_pkt)

        # Encode and write video frame
        video_frame = av.VideoFrame.from_ndarray(frame, format=self._input_format)
        video_frame.pts = self._pts
        try:
            for packet in self._video_stream.encode(video_frame):
                self._output_container.mux(packet)
        except av.FFmpegError as e:
            # NVENC minimum size is ~145x49 (varies by driver/GPU)
            # If encoding fails with "Invalid argument" and we're using GPU, check frame size
            if self._gpu and (frame.shape[0] < 50 or frame.shape[1] < 150):
                raise VideoEncodeError(
                    self.output_path or '<file-like>',
                    self._pts,
                    e,
                    resolution=(frame.shape[1], frame.shape[0]),
                    codec='h264_nvenc',
                ) from e
            raise VideoEncodeError(self.output_path or '<file-like>', self._pts, e) from e

        self._pts += 1

    def _prepare_frame(self, frame: NDArray) -> NDArray:
        """Normalize an incoming frame before validation and encoding."""
        # Accept grayscale input by replicating to 3 channels (symmetric
        # with VideoFrames(gray=True) on the reading side)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        # Convert float to uint16 for high precision encoding
        if np.issubdtype(frame.dtype, np.floating):
            return _float_to_uint16(frame)
        return frame

    def _validate_first_frame(self, first_frame: NDArray) -> None:
        if first_frame.dtype not in (np.uint8, np.uint16):
            raise ValueError(f'Unsupported frame dtype: {first_frame.dtype}')

        if self._gpu and first_frame.dtype == np.uint16:
            # h264_nvenc only takes 8-bit input; 10-bit NVENC output would
            # need hevc_nvenc with p010le, which is not implemented.
            raise ValueError(
                'GPU encoding (h264_nvenc) supports only 8-bit uint8 frames; '
                'uint16/float high-bit-depth frames require CPU encoding (gpu=False).'
            )

        height, width = first_frame.shape[:2]
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f'Frame dimensions must be even for H.264 encoding (yuv420p), '
                f'got {width}x{height}'
            )

        # NVENC has a max resolution of 4096x4096 for H.264
        if self._gpu and (width > 4096 or height > 4096):
            warnings.warn(
                f'Frame size {width}x{height} exceeds NVENC limit (4096x4096), '
                f'falling back to CPU encoding'
            )
            self._gpu = False

    def _setup_video_stream(self, first_frame: NDArray) -> None:
        height, width = first_frame.shape[:2]
        codec_name = self._encoder_config.get_codec_name(self._gpu)
        try:
            self._video_stream = self._output_container.add_stream(codec_name, rate=self._fps_frac)
        except (av.FFmpegError, ValueError) as e:
            raise VideoEncodeError(
                self.output_path or '<file-like>', 0, e, codec=codec_name
            ) from e
        self._video_stream.width = width
        self._video_stream.height = height

        if first_frame.dtype == np.uint8:
            self._video_stream.pix_fmt = 'yuv420p'
            self._input_format = 'rgb24'
        else:  # uint16
            self._video_stream.pix_fmt = 'yuv420p10le'
            self._input_format = 'rgb48le'

        self._video_stream.options = self._encoder_config.build_options(self._gpu)

    def _open(self, first_frame: NDArray) -> None:
        """Open containers and set up streams based on first frame."""
        self._validate_first_frame(first_frame)

        self._frame_dtype = first_frame.dtype
        self._frame_shape = first_frame.shape

        if self._temp_file is not None:
            self._output_container = av.open(
                os.fspath(self._temp_file.temp_path), 'w', format=self._format
            )
        else:
            self._output_container = av.open(self._file_output, 'w', format=self._format)

        self._setup_video_stream(first_frame)

        # Set up audio if provided
        if self._audio_source_path is not None:
            self._audio_input_container = av.open(str(self._audio_source_path))
            if not self._audio_input_container.streams.audio:
                # A source without audio simply means there is nothing to carry
                # over; write a video-only file.
                self._audio_input_container.close()
                self._audio_input_container = None
            else:
                if self._audio_stream_index >= len(self._audio_input_container.streams.audio):
                    raise ValueError(
                        f'Audio stream index {self._audio_stream_index} out of range, '
                        f'file has {len(self._audio_input_container.streams.audio)} audio streams'
                    )
                src_audio = self._audio_input_container.streams.audio[self._audio_stream_index]
                self._audio_stream = self._output_container.add_stream_from_template(src_audio)
                self._audio_time_base = src_audio.time_base
                self._audio_pkts = (
                    pkt
                    for pkt in self._audio_input_container.demux(src_audio)
                    if pkt.dts is not None
                )

    def close(self) -> None:
        """Flush encoder and close containers, then rename temp to final.

        If flushing or muxing fails, the temp file is deleted (a partial
        file is never promoted to the final path) and the error propagates.
        """
        if self._closed:
            return
        self._closed = True

        if self._output_container is None:
            # Never opened (no frames written)
            return

        try:
            try:
                # Flush video encoder
                for packet in self._video_stream.encode():
                    self._output_container.mux(packet)

                # Flush remaining audio packets
                for audio_pkt in self._audio_pkts:
                    audio_pkt.stream = self._audio_stream
                    self._output_container.mux(audio_pkt)
            finally:
                try:
                    self._output_container.close()
                finally:
                    if self._audio_input_container is not None:
                        self._audio_input_container.close()
        except BaseException:
            if self._temp_file is not None:
                self._temp_file.cleanup()
            raise

        if self._temp_file is not None:
            self._temp_file.finalize()

    def _abort(self) -> None:
        """Abort write - close resources without flushing, delete temp file.

        Closing the output container writes a trailer and can itself raise
        (e.g. the very disk-full error that triggered the abort); the temp
        file must be deleted regardless.
        """
        if self._closed:
            return
        self._closed = True

        try:
            try:
                if self._output_container is not None:
                    self._output_container.close()
            finally:
                if self._audio_input_container is not None:
                    self._audio_input_container.close()
        finally:
            if self._temp_file is not None:
                self._temp_file.cleanup()

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()


# SequenceWriter is defined after VideoWriter, so the binding lives here
VideoWriter._sequence_writer_cls = SequenceWriter


class DepthVideoWriter(VideoWriter):
    """Threaded writer for 16-bit grayscale depth videos (lossless FFV1).

    Stores depth maps as FFV1-encoded ``gray16le`` in an MKV container:
    truly lossless for 16-bit data, roughly half the size of a PNG sequence
    thanks to temporal compression. Frames are ``(height, width)`` uint16
    arrays (e.g. depth in millimeters).

    Read the result back losslessly with
    ``VideoFrames(path, dtype=np.uint16, gray=True)``.

    Example:
        >>> with DepthVideoWriter('depth.mkv', fps=5) as writer:
        ...     for depth in depth_frames:  # (H, W) uint16
        ...         writer.append_data(depth)
    """

    _sequence_writer_cls = None  # bound below

    def __init__(self, *args: Any, gpu: bool | int = False, **kwargs: Any) -> None:
        if gpu:
            raise ValueError(
                'DepthVideoWriter is CPU-only (FFV1 has no hardware encoder); '
                'remove the gpu argument'
            )
        super().__init__(*args, gpu=False, **kwargs)

    def start_sequence(self, *args: Any, gpu: bool | int | None = None, **kwargs: Any):
        if gpu:
            raise ValueError(
                'DepthVideoWriter is CPU-only (FFV1 has no hardware encoder); '
                'remove the gpu argument'
            )
        return super().start_sequence(*args, gpu=False, **kwargs)


class _DepthSequenceWriter(SequenceWriter):
    """SequenceWriter variant encoding (H, W) uint16 frames as FFV1 gray16le."""

    _SUPPORTED_FORMATS = ('matroska', 'avi', 'nut')

    def _prepare_frame(self, frame: NDArray) -> NDArray:
        # Float depth is ambiguous (meters? normalized?); require explicit
        # integer values instead of guessing a scale
        if np.issubdtype(frame.dtype, np.floating):
            raise ValueError(
                'DepthVideoWriter requires uint16 frames; convert float depth '
                'to integer units (e.g. millimeters) explicitly'
            )
        return frame

    def _validate_first_frame(self, first_frame: NDArray) -> None:
        if first_frame.dtype != np.uint16:
            raise ValueError(f'DepthVideoWriter requires uint16 frames, got {first_frame.dtype}')
        if first_frame.ndim != 2:
            raise ValueError(
                f'DepthVideoWriter requires (height, width) frames, '
                f'got shape {first_frame.shape}'
            )
        if self._format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f'FFV1 depth video needs an MKV (or AVI/NUT) container, '
                f"got format {self._format!r} — use a '.mkv' output path"
            )

    def _setup_video_stream(self, first_frame: NDArray) -> None:
        height, width = first_frame.shape
        self._video_stream = self._output_container.add_stream('ffv1', rate=self._fps_frac)
        self._video_stream.width = width
        self._video_stream.height = height
        self._video_stream.pix_fmt = 'gray16le'
        self._input_format = 'gray16le'


DepthVideoWriter._sequence_writer_cls = _DepthSequenceWriter


def video_audio_mux(
    vidpath_audiosource: PathLike,
    vidpath_imagesource: PathLike,
    out_video_path: PathLike,
) -> None:
    """Mux video from one file with audio from another using PyAV.

    Args:
        vidpath_audiosource: Path to file containing audio. A source without
            any audio stream is fine: the output is then video-only.
        vidpath_imagesource: Path to file containing video.
        out_video_path: Output path.
    """
    spu.ensure_parent_dir_exists(out_video_path)

    # Write through a temp file so an error leaves nothing at the final path
    temp_file = TempFile(out_video_path)
    out_format = Path(out_video_path).suffix.lstrip('.')
    out_format = _FORMAT_ALIASES.get(out_format, out_format)
    try:
        _video_audio_mux_to_path(
            vidpath_audiosource, vidpath_imagesource, temp_file.temp_path, out_format
        )
    except BaseException:
        temp_file.cleanup()
        raise
    temp_file.finalize()


def _video_audio_mux_to_path(
    vidpath_audiosource: PathLike,
    vidpath_imagesource: PathLike,
    out_video_path: PathLike,
    out_format: str,
) -> None:
    with (
        av.open(str(vidpath_imagesource)) as video_src,
        av.open(str(vidpath_audiosource)) as audio_src,
        av.open(str(out_video_path), 'w', format=out_format) as output,
    ):
        src_video = video_src.streams.video[0]
        out_video = output.add_stream_from_template(src_video)
        if audio_src.streams.audio:
            src_audio = audio_src.streams.audio[0]
            out_audio = output.add_stream_from_template(src_audio)
            audio_pkts = (p for p in audio_src.demux(src_audio) if p.dts is not None)
        else:
            # A source without audio simply means there is nothing to mux in;
            # produce a video-only file.
            src_audio = None
            out_audio = None
            audio_pkts = iter(())

        video_pkts = (p for p in video_src.demux(src_video) if p.dts is not None)

        for video_pkt in video_pkts:
            if out_audio is not None:
                video_time = video_pkt.dts * src_video.time_base

                # Write audio packets up to current video time
                for audio_pkt in audio_pkts:
                    if audio_pkt.dts * src_audio.time_base > video_time:
                        # Put back the packet for next round
                        audio_pkts = itertools.chain([audio_pkt], audio_pkts)
                        break
                    audio_pkt.stream = out_audio
                    output.mux(audio_pkt)

            # Write video packet
            video_pkt.stream = out_video
            output.mux(video_pkt)

        # Flush remaining audio
        for audio_pkt in audio_pkts:
            audio_pkt.stream = out_audio
            output.mux(audio_pkt)


def trim_video(
    input_path: PathLike,
    output_path: PathLike,
    start_time: float | str,
    end_time: float | str,
    gpu: bool | int | None = None,
) -> None:
    """Trim video to a time range using PyAV.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        start_time: Start time as seconds (float) or timestamp string
            ('HH:MM:SS', 'MM:SS', or 'SS', with optional fractional seconds).
        end_time: End time as seconds (float) or timestamp string.
        gpu: False for CPU encoding, True for GPU (NVENC) on default device,
            or an int to select a specific GPU device ordinal. If None, auto-detect.
    """
    start_time = _parse_time(start_time)
    end_time = _parse_time(end_time)

    spu.ensure_parent_dir_exists(output_path)

    # Build frame index for accurate seeking
    with PyAVReader(input_path) as reader:
        index = FrameIndexPyAV(input_path, reader)
        width, height = reader.resolution

    # Output dimensions after the even-dimension rounding applied below
    out_w = width + (width % 2)
    out_h = height + (height % 2)
    if gpu is None:
        # NVENC cannot encode below its hardware minimum; auto-detect must not
        # pick an encoder that will reject the video at the first frame.
        gpu = _nvenc_available() and out_w >= _NVENC_MIN_W and out_h >= _NVENC_MIN_H
    elif gpu and (out_w < _NVENC_MIN_W or out_h < _NVENC_MIN_H):
        raise ValueError(
            f'NVENC cannot encode {width}x{height} video: the hardware minimum '
            f'is {_NVENC_MIN_W}x{_NVENC_MIN_H}. Use gpu=False.'
        )

    start_frame_idx = _find_frame_at_time(index, start_time)
    end_frame_idx = _find_frame_at_time(index, end_time)
    if start_frame_idx >= index.frame_count:
        duration = float(index.frame_pts[-1]) if index.frame_count else 0.0
        raise ValueError(
            f'start_time {start_time} is past the last frame of {input_path} '
            f'(last frame at {duration:.3f} s)'
        )
    if end_frame_idx <= start_frame_idx:
        raise ValueError(
            f'Requested range [{start_time}, {end_time}) contains no frames: '
            f'end_time must be greater than start_time by at least one frame interval'
        )
    target_pts = index.frame_pts[start_frame_idx]
    # None means "no end bound": the requested end is at/past the last frame
    end_pts = index.frame_pts[end_frame_idx] if end_frame_idx < index.frame_count else None
    safe_seek_pts = index.safe_seek_pts[start_frame_idx]

    # Write through a temp file so an error leaves nothing at the final path
    temp_file = TempFile(output_path)
    out_format = Path(output_path).suffix.lstrip('.')
    out_format = _FORMAT_ALIASES.get(out_format, out_format)
    try:
        _trim_video_to_path(
            input_path, temp_file.temp_path, out_format, gpu, target_pts, end_pts, safe_seek_pts
        )
    except BaseException:
        temp_file.cleanup()
        raise
    temp_file.finalize()


def _trim_video_to_path(
    input_path: PathLike,
    output_path: PathLike,
    out_format: str,
    gpu: bool | int,
    target_pts: Fraction,
    end_pts: Fraction | None,
    safe_seek_pts: Fraction,
) -> None:
    with (
        av.open(str(input_path)) as input_container,
        av.open(str(output_path), 'w', format=out_format) as output_container,
    ):
        input_video = input_container.streams.video[0]
        input_audio = input_container.streams.audio[0] if input_container.streams.audio else None

        # Set up output video stream
        codec_name = 'h264_nvenc' if gpu else 'libx264'
        rate = input_video.guessed_rate or input_video.average_rate
        fps = Fraction(rate.numerator, rate.denominator) if rate else Fraction(30)
        video_stream = output_container.add_stream(codec_name, rate=fps)

        # H.264 requires even dimensions (yuv420p chroma subsampling)
        out_w = input_video.width + (input_video.width % 2)
        out_h = input_video.height + (input_video.height % 2)
        video_stream.width = out_w
        video_stream.height = out_h
        video_stream.pix_fmt = 'yuv420p'
        options = {'rc': 'vbr', 'cq': '20'} if gpu else {'crf': '20'}
        if type(gpu) is int:  # noqa: E721 (bool is excluded on purpose)
            options['gpu'] = str(gpu)
        video_stream.options = options

        # Set up output audio stream if present
        audio_stream = None
        if input_audio:
            try:
                audio_stream = output_container.add_stream_from_template(input_audio)
            except (av.error.FFmpegError, ValueError):
                warnings.warn(
                    'Audio codec is not compatible with the output format, '
                    'skipping audio stream.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                input_audio = None

        # Filter graph: reset timestamps, scale to even dims, convert to yuv420p
        graph = av.filter.Graph()
        buffer_in = graph.add_buffer(template=input_video)
        setpts = graph.add('setpts', 'PTS-STARTPTS')
        scale = graph.add('scale', f'{out_w}:{out_h}')
        fmt = graph.add('format', 'pix_fmts=yuv420p')
        buffer_out = graph.add('buffersink')
        buffer_in.link_to(setpts)
        setpts.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(buffer_out)
        graph.configure()

        # Seek and decode video frames
        input_container.seek(int(safe_seek_pts / input_video.time_base), stream=input_video)
        for frame in input_container.decode(input_video):
            frame_pts = frame.pts * input_video.time_base
            if frame_pts < target_pts:
                continue
            if end_pts is not None and frame_pts >= end_pts:
                break

            graph.push(frame)
            for packet in video_stream.encode(graph.pull()):
                output_container.mux(packet)

        # Flush video encoder
        for packet in video_stream.encode():
            output_container.mux(packet)

        # Copy audio packets in range
        if input_audio and audio_stream:
            audio_time_base = input_audio.time_base
            audio_offset = int(target_pts / audio_time_base)

            input_container.seek(audio_offset, stream=input_audio)
            for packet in input_container.demux(input_audio):
                # The seek lands at/before the offset: skip packets from before
                # the start point (and packets without timestamps)
                if packet.pts is None or packet.pts < audio_offset:
                    continue
                if end_pts is not None and float(packet.pts * audio_time_base) >= float(end_pts):
                    break
                packet.pts -= audio_offset
                if packet.dts is not None:
                    packet.dts -= audio_offset
                packet.stream = audio_stream
                output_container.mux(packet)


def _parse_time(value: float | str) -> float:
    """Parse a time value given as seconds (float/int) or as a timestamp string.

    Accepted string formats: 'HH:MM:SS.fff', 'MM:SS.fff', 'SS.fff'
    (fractional part is optional).
    """
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(':')
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def _find_frame_at_time(index: FrameIndexPyAV, time_seconds: float) -> int:
    """Find the index of the first frame at or after the given time.

    Returns ``index.frame_count`` when the time is past the last frame, so a
    trim end at or beyond the video duration includes the final frame instead
    of clamping to (and then excluding) it.
    """
    target = Fraction(time_seconds).limit_denominator(1000000)
    for i, pts in enumerate(index.frame_pts):
        if pts >= target:
            return i
    return index.frame_count


# NVENC hardware minimum encode dimensions (H.264), verified empirically
_NVENC_MIN_W = 145
_NVENC_MIN_H = 49


def _nvenc_available() -> bool:
    """Check if NVENC is available."""
    try:
        import ctypes

        ctypes.CDLL('libnvidia-encode.so.1')
        return True
    except OSError:
        return False


def _float_to_uint16(frame: NDArray) -> NDArray:
    """Convert float frame [0,1] to uint16 [0,65535] for high precision encoding."""
    # Raw depth/label arrays fed by accident would silently clip to white;
    # tolerate float rounding jitter just past the bounds.
    if float(np.max(frame, initial=0.0)) > 1.001 or float(np.min(frame, initial=0.0)) < -0.001:
        warnings.warn(
            'Float frames are expected in [0, 1]; values outside this range are '
            'clipped. Scale your data (e.g. divide by its maximum) before writing.',
            stacklevel=3,
        )
    # Scale in float32: under NEP 50 promotion (numpy >= 2), float16 * 65535
    # stays float16 and overflows to inf, which the uint16 cast maps to 0.
    scaled = frame.astype(np.float32, copy=False) * np.float32(65535)
    return np.clip(scaled, 0, 65535).astype(np.uint16)


@dataclass
class StartSequence(Message):
    video_output: VideoOutput
    fps: float
    audio_source_path: PathLike | None = None
    audio_stream_index: int = 0
    gpu: bool | int = False
    encoder_config: EncoderConfig | None = None
    format: str | None = None


@dataclass
class AppendFrame(Message):
    frame: NDArray


class AbortSequence(Message):
    """Discard the current sequence: no file appears at the final path."""

    pass


class EndSequence(Message):
    pass


class Quit(Message):
    pass


class FeedbackMessage(Message):
    pass


@dataclass
class EndSequenceDone(FeedbackMessage):
    initial_msg: EndSequence


@dataclass
class ExceptionRaised(FeedbackMessage):
    """Worker failure notification.

    Carries the exception itself so consuming the message is sufficient —
    there is no separate stored state that could be consumed twice.
    """

    error: Exception
    video_path: PathLike | None = None
