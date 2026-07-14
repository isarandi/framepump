"""Failure-injection tests for VideoWriter's producer/worker lifecycle.

These cover the failure modes of the threaded writer: worker death must never
deadlock the producer, must never promote a partial output file to the final
path, and must leave the writer in a state that either fails fast or can be
deliberately restarted via start_sequence().

All tests are hang-proof: anything that could block runs through
call_with_timeout(), and the whole module has a pytest-timeout safety net.
"""

import gc
import time
import threading
import warnings

import numpy as np
import pytest

from framepump import VideoFrames, VideoWriter

pytestmark = pytest.mark.timeout(120)

GOOD_SHAPE = (48, 48, 3)


def good_frame(value=128):
    return np.full(GOOD_SHAPE, value, dtype=np.uint8)


def bad_frame():
    """Frame whose shape mismatches the first frame -> worker raises mid-sequence."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


def call_with_timeout(fn, timeout=15.0):
    """Run fn in a daemon thread; fail the test if it doesn't return in time."""
    outcome = {}

    def target():
        try:
            outcome['value'] = fn()
        except BaseException as e:
            outcome['error'] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f'call did not return within {timeout}s (deadlock)')
    if 'error' in outcome:
        raise outcome['error']
    return outcome.get('value')


def wait_for_worker_death(writer, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = writer._thread
        if thread is None or not thread.is_alive():
            return
        time.sleep(0.05)
    raise TimeoutError('worker thread did not die in time')


def kill_worker_with_bad_frame(writer, path):
    """Start a sequence, write one good frame, then a shape-mismatched one."""
    writer.start_sequence(str(path), fps=30)
    writer.append_data(good_frame())
    writer.append_data(bad_frame())
    wait_for_worker_death(writer)


def no_temp_files(directory):
    return not list(directory.glob('*.tmp_*'))


def wait_for_queue_drain(writer, timeout=10.0):
    """Wait until the worker has consumed every queued message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if writer._queue.qsize() == 0:
            time.sleep(0.2)  # let the worker finish processing the last message
            return
        time.sleep(0.05)
    raise TimeoutError('worker did not drain the queue in time')


class TestNoDeadlock:
    def test_close_after_worker_death_with_full_queue(self, tmp_path):
        """Worker dies, queue fills up, close() must not block forever."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter(queue_size=1)
        kill_worker_with_bad_frame(writer, video_path)

        # Every subsequent producer call must return promptly, either
        # succeeding or raising the worker's error - never deadlocking.
        with pytest.raises(RuntimeError):
            call_with_timeout(lambda: writer.end_sequence(block=False))
        call_with_timeout(writer.close)

    def test_append_after_worker_death_raises_promptly(self, tmp_path):
        """append_data into a dead worker raises instead of blocking forever."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter(queue_size=1)
        kill_worker_with_bad_frame(writer, video_path)

        with pytest.raises(RuntimeError, match='shape'):
            for _ in range(5):
                call_with_timeout(lambda: writer.append_data(good_frame()))
        call_with_timeout(writer.close)

    def test_blocking_end_sequence_after_worker_death(self, tmp_path):
        """end_sequence(block=True) reports the worker error, never hangs."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        kill_worker_with_bad_frame(writer, video_path)

        with pytest.raises(RuntimeError, match='shape'):
            call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)


class TestNoPartialOutput:
    def test_worker_error_leaves_no_output_file(self, tmp_path):
        """An encoding error must not promote a truncated file to the final path."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        for _ in range(3):
            writer.append_data(good_frame())
        writer.append_data(bad_frame())
        wait_for_worker_death(writer)

        with pytest.raises(RuntimeError):
            call_with_timeout(writer.close)

        assert not video_path.exists()
        assert no_temp_files(tmp_path)

    def test_shutdown_leaves_no_partial_file(self, tmp_path):
        """shutdown() mid-sequence aborts the file instead of finalizing it."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        for _ in range(3):
            writer.append_data(good_frame())
        wait_for_queue_drain(writer)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            call_with_timeout(writer.shutdown)

        assert not video_path.exists()
        assert no_temp_files(tmp_path)

    def test_keyboard_interrupt_exit_leaves_no_partial_file(self, tmp_path):
        """The Ctrl+C context-manager path aborts instead of finalizing."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        writer.append_data(good_frame())
        wait_for_queue_drain(writer)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            call_with_timeout(lambda: writer.__exit__(KeyboardInterrupt, None, None))

        assert not video_path.exists()
        assert no_temp_files(tmp_path)


class TestReuse:
    def test_reusable_after_shutdown(self, tmp_path):
        """A writer must actually work again after shutdown()."""
        video1 = tmp_path / 'one.mp4'
        video2 = tmp_path / 'two.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video1), fps=30)
        writer.append_data(good_frame())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            call_with_timeout(writer.shutdown)

        writer.start_sequence(str(video2), fps=30)
        for i in range(5):
            call_with_timeout(lambda i=i: writer.append_data(good_frame(i * 40)))
        call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)

        frames = list(VideoFrames(str(video2)))
        assert len(frames) == 5
        for i, frame in enumerate(frames):
            assert abs(float(np.median(frame)) - i * 40) < 10

    def test_reusable_after_worker_error(self, tmp_path):
        """After the error was reported, start_sequence() starts a fresh worker."""
        video1 = tmp_path / 'one.mp4'
        video2 = tmp_path / 'two.mp4'
        writer = VideoWriter()
        kill_worker_with_bad_frame(writer, video1)

        with pytest.raises(RuntimeError):
            call_with_timeout(lambda: writer.end_sequence(block=True))

        writer.start_sequence(str(video2), fps=30)
        for i in range(5):
            call_with_timeout(lambda i=i: writer.append_data(good_frame(i * 40)))
        call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)

        frames = list(VideoFrames(str(video2)))
        assert len(frames) == 5
        for i, frame in enumerate(frames):
            assert abs(float(np.median(frame)) - i * 40) < 10

    def test_stale_state_not_replayed_after_restart(self, tmp_path):
        """Queued messages from before a shutdown must not leak into the new worker."""
        video1 = tmp_path / 'one.mp4'
        video2 = tmp_path / 'two.mp4'
        writer = VideoWriter(queue_size=32)
        writer.start_sequence(str(video1), fps=30)
        for _ in range(10):
            writer.append_data(good_frame())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            call_with_timeout(writer.shutdown)

        writer.start_sequence(str(video2), fps=30)
        for _ in range(3):
            call_with_timeout(lambda: writer.append_data(good_frame()))
        call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)

        assert not video1.exists()
        frames = VideoFrames(str(video2))
        assert len(frames) == 3


class TestErrorReporting:
    def test_error_raised_once_then_close_is_silent(self, tmp_path):
        """The error surfaces exactly once via producer calls; close() then passes."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        kill_worker_with_bad_frame(writer, video_path)

        with pytest.raises(RuntimeError, match='shape'):
            call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)

    def test_never_assertion_error(self, tmp_path):
        """Consuming the error twice must give RuntimeError, never AssertionError."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        kill_worker_with_bad_frame(writer, video_path)

        errors = []
        for op in (
            lambda: writer.append_data(good_frame()),
            lambda: writer.end_sequence(block=True),
            lambda: writer.end_sequence(block=True),
        ):
            try:
                call_with_timeout(op)
            except (RuntimeError, ValueError) as e:
                errors.append(e)
            except AssertionError as e:
                pytest.fail(f'AssertionError leaked to caller: {e}')
        assert any(isinstance(e, RuntimeError) for e in errors)
        call_with_timeout(writer.close)

    def test_unreported_error_raises_in_close(self, tmp_path):
        """If no producer call observed the error, close() must report it."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        writer.append_data(good_frame())
        writer.append_data(bad_frame())
        wait_for_worker_death(writer)

        with pytest.raises(RuntimeError, match='shape'):
            call_with_timeout(writer.close)


class TestResourceHygiene:
    def test_no_feedback_backlog(self, tmp_path):
        """Repeated end_sequence(block=False) must not grow the feedback queue."""
        writer = VideoWriter()
        for i in range(15):
            video_path = tmp_path / f'seq_{i}.mp4'
            writer.start_sequence(str(video_path), fps=30)
            writer.append_data(good_frame())
            call_with_timeout(lambda: writer.end_sequence(block=False))
        call_with_timeout(writer.close)
        assert writer._feedback_queue.qsize() == 0

    def test_del_warns_about_forgotten_close(self, tmp_path):
        """Garbage collection of an active writer warns instead of staying silent."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        writer.append_data(good_frame())

        with pytest.warns(ResourceWarning):
            writer.__del__()
        wait_for_worker_death(writer)
        gc.collect()


class TestNormalOperation:
    def test_happy_path_content_roundtrip(self, tmp_path):
        """The restructured lifecycle still writes correct content."""
        video_path = tmp_path / 'out.mp4'
        n = 10
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(n):
                writer.append_data(good_frame(i * 20))

        frames = list(VideoFrames(str(video_path)))
        assert len(frames) == n
        for i, frame in enumerate(frames):
            assert abs(float(np.median(frame)) - i * 20) < 10

    def test_zero_frame_sequence_creates_no_file(self, tmp_path):
        """A sequence with no frames produces no output file (pinned behavior)."""
        video_path = tmp_path / 'out.mp4'
        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30)
        call_with_timeout(lambda: writer.end_sequence(block=True))
        call_with_timeout(writer.close)
        assert not video_path.exists()
        assert no_temp_files(tmp_path)
