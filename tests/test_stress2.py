"""Additional stress tests - more edge cases to break framepump."""

import gc
import numpy as np
import pytest
import threading
import time
from pathlib import Path

from framepump import VideoFrames, VideoWriter, get_fps, get_duration, num_frames


DATA_DIR = Path(__file__).parent.parent / 'data'
FATE_DIR = Path(__file__).parent.parent / 'fate'

_has_data = DATA_DIR.exists() and any(DATA_DIR.glob('*.*'))


class TestVeryShortVideos:
    """Test videos with very few frames."""

    def test_one_frame_video(self, tmp_path):
        """Single frame video."""
        video_path = tmp_path / 'one_frame.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        assert len(frames) == 1

        # Should be able to iterate
        count = sum(1 for _ in frames)
        assert count == 1

        # Should be able to index
        frame = frames[0]
        assert frame.ndim == 3

    def test_two_frame_video(self, tmp_path):
        """Two frame video."""
        video_path = tmp_path / 'two_frames.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(2):
                frame = np.full((64, 64, 3), i * 127, dtype=np.uint8)
                writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        assert len(frames) == 2

        # Frames should be different
        f0 = frames[0]
        f1 = frames[1]
        assert not np.array_equal(f0, f1)


class TestSliceChaining:
    """Test complex slice chaining."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(100):
                frame = np.full((64, 64, 3), i * 2, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_triple_slice(self, sample_video):
        """Three chained slices."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:90][5:40][::2]
        count = sum(1 for _ in sliced)
        # 10:90 = 80 frames, [5:40] = 35 frames, [::2] = 18 frames
        assert count == 18

    def test_quadruple_slice(self, sample_video):
        """Four chained slices."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:90][5:40][::2][:5]
        assert len(sliced) == 5

    def test_slice_to_single_frame(self, sample_video):
        """Slice down to just one frame."""
        frames = VideoFrames(sample_video)
        sliced = frames[50:51]
        assert len(sliced) == 1

    def test_slice_then_resize(self, sample_video):
        """Slice then resize."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:20].resized((32, 32))
        for f in sliced:
            assert f.shape == (32, 32, 3)
            break


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(60):
                frame = np.full((64, 64, 3), i * 4, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_multiple_iterators_same_video(self, sample_video):
        """Multiple iterators on same VideoFrames."""
        frames = VideoFrames(sample_video)

        # Create two iterators
        iter1 = iter(frames)
        iter2 = iter(frames)

        # Interleave reads
        f1_a = next(iter1)
        f2_a = next(iter2)
        f1_b = next(iter1)
        f2_b = next(iter2)

        # First frames from each should be equal
        np.testing.assert_array_equal(f1_a, f2_a)

    def test_random_access_while_iterating(self, sample_video):
        """Random access while another iterator is active."""
        frames = VideoFrames(sample_video)

        iterator = iter(frames)
        _ = next(iterator)  # Start iterating

        # Now do random access
        f10 = frames[10]
        f20 = frames[20]

        # Continue iterating
        _ = next(iterator)
        _ = next(iterator)

        # Random access should still work
        f10_again = frames[10]
        np.testing.assert_array_equal(f10, f10_again)

    def test_threaded_reads(self, sample_video):
        """Read from multiple threads."""
        errors = []
        results = {}

        def read_frames(frames, start, end, key):
            try:
                for i in range(start, end):
                    f = frames[i]
                    if f.ndim != 3:
                        errors.append(f'Wrong ndim at {i}')
                results[key] = True
            except Exception as e:
                errors.append(str(e))

        frames = VideoFrames(sample_video)

        threads = []
        for i in range(3):
            t = threading.Thread(target=read_frames, args=(frames, i*10, (i+1)*10, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f'Errors: {errors}'
        assert len(results) == 3


class TestWriterThreadingEdgeCases:
    """Test VideoWriter threading edge cases."""

    def test_rapid_sequence_switches(self, tmp_path):
        """Rapidly switch between sequences."""
        writer = VideoWriter()

        for i in range(10):
            path = tmp_path / f'rapid_{i}.mp4'
            writer.start_sequence(str(path), fps=30)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)
            writer.end_sequence()

        writer.close()

        # All files should exist
        for i in range(10):
            assert (tmp_path / f'rapid_{i}.mp4').exists()

    def test_large_queue_fill(self, tmp_path):
        """Fill the queue rapidly."""
        video_path = tmp_path / 'queue_fill.mp4'

        with VideoWriter(str(video_path), fps=30, queue_size=8) as writer:
            # Write faster than encoder can process
            for i in range(50):
                frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                writer.append_data(frame)

        assert video_path.exists()

    def test_writer_close_without_end_sequence(self, tmp_path):
        """Close writer without explicitly ending sequence."""
        video_path = tmp_path / 'no_end.mp4'

        writer = VideoWriter(str(video_path), fps=30)
        for i in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)

        # Just close, don't end_sequence
        writer.close()

        # File should still be valid
        assert video_path.exists()
        frames = VideoFrames(str(video_path))
        assert len(frames) > 0


class TestFrameIndexEdgeCases:
    """Test frame index edge cases."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(30):
                frame = np.full((64, 64, 3), i * 8, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_seek_first_then_last(self, sample_video):
        """Seek to first frame then last."""
        frames = VideoFrames(sample_video)
        f_first = frames[0]
        f_last = frames[-1]
        assert not np.array_equal(f_first, f_last)

    def test_seek_last_then_first(self, sample_video):
        """Seek to last frame then first."""
        frames = VideoFrames(sample_video)
        f_last = frames[-1]
        f_first = frames[0]
        assert not np.array_equal(f_first, f_last)

    def test_repeated_seek_same_frame(self, sample_video):
        """Seek to same frame multiple times."""
        frames = VideoFrames(sample_video)

        results = []
        for _ in range(5):
            results.append(frames[15].copy())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestMemoryAndResources:
    """Test for memory leaks and resource cleanup."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(30):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_many_videoframes_instances(self, sample_video):
        """Create many VideoFrames instances."""
        for _ in range(50):
            frames = VideoFrames(sample_video)
            _ = len(frames)
            del frames

        gc.collect()
        # If we get here without error, we're okay

    def test_many_incomplete_iterations(self, sample_video):
        """Start many iterations without completing."""
        for _ in range(30):
            frames = VideoFrames(sample_video)
            iterator = iter(frames)
            _ = next(iterator)
            _ = next(iterator)
            # Don't finish iteration
            del iterator
            del frames

        gc.collect()


class TestSpecialResolutions:
    """Test special resolutions and aspect ratios."""

    def test_very_wide_video(self, tmp_path):
        """Very wide aspect ratio."""
        video_path = tmp_path / 'wide.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((32, 640, 3), dtype=np.uint8)  # 20:1 aspect
            writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        for f in frames:
            assert f.shape[0] == 32
            assert f.shape[1] == 640
            break

    def test_very_tall_video(self, tmp_path):
        """Very tall aspect ratio."""
        video_path = tmp_path / 'tall.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((640, 32, 3), dtype=np.uint8)  # 1:20 aspect
            writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        for f in frames:
            assert f.shape[0] == 640
            assert f.shape[1] == 32
            break

    def test_square_video(self, tmp_path):
        """Square video."""
        video_path = tmp_path / 'square.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        for f in frames:
            assert f.shape[0] == 128
            assert f.shape[1] == 128
            break


class TestFPSEdgeCases:
    """Test various FPS values."""

    @pytest.mark.parametrize('fps', [0.5, 1, 15, 23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120])
    def test_various_fps(self, tmp_path, fps):
        """Write videos at various FPS values."""
        video_path = tmp_path / f'fps_{fps}.mp4'

        with VideoWriter(str(video_path), fps=fps) as writer:
            for i in range(10):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)

        # Read back and check FPS
        actual_fps = get_fps(str(video_path))
        assert abs(actual_fps - fps) < 0.1, f'Expected {fps}, got {actual_fps}'


@pytest.mark.skipif(not _has_data, reason='data/ test files not available')
class TestRealVideoEdgeCases:
    """Test edge cases with real video files from data/."""

    def test_webm_files(self):
        """Test WebM files from data/."""
        webms = list(DATA_DIR.glob('*.webm'))
        if not webms:
            pytest.skip('No webm files')

        for webm in webms[:2]:
            frames = VideoFrames(str(webm))
            count = 0
            for f in frames:
                count += 1
                if count >= 10:
                    break
            assert count > 0

    def test_4k_videos(self):
        """Test 4K videos."""
        videos_4k = [v for v in DATA_DIR.glob('*.mp4') if '3840' in v.name or '2160' in v.name]
        if not videos_4k:
            pytest.skip('No 4K videos')

        video = videos_4k[0]
        frames = VideoFrames(str(video))

        # Just verify we can get metadata
        assert frames.imshape[0] >= 2000 or frames.imshape[1] >= 3000

        # Read first frame
        f = frames[0]
        assert f.ndim == 3


class TestCloneSharing:
    """Test that clones share resources properly."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(30):
                frame = np.full((64, 64, 3), i * 8, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_slice_shares_index(self, sample_video):
        """Sliced VideoFrames should share frame index."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:20]

        # They should share the same index object
        assert frames._index is sliced._index

    def test_resize_shares_index(self, sample_video):
        """Resized VideoFrames should share frame index."""
        frames = VideoFrames(sample_video)
        resized = frames.resized((32, 32))

        assert frames._index is resized._index

    def test_multiple_clones(self, sample_video):
        """Multiple operations should share index."""
        frames = VideoFrames(sample_video)
        modified = frames[::2][:10].resized((32, 32))

        assert frames._index is modified._index


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])