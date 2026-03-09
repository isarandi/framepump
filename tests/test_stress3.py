"""Esoteric edge cases - digging deeper for bugs."""

import numpy as np
import pytest
from pathlib import Path

from framepump import VideoFrames, VideoWriter, num_frames, get_fps, get_duration

DATA_DIR = Path(__file__).parent.parent / 'data'
FATE_DIR = Path(__file__).parent.parent / 'fate'


class TestSlicingAfterSeek:
    """Test slicing behavior after random access."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(50):
                frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_len_after_slice(self, sample_video):
        """Length after slicing should be accurate."""
        frames = VideoFrames(sample_video)

        sliced = frames[10:40]  # 30 frames
        assert len(sliced) == 30

        double_sliced = sliced[5:20]  # 15 frames
        assert len(double_sliced) == 15

    def test_index_after_slice(self, sample_video):
        """Indexing after slice should use relative indices."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:40]

        # Index 0 of sliced should be frame 10 of original
        # Hard to verify pixel values after encoding, but at least it shouldn't crash
        f = sliced[0]
        assert f.ndim == 3

        # Index -1 of sliced should be frame 39 of original
        f = sliced[-1]
        assert f.ndim == 3

    def test_out_of_bounds_after_slice(self, sample_video):
        """Out of bounds indexing after slice."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:20]  # 10 frames

        with pytest.raises(IndexError):
            _ = sliced[15]  # Only 10 frames available

    def test_iteration_count_matches_len_after_slice(self, sample_video):
        """Iteration count should match len() after slicing."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:40:2]

        expected = len(sliced)
        actual = sum(1 for _ in sliced)
        assert actual == expected


class TestNumFramesMethods:
    """Test different frame counting methods."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(47):  # Odd number
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_num_frames_default(self, sample_video):
        """Default frame count estimation."""
        count = num_frames(sample_video)
        # Should be close to 47
        assert abs(count - 47) <= 2

    def test_num_frames_exact(self, sample_video):
        """Exact frame count."""
        count = num_frames(sample_video, exact=True)
        # This should be exactly right or very close
        assert abs(count - 47) <= 1

    def test_num_frames_absolutely_exact(self, sample_video):
        """Absolutely exact frame count (decodes all frames)."""
        count = num_frames(sample_video, absolutely_exact=True)
        # This must be exact
        assert count == 47


class TestGrayscaleHandling:
    """Test behavior with grayscale input."""

    def test_write_grayscale_fails(self, tmp_path):
        """Writing grayscale frames should fail or be handled."""
        video_path = tmp_path / 'gray.mp4'

        # This should probably fail since the writer expects RGB
        with pytest.raises(Exception):
            with VideoWriter(str(video_path), fps=10) as writer:
                frame = np.zeros((64, 64), dtype=np.uint8)  # 2D, not 3D
                writer.append_data(frame)


class TestNonContiguousArrays:
    """Test behavior with non-contiguous numpy arrays."""

    def test_write_non_contiguous(self, tmp_path):
        """Write non-contiguous array."""
        video_path = tmp_path / 'non_contig.mp4'

        # Create a non-contiguous array (e.g., transposed or sliced)
        full = np.zeros((64, 64, 6), dtype=np.uint8)
        non_contig = full[:, :, ::2]  # Every other channel - non-contiguous
        assert not non_contig.flags['C_CONTIGUOUS']

        with VideoWriter(str(video_path), fps=10) as writer:
            writer.append_data(non_contig)

        # Should work - iter_video_write calls np.ascontiguousarray
        assert video_path.exists()


class TestDTypeWriting:
    """Test writing various dtypes."""

    def test_write_uint16(self, tmp_path):
        """Write uint16 frames and verify precision is preserved."""
        video_path = tmp_path / 'uint16.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((64, 64, 3), dtype=np.uint16)
            frame[:] = 32768  # Mid value
            writer.append_data(frame)

        # Read back as uint16 — should preserve more precision than uint8
        frames_16 = VideoFrames(str(video_path), dtype=np.uint16)
        f16 = frames_16[0]
        assert f16.dtype == np.uint16
        # Mid-gray (32768/65535 ≈ 0.5) → uint16 should be near 32768
        # Allow tolerance for codec lossy compression
        assert f16.mean() > 255, (
            f'uint16 read-back mean is {f16.mean()}, expected >> 255 '
            f'(uint16 precision not preserved)'
        )

    def test_write_float32_converts_to_uint16(self, tmp_path):
        """Writing float32 frames auto-converts to uint16 for high precision."""
        video_path = tmp_path / 'float32.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            # Values in [0,1] range
            frame = np.full((64, 64, 3), 0.5, dtype=np.float32)
            writer.append_data(frame)

        # Verify the video was created and is readable
        frames = VideoFrames(str(video_path))
        assert len(frames) == 1
        f = frames[0]
        # Mid-gray value (0.5 * 255 ≈ 128)
        assert 100 < f.mean() < 160


class TestSeekingBeyondVideo:
    """Test seeking edge cases."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(20):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_index_exact_last(self, sample_video):
        """Index the exact last frame."""
        frames = VideoFrames(sample_video)
        last_idx = len(frames) - 1
        f = frames[last_idx]
        assert f.ndim == 3

    def test_index_one_beyond_last(self, sample_video):
        """Index one beyond last should raise IndexError."""
        frames = VideoFrames(sample_video)
        with pytest.raises(IndexError):
            _ = frames[len(frames)]


class TestConstantFramerateValues:
    """Test constant_framerate parameter variations."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(30):
                # Use a gradient so frames have real color variation
                frame = np.full((64, 64, 3), i * 8, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_cfr_false(self, sample_video):
        """constant_framerate=False (VFR mode)."""
        frames = VideoFrames(sample_video, constant_framerate=False)
        count = sum(1 for _ in frames)
        assert count > 0

    def test_cfr_true(self, sample_video):
        """constant_framerate=True (default CFR)."""
        frames = VideoFrames(sample_video, constant_framerate=True)
        count = sum(1 for _ in frames)
        assert count > 0

    def test_cfr_half_fps(self, sample_video):
        """constant_framerate set to half the FPS."""
        original_fps = get_fps(sample_video)
        frames = VideoFrames(sample_video, constant_framerate=original_fps / 2)
        # Should get roughly half the frames
        assert len(frames) < 20

    def test_cfr_double_fps(self, sample_video):
        """constant_framerate set to double the FPS."""
        original_fps = get_fps(sample_video)
        frames = VideoFrames(sample_video, constant_framerate=original_fps * 2)
        # Should get roughly double the frames
        assert len(frames) > 40


def _load_fate_categories():
    """Load fate file categories (verified with ffmpeg)."""
    import json
    categories_path = Path(__file__).parent / 'fate_categories.json'
    if not categories_path.exists():
        return None
    with open(categories_path) as f:
        return json.load(f)


def _collect_fate_files_by_category():
    """Collect fate files grouped by expected behavior."""
    if not FATE_DIR.exists():
        return [], [], []

    categories = _load_fate_categories()
    if categories is None:
        return [], [], []

    has_video = [(FATE_DIR / p, 'has_video') for p in categories['has_video']]
    no_video = [(FATE_DIR / p, 'no_video') for p in categories['no_video']]
    errors = [(FATE_DIR / p, 'error') for p in categories['decode_error'] + categories['not_recognized']]

    return has_video, no_video, errors


class TestFateHasVideo:
    """Test fate files that MUST decode successfully (verified with ffmpeg)."""

    _has_video, _, _ = _collect_fate_files_by_category()

    @pytest.mark.parametrize('video_path', [p for p, _ in _has_video],
                             ids=lambda p: str(p.relative_to(FATE_DIR)))
    def test_must_decode(self, video_path):
        """File must decode - ffmpeg verified it has video."""
        frames = VideoFrames(str(video_path))
        _ = len(frames)
        _ = frames.fps
        for f in frames:
            assert f.ndim == 3
            break


class TestFateNoVideo:
    """Test fate files that have no video stream (verified with ffprobe)."""

    _, _no_video, _ = _collect_fate_files_by_category()

    @pytest.mark.parametrize('video_path', [p for p, _ in _no_video],
                             ids=lambda p: str(p.relative_to(FATE_DIR)))
    def test_must_raise_no_video(self, video_path):
        """File must raise 'No video stream' error."""
        with pytest.raises(ValueError, match='No video stream'):
            VideoFrames(str(video_path))


class TestFateDecodeError:
    """Test fate files that ffmpeg cannot decode (corrupt/unsupported)."""

    _, _, _errors = _collect_fate_files_by_category()

    @pytest.mark.parametrize('video_path', [p for p, _ in _errors],
                             ids=lambda p: str(p.relative_to(FATE_DIR)))
    def test_must_error(self, video_path):
        """File must raise some error (ffmpeg also fails on these)."""
        with pytest.raises(Exception):
            frames = VideoFrames(str(video_path))
            _ = len(frames)
            for f in frames:
                break


class TestRepeatWithSlice:
    """Test interaction between repeat and slice."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(10):
                frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_repeat_then_len(self, sample_video):
        """Length after repeat should be multiplied."""
        frames = VideoFrames(sample_video)
        repeated = frames.repeat_each_frame(3)
        assert len(repeated) == 30

    def test_repeat_iteration(self, sample_video):
        """Iterating repeated frames should yield duplicates."""
        frames = VideoFrames(sample_video)
        repeated = frames.repeat_each_frame(2)

        prev = None
        same_count = 0
        for i, f in enumerate(repeated):
            if prev is not None and np.array_equal(f, prev):
                same_count += 1
            prev = f.copy()
            if i >= 10:
                break

        # Should have some consecutive same frames
        assert same_count > 0


class TestEmptyOrZeroEdgeCases:
    """Test zero/empty edge cases."""

    def test_zero_step_slice_raises_immediately(self, tmp_path):
        """Zero step raises error immediately during slicing."""
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(10):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        # Error is raised immediately when slicing with step=0
        with pytest.raises(ValueError, match='zero'):
            _ = frames[::0]


class TestNegativeIndexSlicing:
    """Test negative index slicing (not reverse iteration)."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(20):
                frame = np.full((64, 64, 3), i * 12, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_negative_start_index(self, sample_video):
        """Negative start index should work."""
        frames = VideoFrames(sample_video)
        last_five = frames[-5:]
        assert len(last_five) == 5

    def test_negative_stop_index(self, sample_video):
        """Negative stop index should work."""
        frames = VideoFrames(sample_video)
        without_last_five = frames[:-5]
        assert len(without_last_five) == 15




if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
