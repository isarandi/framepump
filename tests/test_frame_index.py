"""Tests for frame indexing functionality."""

from pathlib import Path

import numpy as np
import pytest

from framepump import VideoFrames
from framepump._pyav import FrameIndexPyAV

TEST_DATA_DIR = Path(__file__).parent / 'data'

_has_test_data = TEST_DATA_DIR.exists() and any(TEST_DATA_DIR.glob('*.mp4'))
pytestmark = pytest.mark.skipif(not _has_test_data, reason='Test data not available')


@pytest.fixture
def short_video():
    """Path to a short test video."""
    p = TEST_DATA_DIR / 'short.mp4'
    if not p.exists():
        pytest.skip(f'{p.name} not found')
    return p


@pytest.fixture
def tiny_video():
    """Path to a tiny test video (~0.3s)."""
    p = TEST_DATA_DIR / 'tiny.mp4'
    if not p.exists():
        pytest.skip(f'{p.name} not found')
    return p


@pytest.fixture
def single_frame_video():
    """Path to a single-frame video."""
    p = TEST_DATA_DIR / 'single_frame.mp4'
    if not p.exists():
        pytest.skip(f'{p.name} not found')
    return p


class TestFrameIndex:
    """Tests for the FrameIndexPyAV class."""

    def test_build_index(self, short_video):
        """Test building an index from a video file."""
        index = FrameIndexPyAV(short_video)

        assert index.frame_count > 0
        assert len(index.frame_pts) == index.frame_count
        assert len(index.safe_seek_pts) == index.frame_count

    def test_get_seek_params(self, short_video):
        """Test getting seek parameters."""
        index = FrameIndexPyAV(short_video)

        seek_pts, trim = index.get_seek_params(0)
        assert seek_pts >= 0
        assert trim >= 0

        if index.frame_count > 5:
            seek_pts, trim = index.get_seek_params(5)
            assert seek_pts >= 0
            assert trim >= 0

    def test_get_frame_pts(self, short_video):
        """Test getting PTS for a frame."""
        index = FrameIndexPyAV(short_video)

        pts = index.get_frame_pts(0)
        assert pts >= 0

        if index.frame_count > 5:
            pts5 = index.get_frame_pts(5)
            assert pts5 > pts  # PTS should increase


class TestVideoFramesIndexing:
    """Tests for VideoFrames with indexing."""

    def test_integer_indexing_matches_iteration(self, short_video):
        """Test that indexed access returns same frame as iteration."""
        frames = VideoFrames(short_video)

        # Get frame via iteration
        frame_iter = None
        for i, f in enumerate(frames):
            if i == 5:
                frame_iter = f.copy()
                break

        # Get frame via indexing
        frame_indexed = frames[5]

        # Should be identical
        np.testing.assert_array_equal(frame_indexed, frame_iter)

    def test_negative_indexing(self, short_video):
        """Test negative index access."""
        frames = VideoFrames(short_video)

        # Get last frame via iteration
        last_iter = None
        for f in frames:
            last_iter = f.copy()

        # Get last frame via negative index
        last_indexed = frames[-1]

        np.testing.assert_array_equal(last_indexed, last_iter)

    def test_sliced_iteration_with_offset(self, short_video):
        """Test that sliced iteration with offset uses seeking."""
        frames = VideoFrames(short_video)

        # Get frames 5-10 via full iteration then slicing
        all_frames = list(frames)
        expected = all_frames[5:10]

        # Get frames 5-10 via sliced iteration (should use seeking)
        sliced_frames = list(frames[5:10])

        assert len(sliced_frames) == len(expected)
        for exp, got in zip(expected, sliced_frames):
            np.testing.assert_array_equal(exp, got)

    def test_sliced_iteration_with_step(self, short_video):
        """Test sliced iteration with step uses seeking correctly."""
        frames = VideoFrames(short_video)

        # Get every other frame starting from 4
        all_frames = list(frames)
        expected = all_frames[4::2]

        sliced_frames = list(frames[4::2])

        assert len(sliced_frames) == len(expected)
        for exp, got in zip(expected, sliced_frames):
            np.testing.assert_array_equal(exp, got)

    def test_single_frame_video(self, single_frame_video):
        """Test indexing on single-frame video."""
        frames = VideoFrames(single_frame_video)

        assert len(frames) == 1
        frame = frames[0]
        assert frame.shape[2] == 3  # RGB

        # Should also work via iteration
        frame_iter = next(iter(frames))
        np.testing.assert_array_equal(frame, frame_iter)

    def test_clone_shares_index(self, short_video):
        """Test that cloned VideoFrames shares the index."""
        frames = VideoFrames(short_video)

        sliced = frames[5:15]

        # Should share the same index object
        assert sliced._index is frames._index


class TestUint16Dtype:
    """Tests for uint16 dtype support."""

    def test_uint16_iteration(self, short_video):
        """Test that uint16 dtype produces correct frame shape and dtype."""
        frames = VideoFrames(short_video, dtype=np.uint16)

        frame = next(iter(frames))
        assert frame.dtype == np.uint16
        assert frame.shape[2] == 3  # RGB

    def test_uint16_indexing(self, short_video):
        """Test that uint16 dtype works with integer indexing."""
        frames = VideoFrames(short_video, dtype=np.uint16)

        frame = frames[5]
        assert frame.dtype == np.uint16
        assert frame.shape[2] == 3

    def test_uint16_sliced_iteration(self, short_video):
        """Test that uint16 dtype works with sliced iteration."""
        frames = VideoFrames(short_video, dtype=np.uint16)

        # Get frames 5-10 via sliced iteration
        sliced_frames = list(frames[5:10])

        assert len(sliced_frames) == 5
        for f in sliced_frames:
            assert f.dtype == np.uint16

    def test_uint16_iteration_matches_indexing(self, short_video):
        """Test that uint16 iteration and indexing return the same frame."""
        frames = VideoFrames(short_video, dtype=np.uint16)

        # Get frame via iteration
        frame_iter = None
        for i, f in enumerate(frames):
            if i == 5:
                frame_iter = f.copy()
                break

        # Get frame via indexing
        frame_indexed = frames[5]

        # Should be identical
        np.testing.assert_array_equal(frame_indexed, frame_iter)

    def test_uint16_has_higher_precision(self, short_video):
        """Test that uint16 values have more range than uint8."""
        frames_u8 = VideoFrames(short_video, dtype=np.uint8)
        frames_u16 = VideoFrames(short_video, dtype=np.uint16)

        frame_u8 = next(iter(frames_u8))
        frame_u16 = next(iter(frames_u16))

        # uint16 should have larger max value
        assert frame_u16.max() > frame_u8.max() or frame_u8.max() == 255


class TestSeekAccuracy:
    """Tests for seek accuracy."""

    def test_seek_returns_correct_frame(self, short_video):
        """Test that seeking returns the correct frame."""
        frames = VideoFrames(short_video)

        # Test multiple random positions
        all_frames = list(VideoFrames(short_video))

        for idx in [0, 5, 10, len(all_frames) - 1]:
            if idx < len(all_frames):
                indexed_frame = frames[idx]
                np.testing.assert_array_equal(
                    indexed_frame, all_frames[idx], err_msg=f'Frame {idx} mismatch'
                )

    def test_seek_with_resize(self, short_video):
        """Test that seeking works with resize."""
        frames = VideoFrames(short_video).resized((64, 64))

        # Get frame via indexing
        frame = frames[5]

        assert frame.shape[:2] == (64, 64)

        # Compare with iteration
        frame_iter = list(frames[5:6])[0]
        np.testing.assert_array_equal(frame, frame_iter)
