"""Tests for VideoFrames class."""

import numpy as np
import pytest
import tempfile
import os

from framepump import VideoFrames, VideoWriter


@pytest.fixture
def sample_video(tmp_path):
    """Create a small test video for testing."""
    video_path = tmp_path / 'test_video.mp4'
    fps = 10
    n_frames = 30
    height, width = 64, 64

    with VideoWriter(str(video_path), fps=fps) as writer:
        for i in range(n_frames):
            # Create a frame with a unique pattern for each frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add frame number as intensity
            frame[:, :, 0] = i * 8  # Red channel varies with frame
            frame[:, :, 1] = 128    # Green constant
            frame[:, :, 2] = 64     # Blue constant
            writer.append_data(frame)

    return str(video_path), fps, n_frames, (height, width)


class TestVideoFramesBasic:
    """Basic VideoFrames functionality tests."""

    def test_create_videoframes(self, sample_video):
        """Test creating a VideoFrames instance."""
        video_path, fps, n_frames, imshape = sample_video
        frames = VideoFrames(video_path)

        assert frames.path == video_path
        assert frames.original_fps == fps
        assert tuple(frames.imshape) == imshape

    def test_len(self, sample_video):
        """Test __len__ returns correct frame count."""
        video_path, fps, n_frames, _ = sample_video
        frames = VideoFrames(video_path)
        assert len(frames) == n_frames

    def test_iterate_frames(self, sample_video):
        """Test iterating over frames yields numpy arrays."""
        video_path, fps, n_frames, imshape = sample_video
        frames = VideoFrames(video_path)

        count = 0
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (*imshape, 3)
            assert frame.dtype == np.uint8
            count += 1

        assert count == n_frames

    def test_fps_property(self, sample_video):
        """Test fps property returns correct value."""
        video_path, fps, _, _ = sample_video
        frames = VideoFrames(video_path)
        assert frames.fps == fps

    def test_imshape_property(self, sample_video):
        """Test imshape property returns correct value."""
        video_path, _, _, imshape = sample_video
        frames = VideoFrames(video_path)
        assert tuple(frames.imshape) == imshape


class TestVideoFramesSlicing:
    """Test slicing operations on VideoFrames."""

    def test_slice_start_stop(self, sample_video):
        """Test slicing with start:stop."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        sliced = frames[5:15]
        assert len(sliced) == 10

    def test_slice_with_step(self, sample_video):
        """Test slicing with step."""
        video_path, fps, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        # Every second frame
        sliced = frames[::2]
        assert len(sliced) == (n_frames + 1) // 2

        # FPS should be halved when step is 2
        assert sliced.fps == fps / 2

    def test_slice_chaining(self, sample_video):
        """Test chaining multiple slices."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        # First get every 2nd frame, then take first 10
        sliced = frames[::2][:10]
        assert len(sliced) == 10

    def test_slice_returns_videoframes(self, sample_video):
        """Test that slicing returns a new VideoFrames instance."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        sliced = frames[5:15]
        assert isinstance(sliced, VideoFrames)
        assert sliced is not frames

    def test_slice_is_lazy(self, sample_video):
        """Test that slicing doesn't read frames immediately."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        # This should be instant (no frame reading)
        sliced = frames[::10][:5]
        assert len(sliced) == min(5, (30 + 9) // 10)


class TestVideoFramesResize:
    """Test resize functionality."""

    def test_resized_returns_new_instance(self, sample_video):
        """Test that resized() returns a new VideoFrames instance."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        resized = frames.resized((32, 32))
        assert isinstance(resized, VideoFrames)
        assert resized is not frames

    def test_resized_imshape(self, sample_video):
        """Test that resized frames have correct shape."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        new_shape = (32, 48)
        resized = frames.resized(new_shape)
        assert resized.imshape == new_shape

    def test_resized_frames_actual_shape(self, sample_video):
        """Test that iterated frames have the resized shape."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        new_shape = (32, 48)
        resized = frames.resized(new_shape)

        for frame in resized:
            assert frame.shape == (new_shape[0], new_shape[1], 3)
            break  # Only check first frame


class TestVideoFramesRepeat:
    """Test frame repetition functionality."""

    def test_repeat_each_frame(self, sample_video):
        """Test repeat_each_frame multiplies frame count."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        repeated = frames.repeat_each_frame(3)
        assert len(repeated) == n_frames * 3

    def test_repeat_fps(self, sample_video):
        """Test that fps is multiplied when repeating frames."""
        video_path, fps, _, _ = sample_video
        frames = VideoFrames(video_path)

        repeated = frames.repeat_each_frame(2)
        assert repeated.fps == fps * 2

    def test_repeat_invalid_count(self, sample_video):
        """Test that repeat_each_frame rejects invalid counts."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        with pytest.raises(ValueError):
            frames.repeat_each_frame(0)


class TestVideoFramesDtype:
    """Test dtype conversion functionality."""

    def test_default_dtype_uint8(self, sample_video):
        """Test default dtype is uint8."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        for frame in frames:
            assert frame.dtype == np.uint8
            break

    def test_dtype_uint16(self, sample_video):
        """Test uint16 dtype."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path, dtype=np.uint16)

        for frame in frames:
            assert frame.dtype == np.uint16
            break

    def test_dtype_float32(self, sample_video):
        """Test float32 dtype normalization."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path, dtype=np.float32)

        for frame in frames:
            assert frame.dtype == np.float32
            assert frame.min() >= 0.0
            assert frame.max() <= 1.0
            break

    def test_invalid_dtype(self, sample_video):
        """Test that invalid dtypes are rejected."""
        video_path, _, _, _ = sample_video

        with pytest.raises(ValueError):
            VideoFrames(video_path, dtype=np.int32)
