"""Tests for utility functions."""

import numpy as np
import pytest

from framepump import (
    VideoWriter,
    VideoFrames,
    get_fps,
    get_duration,
    num_frames,
    video_extents,
)
from framepump._core import has_audio


@pytest.fixture
def sample_video(tmp_path):
    """Create a small test video for testing."""
    video_path = tmp_path / 'test_video.mp4'
    fps = 25
    n_frames = 50
    height, width = 80, 120

    with VideoWriter(str(video_path), fps=fps) as writer:
        for i in range(n_frames):
            frame = np.full((height, width, 3), i * 5, dtype=np.uint8)
            writer.append_data(frame)

    return str(video_path), fps, n_frames, (height, width)


class TestGetFps:
    """Tests for get_fps function."""

    def test_get_fps_returns_float(self, sample_video):
        """Test that get_fps returns a float."""
        video_path, expected_fps, _, _ = sample_video
        fps = get_fps(video_path)
        assert isinstance(fps, float)

    def test_get_fps_correct_value(self, sample_video):
        """Test that get_fps returns correct value."""
        video_path, expected_fps, _, _ = sample_video
        fps = get_fps(video_path)
        assert fps == pytest.approx(expected_fps, rel=0.01)

    def test_get_fps_invalid_path(self):
        """Test that get_fps raises for invalid path."""
        with pytest.raises(Exception):
            get_fps('/nonexistent/video.mp4')


class TestGetDuration:
    """Tests for get_duration function."""

    def test_get_duration_returns_float(self, sample_video):
        """Test that get_duration returns a float."""
        video_path, _, _, _ = sample_video
        duration = get_duration(video_path)
        assert isinstance(duration, float)

    def test_get_duration_correct_value(self, sample_video):
        """Test that get_duration returns approximately correct value."""
        video_path, fps, n_frames, _ = sample_video
        expected_duration = n_frames / fps
        duration = get_duration(video_path)
        assert duration == pytest.approx(expected_duration, rel=0.1)

    def test_get_duration_invalid_path(self):
        """Test that get_duration raises for invalid path."""
        with pytest.raises(Exception):
            get_duration('/nonexistent/video.mp4')


class TestNumFrames:
    """Tests for num_frames function."""

    def test_num_frames_returns_int(self, sample_video):
        """Test that num_frames returns an int."""
        video_path, _, _, _ = sample_video
        n = num_frames(video_path)
        assert isinstance(n, int)

    def test_num_frames_correct_value(self, sample_video):
        """Test that num_frames returns correct value."""
        video_path, _, expected_n, _ = sample_video
        n = num_frames(video_path)
        assert n == expected_n

    def test_num_frames_exact(self, sample_video):
        """Test num_frames with exact=True."""
        video_path, _, expected_n, _ = sample_video
        n = num_frames(video_path, exact=True)
        assert n == expected_n

    def test_num_frames_invalid_path(self):
        """Test that num_frames raises for invalid path."""
        with pytest.raises(Exception):
            num_frames('/nonexistent/video.mp4')


class TestVideoExtents:
    """Tests for video_extents function."""

    def test_video_extents_returns_array(self, sample_video):
        """Test that video_extents returns a numpy array."""
        video_path, _, _, _ = sample_video
        extents = video_extents(video_path)
        assert isinstance(extents, np.ndarray)

    def test_video_extents_correct_value(self, sample_video):
        """Test that video_extents returns correct (width, height)."""
        video_path, _, _, (height, width) = sample_video
        extents = video_extents(video_path)
        assert extents[0] == width
        assert extents[1] == height

    def test_video_extents_invalid_path(self):
        """Test that video_extents raises for invalid path."""
        with pytest.raises(Exception):
            video_extents('/nonexistent/video.mp4')


class TestHasAudio:
    """Tests for has_audio function."""

    def test_has_audio_video_without_audio(self, sample_video):
        """Test has_audio returns False for video without audio."""
        video_path, _, _, _ = sample_video
        assert has_audio(video_path) is False
