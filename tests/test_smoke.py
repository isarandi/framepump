"""Smoke tests to verify basic functionality."""

import framepump


def test_version():
    """Test that version is accessible."""
    assert hasattr(framepump, '__version__')
    assert isinstance(framepump.__version__, str)


def test_imports():
    """Test that all public API can be imported."""
    from framepump import VideoFrames, VideoWriter, GLVideoWriter, AbstractVideoWriter
    from framepump import get_fps, get_duration, num_frames, video_extents, has_audio
    from framepump import trim_video, video_audio_mux


def test_videoframes_class_exists():
    """Test VideoFrames class has expected attributes."""
    from framepump import VideoFrames
    assert hasattr(VideoFrames, 'resized')
    assert hasattr(VideoFrames, 'repeat_each_frame')
    assert hasattr(VideoFrames, '__iter__')
    assert hasattr(VideoFrames, '__getitem__')
    assert hasattr(VideoFrames, '__len__')


def test_videowriter_class_exists():
    """Test VideoWriter class has expected attributes."""
    from framepump import VideoWriter
    assert hasattr(VideoWriter, 'start_sequence')
    assert hasattr(VideoWriter, 'end_sequence')
    assert hasattr(VideoWriter, 'append_data')
    assert hasattr(VideoWriter, 'close')
    assert hasattr(VideoWriter, '__enter__')
    assert hasattr(VideoWriter, '__exit__')
