"""Shared pytest fixtures."""

import os

# Route OpenGL to the NVIDIA GPU on hybrid GPU systems (e.g., AMD iGPU + NVIDIA dGPU).
# Must be set before any GL library is loaded.
os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

import numpy as np
import pytest

from framepump import VideoWriter


@pytest.fixture
def create_test_video(tmp_path):
    """Factory fixture to create test videos with custom parameters."""

    def _create_video(
        filename='test.mp4',
        fps=30,
        n_frames=10,
        height=64,
        width=64,
        dtype=np.uint8,
    ):
        video_path = tmp_path / filename
        with VideoWriter(str(video_path), fps=fps) as writer:
            for i in range(n_frames):
                if dtype == np.uint8:
                    frame = np.full((height, width, 3), i * 10, dtype=dtype)
                else:
                    frame = np.full((height, width, 3), i * 1000, dtype=dtype)
                writer.append_data(frame)
        return str(video_path)

    return _create_video
