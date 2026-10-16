"""Shared pytest fixtures."""

import os

# Route OpenGL to the NVIDIA GPU on hybrid GPU systems (e.g., AMD iGPU + NVIDIA dGPU).
# Must be set before any GL library is loaded.
os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from framepump import VideoWriter  # noqa: E402


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


# ---------------------------------------------------------------------------
# GL contexts for the NVENC / GL-writer tests. Headless must not need a
# display: without DISPLAY a standalone EGL context is created (what the
# headless production path uses); with DISPLAY the GLFW/GLX path is used so
# that coverage survives too. Both yield a moderngl context.
_glfw_initialized = False


def make_gl_test_context():
    """Return (moderngl_ctx, cleanup_callable) on an NVIDIA GPU, or skip."""
    import moderngl

    if os.environ.get('DISPLAY'):
        global _glfw_initialized
        import glfw

        if not _glfw_initialized:
            if not glfw.init():
                pytest.skip('Failed to initialize GLFW (DISPLAY is set but unusable)')
            _glfw_initialized = True
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(640, 480, 'test', None, None)
        if not window:
            pytest.skip('Failed to create GLFW window')
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        cleanup = lambda: glfw.destroy_window(window)  # noqa: E731
    else:
        try:
            ctx = moderngl.create_standalone_context(backend='egl')
        except Exception as e:
            pytest.skip(f'no DISPLAY and EGL context creation failed: {e}')
        cleanup = ctx.release
    renderer = ctx.info.get('GL_RENDERER', '')
    if 'nvidia' not in renderer.lower():
        cleanup()
        pytest.skip(f'GL context is on a non-NVIDIA GPU: {renderer}')
    return ctx, cleanup


@pytest.fixture
def gl_context():
    """Headless-capable OpenGL context on an NVIDIA GPU (moderngl)."""
    ctx, cleanup = make_gl_test_context()
    try:
        yield ctx
    finally:
        cleanup()
