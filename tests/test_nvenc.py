"""Tests for NVENC module.

Note: Most tests require an NVIDIA GPU with NVENC support.
Tests that don't require GPU are marked accordingly.
"""

import pytest

# Check if NVENC GPU tests can run (requires GLX context, not EGL)
def _nvenc_available():
    """Check if NVENC is available (library + GLFW + moderngl importable).

    This only checks prerequisites. The actual GL renderer check happens
    in NvencEncoder._init_encoder() which raises NvencError (not segfault)
    if the GL context is on a non-NVIDIA GPU.

    We intentionally avoid creating a GL context here because GLFW state
    leaks across tests and can cause hangs.
    """
    try:
        import ctypes
        ctypes.CDLL('libnvidia-encode.so.1')
        import glfw  # noqa: F401
        import moderngl  # noqa: F401
        from framepump.nvenc import NvencEncoder  # noqa: F401
        return True
    except Exception:
        return False

NVENC_AVAILABLE = _nvenc_available()
_glfw_initialized = False


class TestNvencImports:
    """Test that nvenc module imports work (no GPU required)."""

    def test_import_exceptions(self):
        """Test that exception classes can be imported."""
        from framepump.nvenc import (
            NvencError,
            NvencNotAvailable,
            TextureFormatError,
            EncoderNotInitialized,
        )
        assert issubclass(NvencNotAvailable, NvencError)
        assert issubclass(TextureFormatError, NvencError)
        assert issubclass(EncoderNotInitialized, NvencError)

    def test_import_encoder_class(self):
        """Test that NvencEncoder class can be imported."""
        from framepump.nvenc import NvencEncoder
        assert NvencEncoder is not None

    def test_import_bindings(self):
        """Test that bindings module can be imported."""
        from framepump.nvenc import bindings
        assert hasattr(bindings, 'NvencAPI')
        assert hasattr(bindings, 'GUID')
        assert hasattr(bindings, 'NV_ENC_SUCCESS')



class TestGLVideoWriterImport:
    """Test GLVideoWriter import (no GPU required)."""

    def test_import_glvideowriter(self):
        """Test that GLVideoWriter can be imported from main package."""
        from framepump import GLVideoWriter
        assert GLVideoWriter is not None

    def test_glvideowriter_has_expected_methods(self):
        """Test that GLVideoWriter has expected API."""
        from framepump import GLVideoWriter
        assert hasattr(GLVideoWriter, 'start_sequence')
        assert hasattr(GLVideoWriter, 'end_sequence')
        assert hasattr(GLVideoWriter, 'append_data')
        assert hasattr(GLVideoWriter, 'close')
        assert hasattr(GLVideoWriter, '__enter__')
        assert hasattr(GLVideoWriter, '__exit__')


@pytest.fixture
def gl_context():
    """Create a headless OpenGL context for GPU tests (GLX-based for NVENC)."""
    global _glfw_initialized
    import glfw
    import moderngl

    if not _glfw_initialized:
        if not glfw.init():
            pytest.skip('Failed to initialize GLFW')
        _glfw_initialized = True

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, 'test', None, None)
    if not window:
        pytest.skip('Failed to create GLFW window')

    glfw.make_context_current(window)

    # Check if GL context is on an NVIDIA GPU (required for NVENC OpenGL path)
    import ctypes
    gl = ctypes.cdll.LoadLibrary('libGL.so.1')
    gl.glGetString.restype = ctypes.c_char_p
    renderer = (gl.glGetString(0x1F01) or b'').decode(errors='replace')
    if 'nvidia' not in renderer.lower():
        glfw.destroy_window(window)
        pytest.skip(f'GL context is on non-NVIDIA GPU: {renderer}')

    ctx = moderngl.create_context()

    yield ctx

    glfw.destroy_window(window)


@pytest.mark.skipif(
    not NVENC_AVAILABLE,
    reason='Requires NVIDIA GPU with NVENC support'
)
class TestNvencEncoder:
    """Tests that require actual NVENC hardware."""

    def test_encoder_init(self, gl_context):
        """Test that encoder initializes correctly."""
        from framepump.nvenc import NvencEncoder
        encoder = NvencEncoder(640, 480, fps=30)
        assert encoder is not None
        encoder.close()

    def test_encoder_context_manager(self, gl_context):
        """Test encoder as context manager."""
        from framepump.nvenc import NvencEncoder
        with NvencEncoder(640, 480, fps=30) as encoder:
            assert encoder is not None

    def test_encode_single_frame(self, gl_context):
        """Test encode + flush produces valid H.264 with correct structure."""
        from framepump.nvenc import NvencEncoder, EncodedPacket
        texture = gl_context.texture((640, 480), 4)  # RGBA

        with NvencEncoder(640, 480, fps=30) as encoder:
            packets = encoder.encode(texture)
            packets.extend(encoder.flush())

        assert len(packets) == 1
        pkt = packets[0]
        assert isinstance(pkt, EncodedPacket)
        assert isinstance(pkt.data, bytes)
        # H.264 NAL start code
        assert pkt.data[:4] == b'\x00\x00\x00\x01'
        # Single frame must be a keyframe
        assert pkt.is_keyframe
        assert pkt.pts == 0
        assert pkt.dts == 0

    def test_encode_multiple_frames(self, gl_context):
        """Test that N input frames produce exactly N output packets with valid timing."""
        from framepump.nvenc import NvencEncoder
        texture = gl_context.texture((640, 480), 4)
        n_frames = 30

        packets = []
        with NvencEncoder(640, 480, fps=30) as encoder:
            for _ in range(n_frames):
                packets.extend(encoder.encode(texture))
            packets.extend(encoder.flush())

        # Every input frame produces exactly one output packet
        assert len(packets) == n_frames

        # First packet must be a keyframe
        assert packets[0].is_keyframe

        # All packets have H.264 NAL start codes and non-empty data
        for pkt in packets:
            assert pkt.data[:4] == b'\x00\x00\x00\x01'
            assert len(pkt.data) > 4

        # PTS values cover all frame indices (may be reordered due to B-frames)
        pts_values = sorted(pkt.pts for pkt in packets)
        assert pts_values == list(range(n_frames))

        # DTS is monotonically non-decreasing (decode order)
        dts_values = [pkt.dts for pkt in packets]
        assert dts_values == sorted(dts_values)

    def test_encoder_different_resolutions(self, gl_context):
        """Test that larger resolutions produce more encoded bytes."""
        from framepump.nvenc import NvencEncoder

        resolutions = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        sizes = []
        for width, height in resolutions:
            texture = gl_context.texture((width, height), 4)
            with NvencEncoder(width, height, fps=30) as encoder:
                packets = encoder.encode(texture)
                packets.extend(encoder.flush())
            assert len(packets) == 1, f'Expected 1 packet at {width}x{height}'
            assert packets[0].is_keyframe
            sizes.append(len(packets[0].data))

        # Each resolution should produce strictly more bytes than the previous
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], (
                f'{resolutions[i]} ({sizes[i]}B) should be larger than '
                f'{resolutions[i - 1]} ({sizes[i - 1]}B)'
            )

    def test_encoder_different_fps(self, gl_context):
        """Test that different fps values all produce valid encoded output."""
        from framepump.nvenc import NvencEncoder

        fps_values = [24, 25, 30, 60, 30000 / 1001, 24000 / 1001]
        texture = gl_context.texture((320, 240), 4)

        for fps in fps_values:
            with NvencEncoder(320, 240, fps=fps) as encoder:
                packets = encoder.encode(texture)
                packets.extend(encoder.flush())
            assert len(packets) == 1, f'Expected 1 packet at {fps} fps'
            assert packets[0].data[:4] == b'\x00\x00\x00\x01', f'Bad NAL at {fps} fps'
            assert packets[0].is_keyframe, f'Single frame not keyframe at {fps} fps'


@pytest.mark.skipif(
    not NVENC_AVAILABLE,
    reason='Requires NVIDIA GPU with NVENC support'
)
class TestGLVideoWriter:
    """Integration tests for GLVideoWriter."""

    def test_glvideowriter_creates_video(self, gl_context, tmp_path):
        """Test that GLVideoWriter creates a valid video file."""
        from framepump import GLVideoWriter, VideoFrames

        output_path = tmp_path / 'test_output.mp4'
        texture = gl_context.texture((640, 480), 4)

        with GLVideoWriter() as writer:
            writer.start_sequence(str(output_path), fps=30)
            for _ in range(30):
                writer.append_data(texture)
            writer.end_sequence()

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify with VideoFrames
        frames = VideoFrames(str(output_path))
        assert tuple(frames.imshape[:2]) == (480, 640)

    def test_glvideowriter_fps_preserved(self, gl_context, tmp_path):
        """Test that GLVideoWriter preserves frame rate."""
        from framepump import GLVideoWriter, get_fps

        output_path = tmp_path / 'test_fps.mp4'
        texture = gl_context.texture((320, 240), 4)
        target_fps = 30000 / 1001  # 29.97

        with GLVideoWriter() as writer:
            writer.start_sequence(str(output_path), fps=target_fps)
            for _ in range(30):
                writer.append_data(texture)
            writer.end_sequence()

        actual_fps = get_fps(str(output_path))
        assert abs(actual_fps - target_fps) / target_fps < 0.001  # Within 0.1%

    def test_glvideowriter_multiple_sequences(self, gl_context, tmp_path):
        """Test writing multiple video sequences."""
        from framepump import GLVideoWriter

        texture = gl_context.texture((320, 240), 4)

        with GLVideoWriter() as writer:
            for i in range(3):
                output_path = tmp_path / f'sequence_{i}.mp4'
                writer.start_sequence(str(output_path), fps=30)
                for _ in range(10):
                    writer.append_data(texture)
                writer.end_sequence()
                assert output_path.exists()
