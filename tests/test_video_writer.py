"""Tests for VideoWriter class."""

import numpy as np
import pytest
import os

from framepump import VideoWriter, VideoFrames, get_fps, get_duration


class TestVideoWriterBasic:
    """Basic VideoWriter functionality tests."""

    def test_create_video(self, tmp_path):
        """Test creating a video file."""
        video_path = tmp_path / 'output.mp4'
        fps = 30
        n_frames = 10

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                writer.append_data(frame)

        assert video_path.exists()

    def test_video_readable_after_write(self, tmp_path):
        """Test that written video can be read back."""
        video_path = tmp_path / 'output.mp4'
        fps = 30
        n_frames = 15
        shape = (64, 64)

        with VideoWriter(str(video_path), fps=fps) as writer:
            for i in range(n_frames):
                frame = np.full((*shape, 3), i * 16, dtype=np.uint8)
                writer.append_data(frame)

        # Read back and verify
        frames = VideoFrames(str(video_path))
        assert len(frames) == n_frames
        assert tuple(frames.imshape) == shape

    def test_context_manager(self, tmp_path):
        """Test VideoWriter as context manager."""
        video_path = tmp_path / 'output.mp4'

        writer = VideoWriter(str(video_path), fps=30)
        with writer:
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            writer.append_data(frame)

        # After context exit, video should be finalized
        assert video_path.exists()

    def test_requires_fps_with_path(self):
        """Test that fps is required when video_path is provided."""
        with pytest.raises(ValueError):
            VideoWriter('test.mp4')  # No fps provided


class TestVideoWriterSequences:
    """Test multiple sequence writing."""

    def test_multiple_sequences(self, tmp_path):
        """Test writing multiple video sequences."""
        video1 = tmp_path / 'video1.mp4'
        video2 = tmp_path / 'video2.mp4'

        writer = VideoWriter()
        try:
            # First sequence
            writer.start_sequence(str(video1), fps=30)
            for _ in range(5):
                writer.append_data(np.zeros((50, 50, 3), dtype=np.uint8))
            writer.end_sequence()

            # Second sequence
            writer.start_sequence(str(video2), fps=24)
            for _ in range(5):
                writer.append_data(np.zeros((60, 60, 3), dtype=np.uint8))
            writer.end_sequence()
        finally:
            writer.close()

        assert video1.exists()
        assert video2.exists()

    def test_is_active(self, tmp_path):
        """Test accepts_new_frames property."""
        video_path = tmp_path / 'output.mp4'

        writer = VideoWriter()
        assert not writer.accepts_new_frames

        writer.start_sequence(str(video_path), fps=30)
        assert writer.accepts_new_frames

        writer.append_data(np.zeros((50, 50, 3), dtype=np.uint8))
        writer.end_sequence()
        assert not writer.accepts_new_frames

        writer.close()

    def test_append_before_start_raises(self):
        """Test that appending before start_sequence raises error."""
        writer = VideoWriter()

        with pytest.raises(ValueError):
            writer.append_data(np.zeros((50, 50, 3), dtype=np.uint8))

        writer.close()

    def test_end_before_start_raises(self):
        """Test that end_sequence before start_sequence raises error."""
        writer = VideoWriter()

        with pytest.raises(ValueError):
            writer.end_sequence()

        writer.close()


class TestVideoWriterDtypes:
    """Test different frame dtypes."""

    def test_uint8_frames(self, tmp_path):
        """Test writing uint8 frames."""
        video_path = tmp_path / 'output.mp4'

        with VideoWriter(str(video_path), fps=30) as writer:
            frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)

        assert video_path.exists()

    def test_uint16_frames(self, tmp_path):
        """Test writing uint16 frames (10-bit encoding)."""
        video_path = tmp_path / 'output.mp4'

        with VideoWriter(str(video_path), fps=30) as writer:
            frame = np.random.randint(0, 65536, (64, 64, 3), dtype=np.uint16)
            writer.append_data(frame)

        assert video_path.exists()


class TestVideoWriterFractionalFps:
    """Test fractional FPS handling to prevent drift/skew."""

    def test_ntsc_29_97_fps(self, tmp_path):
        """Test NTSC 29.97 fps is preserved accurately."""
        video_path = tmp_path / 'ntsc.mp4'
        # NTSC is exactly 30000/1001
        fps = 30000 / 1001  # 29.97002997...
        n_frames = 100

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        read_fps = get_fps(str(video_path))
        # Should be within 0.01% to avoid drift over long videos
        assert read_fps == pytest.approx(fps, rel=1e-6)

    def test_film_23_976_fps(self, tmp_path):
        """Test film 23.976 fps is preserved accurately."""
        video_path = tmp_path / 'film.mp4'
        # Film pulldown is exactly 24000/1001
        fps = 24000 / 1001  # 23.976023976...
        n_frames = 100

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        read_fps = get_fps(str(video_path))
        assert read_fps == pytest.approx(fps, rel=1e-6)

    def test_ntsc_59_94_fps(self, tmp_path):
        """Test NTSC 59.94 fps is preserved accurately."""
        video_path = tmp_path / 'ntsc_60.mp4'
        fps = 60000 / 1001  # 59.94005994...
        n_frames = 60

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        read_fps = get_fps(str(video_path))
        assert read_fps == pytest.approx(fps, rel=1e-6)

    def test_fractional_fps_duration_accuracy(self, tmp_path):
        """Test that fractional fps doesn't cause duration drift."""
        video_path = tmp_path / 'duration_test.mp4'
        fps = 30000 / 1001
        n_frames = 1000  # Longer video to detect drift

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((32, 32, 3), dtype=np.uint8))

        expected_duration = n_frames / fps
        actual_duration = get_duration(str(video_path))
        # Duration should be accurate within 1 frame
        assert actual_duration == pytest.approx(expected_duration, abs=1/fps)

    def test_high_precision_fps(self, tmp_path):
        """Test that high precision fps values are handled."""
        video_path = tmp_path / 'precise.mp4'
        fps = 29.123456789
        n_frames = 50

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        read_fps = get_fps(str(video_path))
        # Allow slightly more tolerance for arbitrary fps values
        assert read_fps == pytest.approx(fps, rel=1e-6)

    def test_videoframes_fps_matches_writer_ntsc(self, tmp_path):
        """Test VideoFrames.fps matches the fps used in VideoWriter for NTSC."""
        video_path = tmp_path / 'ntsc_roundtrip.mp4'
        fps = 30000 / 1001  # 29.97
        n_frames = 50

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        frames = VideoFrames(str(video_path))
        assert frames.fps == pytest.approx(fps, rel=1e-6)

    def test_videoframes_fps_matches_writer_film(self, tmp_path):
        """Test VideoFrames.fps matches the fps used in VideoWriter for film."""
        video_path = tmp_path / 'film_roundtrip.mp4'
        fps = 24000 / 1001  # 23.976
        n_frames = 50

        with VideoWriter(str(video_path), fps=fps) as writer:
            for _ in range(n_frames):
                writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        frames = VideoFrames(str(video_path))
        assert frames.fps == pytest.approx(fps, rel=1e-6)

    def test_videoframes_fps_matches_writer_integer(self, tmp_path):
        """Test VideoFrames.fps matches for common integer fps values."""
        for fps in [24, 25, 30, 50, 60]:
            video_path = tmp_path / f'int_{fps}fps.mp4'
            n_frames = 30

            with VideoWriter(str(video_path), fps=fps) as writer:
                for _ in range(n_frames):
                    writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

            frames = VideoFrames(str(video_path))
            assert frames.fps == pytest.approx(fps, rel=1e-6), f"Failed for {fps} fps"


class TestVideoWriterIntegration:
    """Integration tests for read/write round-trip."""

    def test_roundtrip_preserves_frame_count(self, tmp_path):
        """Test that frame count is preserved in round-trip."""
        video_path = tmp_path / 'output.mp4'
        n_frames = 20
        fps = 30

        # Write
        with VideoWriter(str(video_path), fps=fps) as writer:
            for i in range(n_frames):
                frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
                writer.append_data(frame)

        # Read back
        frames = VideoFrames(str(video_path))
        assert len(frames) == n_frames

    def test_roundtrip_preserves_shape(self, tmp_path):
        """Test that frame shape is preserved in round-trip."""
        video_path = tmp_path / 'output.mp4'
        shape = (128, 96)

        with VideoWriter(str(video_path), fps=30) as writer:
            frame = np.zeros((*shape, 3), dtype=np.uint8)
            writer.append_data(frame)

        frames = VideoFrames(str(video_path))
        assert tuple(frames.imshape) == shape


def make_quadrant_frame_uint8(size=128):
    """Create a quadrant test pattern: red, green, blue, white."""
    half = size // 2
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:half, :half] = [255, 0, 0]      # top-left: red
    frame[:half, half:] = [0, 255, 0]      # top-right: green
    frame[half:, :half] = [0, 0, 255]      # bottom-left: blue
    frame[half:, half:] = [255, 255, 255]  # bottom-right: white
    return frame


def assert_quadrants_match(original, decoded, atol=5):
    """Check quadrant interiors match (avoid edges where H.264 blurs)."""
    size = original.shape[0]
    half = size // 2
    margin = size // 16  # Stay away from edges
    assert np.allclose(original[margin:half-margin, margin:half-margin],
                       decoded[margin:half-margin, margin:half-margin], atol=atol)
    assert np.allclose(original[margin:half-margin, half+margin:size-margin],
                       decoded[margin:half-margin, half+margin:size-margin], atol=atol)
    assert np.allclose(original[half+margin:size-margin, margin:half-margin],
                       decoded[half+margin:size-margin, margin:half-margin], atol=atol)
    assert np.allclose(original[half+margin:size-margin, half+margin:size-margin],
                       decoded[half+margin:size-margin, half+margin:size-margin], atol=atol)


class TestPixelRoundtrip:
    """Test that pixel values survive encode/decode within tolerance."""

    def test_uint8_cpu(self, tmp_path):
        """Test uint8 pixel roundtrip with CPU encoding."""
        video_path = tmp_path / 'uint8_cpu.mp4'
        frame = make_quadrant_frame_uint8()

        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30, gpu=False)
        for _ in range(5):
            writer.append_data(frame)
        writer.end_sequence()
        writer.close()

        decoded = list(VideoFrames(str(video_path)))
        assert_quadrants_match(frame, decoded[2], atol=5)

    def test_uint8_gpu(self, tmp_path):
        """Test uint8 pixel roundtrip with GPU encoding (NVENC)."""
        video_path = tmp_path / 'uint8_gpu.mp4'
        # NVENC requires minimum ~145x145, use larger frame
        frame = make_quadrant_frame_uint8(size=256)

        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30, gpu=True)
        for _ in range(5):
            writer.append_data(frame)
        writer.end_sequence()
        writer.close()

        if not video_path.exists() or video_path.stat().st_size == 0:
            pytest.skip('NVENC encoding failed (GPU not available or unsupported)')

        decoded = list(VideoFrames(str(video_path)))
        assert_quadrants_match(frame, decoded[2], atol=5)

    def test_uint16_roundtrip(self, tmp_path):
        """Test uint16 pixel values roundtrip correctly."""
        video_path = tmp_path / 'uint16.mp4'
        size = 128
        half = size // 2

        frame = np.zeros((size, size, 3), dtype=np.uint16)
        frame[:half, :half] = [65535, 0, 0]          # top-left: red
        frame[:half, half:] = [0, 65535, 0]          # top-right: green
        frame[half:, :half] = [0, 0, 65535]          # bottom-left: blue
        frame[half:, half:] = [65535, 65535, 65535]  # bottom-right: white

        with VideoWriter(str(video_path), fps=30) as writer:
            for _ in range(5):
                writer.append_data(frame)

        decoded = list(VideoFrames(str(video_path), dtype=np.uint16))
        assert_quadrants_match(frame, decoded[2], atol=300)

    def test_gl_texture_roundtrip(self, tmp_path):
        """Test GL texture to video roundtrip with GLVideoWriter."""
        pytest.importorskip('moderngl')
        import moderngl

        from framepump import GLVideoWriter
        from framepump.nvenc.exceptions import NvencError
        if GLVideoWriter is None:
            pytest.skip('GLVideoWriter not available (NVENC required)')

        video_path = tmp_path / 'gl.mp4'
        frame = make_quadrant_frame_uint8(size=256)  # NVENC needs larger frames

        # Create headless OpenGL context and texture
        ctx = moderngl.create_standalone_context()
        try:
            texture = ctx.texture((256, 256), 3, frame.tobytes())

            try:
                with GLVideoWriter(str(video_path), fps=30) as writer:
                    for _ in range(5):
                        writer.append_data(texture)
            except NvencError as e:
                pytest.skip(f'NVENC not available on current GL context: {e}')

            decoded = list(VideoFrames(str(video_path)))
            assert_quadrants_match(frame, decoded[2], atol=5)
        finally:
            ctx.release()

    def test_nvenc_small_frame_error(self, tmp_path):
        """Test that NVENC raises clear error for too-small frames."""
        video_path = tmp_path / 'small.mp4'
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        writer = VideoWriter()
        writer.start_sequence(str(video_path), fps=30, gpu=True)
        writer.append_data(frame)

        with pytest.raises(RuntimeError, match='NVENC frame size too small'):
            writer.end_sequence()

        writer.close()

    def test_gl_small_frame_error(self, tmp_path):
        """Test that GLVideoWriter raises clear error for too-small frames."""
        pytest.importorskip('moderngl')
        import moderngl
        from framepump import GLVideoWriter
        from framepump.nvenc.exceptions import NvencError

        if GLVideoWriter is None:
            pytest.skip('GLVideoWriter not available (NVENC required)')

        video_path = tmp_path / 'gl_small.mp4'

        ctx = moderngl.create_standalone_context()
        try:
            # 64x64 is too small for NVENC
            texture = ctx.texture((64, 64), 3, np.zeros((64, 64, 3), dtype=np.uint8).tobytes())

            writer = GLVideoWriter()
            try:
                writer.start_sequence(str(video_path), fps=30)
                writer.append_data(texture)
                writer.end_sequence()
            except NvencError as e:
                if 'non-NVIDIA GPU' in str(e):
                    pytest.skip(f'NVENC not available on current GL context: {e}')
                assert 'NV_ENC_ERR_INVALID_PARAM' in str(e)
            else:
                pytest.fail('Expected NvencError for too-small frame')
            finally:
                writer.close()
        finally:
            ctx.release()

    def test_gl_texture_roundtrip_glx(self, tmp_path):
        """Test GL texture roundtrip with GLX (display) context."""
        pytest.importorskip('moderngl')
        import moderngl
        import os

        from framepump import GLVideoWriter
        from framepump.nvenc.exceptions import NvencError
        if GLVideoWriter is None:
            pytest.skip('GLVideoWriter not available (NVENC required)')

        if not os.environ.get('DISPLAY'):
            pytest.skip('No DISPLAY available for GLX context')

        video_path = tmp_path / 'gl_glx.mp4'
        frame = make_quadrant_frame_uint8(size=256)

        # Use default backend which will be X11/GLX when DISPLAY is set
        ctx = moderngl.create_standalone_context()
        try:
            texture = ctx.texture((256, 256), 3, frame.tobytes())

            try:
                with GLVideoWriter(str(video_path), fps=30) as writer:
                    for _ in range(5):
                        writer.append_data(texture)
            except NvencError as e:
                pytest.skip(f'NVENC not available on current GL context: {e}')

            if not video_path.exists() or video_path.stat().st_size == 0:
                pytest.skip('NVENC encoding failed (GLX encoder not available)')

            decoded = list(VideoFrames(str(video_path)))
            assert_quadrants_match(frame, decoded[2], atol=5)
        finally:
            ctx.release()

    def test_gl_texture_roundtrip_egl(self, tmp_path):
        """Test GL texture roundtrip with EGL (headless) context."""
        pytest.importorskip('moderngl')
        import moderngl
        import os

        from framepump import GLVideoWriter
        from framepump.nvenc.exceptions import NvencError
        if GLVideoWriter is None:
            pytest.skip('GLVideoWriter not available (NVENC required)')

        # Force headless by temporarily unsetting DISPLAY
        old_display = os.environ.pop('DISPLAY', None)
        try:
            video_path = tmp_path / 'gl_egl.mp4'
            frame = make_quadrant_frame_uint8(size=256)

            try:
                # Explicitly request EGL backend for headless
                ctx = moderngl.create_standalone_context(backend='egl')
            except Exception as e:
                pytest.skip(f'EGL context creation failed: {e}')

            try:
                texture = ctx.texture((256, 256), 3, frame.tobytes())

                with GLVideoWriter(str(video_path), fps=30) as writer:
                    for _ in range(5):
                        writer.append_data(texture)

                if not video_path.exists() or video_path.stat().st_size == 0:
                    pytest.skip('NVENC encoding failed (CUDA encoder not available)')

                decoded = list(VideoFrames(str(video_path)))
                assert_quadrants_match(frame, decoded[2], atol=5)
            finally:
                ctx.release()
        except (ImportError, NvencError) as e:
            if 'cuda' in str(e).lower() or 'non-NVIDIA' in str(e) or isinstance(e, NvencError):
                pytest.skip(f'NVENC not available for EGL context: {e}')
            raise
        finally:
            if old_display is not None:
                os.environ['DISPLAY'] = old_display
