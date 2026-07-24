"""Tests for H.264 passthrough muxing shared by the GL and CUDA JPEG writers.

The muxer unit tests run CPU-only on synthetic x264 Annex-B packets shaped
like the NVENC encoders' output, so container policy (timestamps, movflags
scoping, avi warning, temp-file lifecycle, audio interleaving) is covered
without a GPU. The writer integration tests exercise the same paths through
GLVideoWriter and JpegVideoWriterCUDA on the GPU.
"""

import io
from fractions import Fraction

import numpy as np
import pytest


def expected_gray(i):
    return (i * 37) % 200 + 20


def make_h264_packets(n, w=64, h=64, fps=30, bframes=2):
    """Synthesize Annex-B H.264 packets shaped like the NVENC EncodedPacket.

    pts is the display-order frame index and dts a 0-based decode-order
    counter, exactly like NvencEncodeSession output. Each frame is a flat
    gray image encoding its index, so decoded content identifies frames.
    """
    import av

    codec = av.CodecContext.create('libx264', 'w')
    codec.width = w
    codec.height = h
    codec.pix_fmt = 'yuv420p'
    codec.framerate = Fraction(fps)
    codec.time_base = Fraction(1, fps)
    codec.options = {'bf': str(bframes), 'g': '250', 'x264-params': 'annexb=1:repeat-headers=1'}
    raw = []
    for i in range(n):
        arr = np.full((h, w, 3), expected_gray(i), np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format='rgb24').reformat(format='yuv420p')
        frame.pts = i
        raw.extend(codec.encode(frame))
    raw.extend(codec.encode(None))

    from framepump.nvenc import EncodedPacket

    return [
        EncodedPacket(data=bytes(pkt), pts=pkt.pts, dts=i, is_keyframe=pkt.is_keyframe)
        for i, pkt in enumerate(raw)
    ]


def make_muxer(output, fps=30, bframes=2, **kwargs):
    from framepump._h264_mux import H264PassthroughMuxer

    return H264PassthroughMuxer(
        output, fps=Fraction(fps), width=64, height=64, bframes=bframes, **kwargs
    )


def check_decoded(path_or_file, n_expected, tol=6):
    from framepump import VideoFrames

    frames = list(VideoFrames(path_or_file))
    assert len(frames) == n_expected
    for i, frame in enumerate(frames):
        med = float(np.median(frame))
        exp = expected_gray(i)
        assert abs(med - exp) <= tol, f'frame {i}: median {med:.0f}, expected {exp}'


def demux_packet_times(path):
    """Return [(stream_type, dts_seconds)] in stored file order."""
    import av

    out = []
    with av.open(str(path)) as container:
        for pkt in container.demux():
            if pkt.dts is None:
                continue
            out.append((pkt.stream.type, float(pkt.dts * pkt.stream.time_base)))
    return out


class TestMuxerUnit:
    """CPU-only muxer tests on synthetic packets."""

    @pytest.mark.parametrize('fmt', ['mp4', 'mkv'])
    @pytest.mark.parametrize('bframes', [0, 2])
    def test_roundtrip(self, tmp_path, fmt, bframes):
        n = 30
        out = tmp_path / f'out_b{bframes}.{fmt}'
        muxer = make_muxer(str(out), bframes=bframes)
        for pkt in make_h264_packets(n, bframes=bframes):
            muxer.mux(pkt)
        muxer.close()
        assert out.exists()
        assert not any(p.name.startswith('out') and 'tmp' in p.suffix for p in tmp_path.iterdir())
        check_decoded(str(out), n)

    def test_avi_bframes_warns(self, tmp_path):
        with pytest.warns(RuntimeWarning, match='avi'):
            muxer = make_muxer(str(tmp_path / 'out.avi'), bframes=2)
        for pkt in make_h264_packets(10, bframes=2):
            muxer.mux(pkt)
        muxer.close()
        check_decoded(str(tmp_path / 'out.avi'), 10)

    @pytest.mark.parametrize('fmt', ['avi', 'mp4', 'mkv'])
    def test_no_warning_without_bframes(self, tmp_path, fmt, recwarn):
        muxer = make_muxer(str(tmp_path / f'out.{fmt}'), bframes=0)
        for pkt in make_h264_packets(10, bframes=0):
            muxer.mux(pkt)
        muxer.close()
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_zero_packets_leaves_no_file(self, tmp_path):
        out = tmp_path / 'out.mp4'
        muxer = make_muxer(str(out))
        muxer.close()
        assert list(tmp_path.iterdir()) == []

    def test_error_during_close_cleans_temp(self, tmp_path):
        out = tmp_path / 'out.mp4'
        muxer = make_muxer(str(out), audio_source_path='tests/data/with_audio.mp4')
        for pkt in make_h264_packets(10):
            muxer.mux(pkt)

        class FailingContainer:
            def __init__(self, real):
                self._real = real

            def mux(self, pkt):
                raise OSError('simulated mux failure')

            def close(self):
                self._real.close()

        muxer._output_container = FailingContainer(muxer._output_container)
        with pytest.raises(OSError, match='simulated'):
            muxer.close()
        assert list(tmp_path.iterdir()) == []

    def test_abort_cleans_temp(self, tmp_path):
        out = tmp_path / 'out.mp4'
        muxer = make_muxer(str(out))
        for pkt in make_h264_packets(10):
            muxer.mux(pkt)
        muxer.abort()
        assert list(tmp_path.iterdir()) == []

    def test_filelike_requires_format(self):
        with pytest.raises(ValueError, match='format'):
            make_muxer(io.BytesIO())

    def test_filelike_output(self):
        buffer = io.BytesIO()
        muxer = make_muxer(buffer, format='mp4')
        for pkt in make_h264_packets(20):
            muxer.mux(pkt)
        muxer.close()
        buffer.seek(0)
        check_decoded(buffer, 20)

    def test_audio_interleaved_by_submit_order(self, tmp_path):
        n = 60  # 2 s of video at 30 fps, matching with_audio.mp4's 2 s audio
        out = tmp_path / 'out.mp4'
        muxer = make_muxer(str(out), audio_source_path='tests/data/with_audio.mp4')
        for pkt in make_h264_packets(n):
            muxer.mux(pkt)
        muxer.close()

        from framepump import has_audio

        assert has_audio(str(out))
        check_decoded(str(out), n)

        times = demux_packet_times(out)
        last_video_pos = max(i for i, (kind, _) in enumerate(times) if kind == 'video')
        interleaved = [
            (i, t) for i, (kind, t) in enumerate(times) if kind == 'audio' and i < last_video_pos
        ]
        assert len(interleaved) >= 10, 'audio must be interleaved, not appended at the end'
        for pos, audio_time in interleaved:
            next_video = next(t for kind, t in times[pos + 1 :] if kind == 'video')
            assert audio_time <= next_video + 0.5
            assert audio_time >= next_video - 0.5


def _gpu_available():
    """Library-presence probe only: no CUDA/GL initialization at collection."""
    try:
        import ctypes
        import importlib.util

        ctypes.CDLL('libnvidia-encode.so.1')
        ctypes.CDLL('libnvjpeg.so')
        return importlib.util.find_spec('cuda.bindings') is not None
    except Exception:
        return False


def _nvenc_gl_available():
    try:
        import ctypes

        ctypes.CDLL('libnvidia-encode.so.1')
        import glfw  # noqa: F401
        import moderngl  # noqa: F401

        return True
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _gpu_available(), reason='Requires NVIDIA GPU')
gl_only = pytest.mark.skipif(not _nvenc_gl_available(), reason='Requires NVENC + GL')

_glfw_initialized = False


@pytest.fixture
def gl_context():
    """Headless GLX context on the NVIDIA GPU (PRIME offload set in conftest)."""
    global _glfw_initialized
    import glfw
    import moderngl

    if not _glfw_initialized:
        if not glfw.init():
            pytest.skip('Failed to initialize GLFW')
        _glfw_initialized = True

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(320, 240, 'test', None, None)
    if not window:
        pytest.skip('Failed to create GLFW window')
    glfw.make_context_current(window)

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


def write_gl_video(gl_context, out, n=30, bframes=2, audio_source_path=None):
    from framepump import EncoderConfig, GLVideoWriter

    w, h = 320, 240
    texture = gl_context.texture((w, h), 4)
    with GLVideoWriter(encoder_config=EncoderConfig(bframes=bframes)) as writer:
        writer.start_sequence(str(out), fps=30, audio_source_path=audio_source_path)
        for i in range(n):
            arr = np.full((h, w, 4), 255, np.uint8)
            arr[..., :3] = expected_gray(i)
            texture.write(arr.tobytes())
            gl_context.finish()
            writer.append_data(texture)
        writer.end_sequence()


def make_jpeg(idx, w=320, h=240):
    from PIL import Image

    buf = io.BytesIO()
    arr = np.full((h, w, 3), expected_gray(idx), np.uint8)
    Image.fromarray(arr).save(buf, 'JPEG', quality=95, subsampling=2)
    return buf.getvalue()


def write_jpeg_video(out, n=30, bframes=2, audio_source_path=None):
    from framepump import EncoderConfig, JpegVideoWriterCUDA

    with JpegVideoWriterCUDA(encoder_config=EncoderConfig(bframes=bframes)) as writer:
        writer.start_sequence(str(out), fps=30, audio_source_path=audio_source_path)
        for i in range(n):
            writer.append_data(make_jpeg(i))
        writer.end_sequence()


class TestWriterIntegration:
    """GPU integration: both writers through the shared muxer."""

    @gl_only
    def test_gl_mkv_roundtrip(self, gl_context, tmp_path):
        out = tmp_path / 'out.mkv'
        write_gl_video(gl_context, out, n=30, bframes=2)
        check_decoded(str(out), 30, tol=15)

    @gpu_only
    def test_jpeg_mkv_roundtrip(self, tmp_path):
        out = tmp_path / 'out.mkv'
        write_jpeg_video(out, n=30, bframes=2)
        check_decoded(str(out), 30)

    @gpu_only
    def test_jpeg_avi_bframes_warns(self, tmp_path):
        out = tmp_path / 'out.avi'
        with pytest.warns(RuntimeWarning, match='avi'):
            write_jpeg_video(out, n=10, bframes=2)
        check_decoded(str(out), 10)

    @gl_only
    def test_gl_audio(self, gl_context, tmp_path):
        from framepump import has_audio

        out = tmp_path / 'out.mp4'
        write_gl_video(gl_context, out, n=60, audio_source_path='tests/data/with_audio.mp4')
        assert has_audio(str(out))
        check_decoded(str(out), 60, tol=15)

    @gpu_only
    def test_jpeg_audio(self, tmp_path):
        from framepump import has_audio

        out = tmp_path / 'out.mp4'
        write_jpeg_video(out, n=60, audio_source_path='tests/data/with_audio.mp4')
        assert has_audio(str(out))
        check_decoded(str(out), 60)

    @gl_only
    def test_gl_mux_error_leaves_no_files(self, gl_context, tmp_path, monkeypatch):
        from framepump import EncoderConfig, GLVideoWriter
        from framepump._h264_mux import H264PassthroughMuxer

        out = tmp_path / 'out.mp4'
        texture = gl_context.texture((320, 240), 4)
        writer = GLVideoWriter(encoder_config=EncoderConfig(bframes=2))
        writer.start_sequence(str(out), fps=30)
        # Write fewer frames than the reorder depth so that the packets only
        # emerge from the encoder flush at end_sequence time.
        for i in range(2):
            texture.write(np.full((240, 320, 4), 128, np.uint8).tobytes())
            gl_context.finish()
            writer.append_data(texture)

        def boom(self, pkt):
            raise OSError('simulated mux failure')

        monkeypatch.setattr(H264PassthroughMuxer, 'mux', boom)
        with pytest.raises(OSError, match='simulated'):
            writer.end_sequence()
        assert list(tmp_path.iterdir()) == []

    @gpu_only
    def test_jpeg_mux_error_leaves_no_files(self, tmp_path, monkeypatch):
        from framepump import EncoderConfig, JpegVideoWriterCUDA
        from framepump._h264_mux import H264PassthroughMuxer

        out = tmp_path / 'out.mp4'
        writer = JpegVideoWriterCUDA(encoder_config=EncoderConfig(bframes=2))
        writer.start_sequence(str(out), fps=30)
        for i in range(10):
            writer.append_data(make_jpeg(i))

        def boom(self, pkt):
            raise OSError('simulated mux failure')

        monkeypatch.setattr(H264PassthroughMuxer, 'mux', boom)
        with pytest.raises(OSError, match='simulated'):
            writer.end_sequence()
        assert list(tmp_path.iterdir()) == []


class TestZeroFrames:
    """Ending a sequence without frames leaves no file and raises no error."""

    def test_gl_zero_frames(self, tmp_path):
        from framepump import GLVideoWriter

        out = tmp_path / 'out.mp4'
        with GLVideoWriter() as writer:
            writer.start_sequence(str(out), fps=30)
            writer.end_sequence()
        assert list(tmp_path.iterdir()) == []

    @gpu_only
    def test_jpeg_zero_frames(self, tmp_path):
        from framepump import JpegVideoWriterCUDA

        out = tmp_path / 'out.mp4'
        with JpegVideoWriterCUDA() as writer:
            writer.start_sequence(str(out), fps=30)
            writer.end_sequence()
        assert list(tmp_path.iterdir()) == []


class TestMuxerAudioValidation:
    def test_audio_source_without_audio_stream_raises(self, tmp_path):
        from framepump import NoAudioStreamError, VideoWriter
        from framepump._h264_mux import H264PassthroughMuxer

        silent = tmp_path / 'silent.mp4'
        with VideoWriter(str(silent), fps=30) as writer:
            for _ in range(3):
                writer.append_data(np.zeros((64, 64, 3), np.uint8))

        out = tmp_path / 'out.mp4'
        with pytest.raises(NoAudioStreamError):
            H264PassthroughMuxer(
                str(out),
                fps=Fraction(30),
                width=64,
                height=64,
                bframes=0,
                audio_source_path=str(silent),
            )
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))


class TestTrimVideo:
    @staticmethod
    def _make_video(tmp_path, n_frames=24, fps=12):
        from framepump import VideoWriter

        path = tmp_path / 'src.mp4'
        with VideoWriter(str(path), fps=fps) as writer:
            for i in range(n_frames):
                writer.append_data(np.full((64, 64, 3), (i * 9) % 255, np.uint8))
        return str(path), n_frames, fps

    def test_trim_to_full_duration_keeps_last_frame(self, tmp_path):
        from framepump import get_duration, num_frames, trim_video

        src, n_frames, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        trim_video(src, str(out), 0.0, get_duration(src), gpu=False)
        assert num_frames(str(out), exact=True) == n_frames

    def test_trim_past_duration_keeps_last_frame(self, tmp_path):
        from framepump import num_frames, trim_video

        src, n_frames, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        trim_video(src, str(out), 0.0, 1e9, gpu=False)
        assert num_frames(str(out), exact=True) == n_frames

    def test_trim_mid_range_frame_count(self, tmp_path):
        from framepump import num_frames, trim_video

        src, _, fps = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        trim_video(src, str(out), 0.5, 1.5, gpu=False)  # 1 second at 12 fps
        assert num_frames(str(out), exact=True) == fps

    def test_trim_empty_range_raises(self, tmp_path):
        from framepump import trim_video

        src, _, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        with pytest.raises(ValueError, match='contains no frames'):
            trim_video(src, str(out), 0.5, 0.5, gpu=False)
        assert not out.exists()

    def test_trim_end_before_start_raises(self, tmp_path):
        from framepump import trim_video

        src, _, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        with pytest.raises(ValueError, match='contains no frames'):
            trim_video(src, str(out), 1.0, 0.5, gpu=False)
        assert not out.exists()

    def test_trim_start_past_end_raises(self, tmp_path):
        from framepump import trim_video

        src, _, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'
        with pytest.raises(ValueError, match='past the last frame'):
            trim_video(src, str(out), 100.0, 200.0, gpu=False)
        assert not out.exists()

    def test_trim_error_leaves_no_output_file(self, tmp_path, monkeypatch):
        from framepump import trim_video
        from framepump import video_writing as vw

        src, _, _ = self._make_video(tmp_path)
        out = tmp_path / 'trimmed.mp4'

        def boom(*args, **kwargs):
            raise RuntimeError('simulated failure')

        monkeypatch.setattr(vw, '_trim_video_to_path', boom)
        with pytest.raises(RuntimeError, match='simulated failure'):
            trim_video(src, str(out), 0.0, 1.0)
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))


class TestVideoAudioMuxErrors:
    def test_mux_error_leaves_no_output_file(self, tmp_path):
        from framepump import NoAudioStreamError, VideoWriter, video_audio_mux

        silent = tmp_path / 'silent.mp4'
        with VideoWriter(str(silent), fps=30) as writer:
            for _ in range(3):
                writer.append_data(np.zeros((64, 64, 3), np.uint8))

        out = tmp_path / 'muxed.mp4'
        with pytest.raises(NoAudioStreamError):
            video_audio_mux(str(silent), str(silent), str(out))
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))


class TestTrimVideoNvencMinimum:
    """NVENC rejects frames below ~145x49; auto-detection must not pick it
    for small videos, and an explicit gpu=True must fail with a clear error."""

    def test_small_video_auto_gpu_falls_back_to_cpu(self, tmp_path):
        from framepump import num_frames, trim_video

        src, n_frames, _ = TestTrimVideo._make_video(tmp_path)  # 64x64
        out = tmp_path / 'trimmed.mp4'
        trim_video(src, str(out), 0.0, 1.0)  # gpu=None auto-detect
        assert num_frames(str(out), exact=True) > 0

    def test_small_video_explicit_gpu_raises_clear_error(self, tmp_path):
        from framepump import trim_video

        src, _, _ = TestTrimVideo._make_video(tmp_path)
        with pytest.raises(ValueError, match='NVENC.*minimum'):
            trim_video(src, str(tmp_path / 'x.mp4'), 0.0, 1.0, gpu=True)
