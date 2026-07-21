"""FFV1 depth video writing and grayscale reading.

The core contract is bit-exact losslessness: uint16 depth frames written by
DepthVideoWriter and read back with VideoFrames(gray=True, dtype=np.uint16)
must be identical, through every access path (streamed, indexed, sliced,
reversed).
"""

import numpy as np
import pytest

from framepump import DepthVideoWriter, VideoFrames, VideoWriter


@pytest.fixture
def depth_video(tmp_path):
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 65536, (47, 63), dtype=np.uint16) for _ in range(12)]
    path = tmp_path / 'depth.mkv'
    with DepthVideoWriter(str(path), fps=5) as writer:
        for f in frames:
            writer.append_data(f)
    return str(path), frames


class TestDepthRoundTrip:
    def test_lossless_streamed(self, depth_video):
        path, frames = depth_video
        vf = VideoFrames(path, dtype=np.uint16, gray=True)
        got = [f for f in vf]
        assert vf._lazy.index is None, 'gray streaming must stay lazy'
        assert len(got) == len(frames)
        for a, b in zip(got, frames):
            assert a.shape == b.shape and a.dtype == np.uint16
            assert np.array_equal(a, b)

    def test_lossless_indexed(self, depth_video):
        path, frames = depth_video
        vf = VideoFrames(path, dtype=np.uint16, gray=True)
        for i in (0, 5, len(frames) - 1, -1):
            assert np.array_equal(vf[i], frames[i])

    def test_lossless_sliced_and_reversed(self, depth_video):
        path, frames = depth_video
        vf = VideoFrames(path, dtype=np.uint16, gray=True)
        for sl in (slice(3, 9), slice(None, None, 2), slice(None, None, -1), slice(9, 2, -3)):
            got = [f for f in vf[sl]]
            want = frames[sl]
            assert len(got) == len(want), sl
            for a, b in zip(got, want):
                assert np.array_equal(a, b), sl

    def test_odd_dimensions_supported(self, depth_video):
        # 47x63 in the fixture: FFV1 grayscale has no even-dimension
        # requirement, unlike the H.264 yuv420p path
        path, frames = depth_video
        assert frames[0].shape == (47, 63)

    def test_gray_float_conversion(self, depth_video):
        path, frames = depth_video
        f = next(iter(VideoFrames(path, dtype=np.float32, gray=True)))
        assert f.dtype == np.float32 and f.ndim == 2
        assert np.allclose(f, frames[0].astype(np.float64) / 65535.0, atol=1e-6)


class TestDepthWriterValidation:
    def test_float_frames_rejected(self, tmp_path):
        with pytest.raises((RuntimeError, ValueError), match='uint16'):
            with DepthVideoWriter(str(tmp_path / 'x.mkv'), fps=5) as writer:
                writer.append_data(np.zeros((32, 32), np.float32))
        assert not (tmp_path / 'x.mkv').exists()

    def test_rgb_frames_rejected(self, tmp_path):
        with pytest.raises((RuntimeError, ValueError), match='height, width'):
            with DepthVideoWriter(str(tmp_path / 'x.mkv'), fps=5) as writer:
                writer.append_data(np.zeros((32, 32, 3), np.uint16))
        assert not (tmp_path / 'x.mkv').exists()

    def test_mp4_container_rejected(self, tmp_path):
        with pytest.raises((RuntimeError, ValueError), match='MKV'):
            with DepthVideoWriter(str(tmp_path / 'x.mp4'), fps=5) as writer:
                writer.append_data(np.zeros((32, 32), np.uint16))
        assert not (tmp_path / 'x.mp4').exists()


class TestGrayReadingOfColorVideo:
    def test_gray_uint8_from_rgb_source(self):
        vf = VideoFrames('tests/data/short.mp4', gray=True)
        f = vf[0]
        assert f.ndim == 2 and f.dtype == np.uint8
        assert f.shape == (720, 1280)

    def test_gray_rejects_gpu(self):
        with pytest.raises(ValueError, match='gray'):
            VideoFrames('tests/data/short.mp4', gray=True, gpu=True)


class TestMkvContainerAlias:
    def test_videowriter_writes_mkv(self, tmp_path):
        # .mkv requires the 'matroska' muxer name; the suffix alone is not a
        # valid libav format name
        path = tmp_path / 'out.mkv'
        with VideoWriter(str(path), fps=10) as writer:
            for i in range(5):
                writer.append_data(np.full((64, 64, 3), i * 40, np.uint8))
        assert len(VideoFrames(str(path))) == 5
