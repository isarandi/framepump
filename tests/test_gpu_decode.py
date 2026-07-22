"""Tests for NVDEC-accelerated decoding via VideoFrames(gpu=True).

With allow_software_fallback disabled in the reader, silent CPU decoding is
structurally impossible: if NVDEC cannot handle the stream, opening fails
loudly. These tests therefore focus on parity with CPU decoding and on the
error surface.
"""

import numpy as np
import pytest

from framepump import FramePumpError, VideoFrames, VideoWriter


@pytest.fixture(scope='module')
def h264_video(tmp_path_factory):
    path = tmp_path_factory.mktemp('gpu_decode') / 'src.mp4'
    rng = np.random.default_rng(0)
    with VideoWriter(str(path), fps=30) as writer:
        base = rng.integers(0, 255, (240, 320, 3), np.uint8)
        for i in range(30):
            writer.append_data(np.roll(base, i * 7, axis=1))
    return str(path)


def _gpu_decode_available(path):
    try:
        VideoFrames(path, gpu=True)[0]
        return True
    except Exception:
        return False


@pytest.fixture(scope='module')
def gpu_or_skip(h264_video):
    if not _gpu_decode_available(h264_video):
        pytest.skip('NVDEC GPU decoding not available')


class TestGpuCpuParity:
    def test_frames_bit_exact(self, h264_video, gpu_or_skip):
        cpu = [f.copy() for f in VideoFrames(h264_video)]
        gpu = [f.copy() for f in VideoFrames(h264_video, gpu=True)]
        assert len(cpu) == len(gpu)
        for i, (a, b) in enumerate(zip(cpu, gpu)):
            assert np.array_equal(a, b), f'frame {i}'

    def test_indexed_access_bit_exact(self, h264_video, gpu_or_skip):
        cpu = VideoFrames(h264_video)
        gpu = VideoFrames(h264_video, gpu=True)
        for i in (0, 7, 15, len(cpu) - 1):
            assert np.array_equal(cpu[i], gpu[i]), f'frame {i}'

    def test_slice_and_reverse_bit_exact(self, h264_video, gpu_or_skip):
        cpu = VideoFrames(h264_video)
        gpu = VideoFrames(h264_video, gpu=True)
        for a, b in zip(cpu[5:20:3], gpu[5:20:3]):
            assert np.array_equal(a, b)
        for a, b in zip(cpu[20:10:-2], gpu[20:10:-2]):
            assert np.array_equal(a, b)

    def test_resized_bit_exact(self, h264_video, gpu_or_skip):
        cpu = VideoFrames(h264_video).resized((120, 160))
        gpu = VideoFrames(h264_video, gpu=True).resized((120, 160))
        assert np.array_equal(cpu[3], gpu[3])

    def test_uint16_bit_exact(self, h264_video, gpu_or_skip):
        cpu = VideoFrames(h264_video, dtype=np.uint16)
        gpu = VideoFrames(h264_video, dtype=np.uint16, gpu=True)
        assert np.array_equal(cpu[3], gpu[3])
        assert gpu[3].dtype == np.uint16


class TestGpuErrorSurface:
    def test_unsupported_codec_raises_with_hint(self, tmp_path, gpu_or_skip):
        import subprocess

        path = tmp_path / 'lossless.mkv'
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-v',
                'error',
                '-f',
                'lavfi',
                '-i',
                'testsrc2=duration=1:size=128x96:rate=10',
                '-c:v',
                'ffv1',
                str(path),
            ],
            check=True,
        )
        with pytest.raises(FramePumpError, match='gpu=False'):
            VideoFrames(str(path), gpu=True)[0]

    def test_invalid_device_ordinal_raises(self, h264_video, gpu_or_skip):
        with pytest.raises(ValueError, match='device'):
            VideoFrames(h264_video, gpu=99)[0]

    def test_file_like_rejected(self, h264_video):
        import io

        with open(h264_video, 'rb') as f:
            data = io.BytesIO(f.read())
        with pytest.raises(ValueError, match='file-like'):
            VideoFrames(data, gpu=True)
