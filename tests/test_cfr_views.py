"""Tests for CFR frame arithmetic, windowed reads, view algebra, and sources.

Covers:
- CFR windowed reads (``cfr[a:b:s]``) content-compared against full decode
- ``len()`` vs. actual iteration count agreement in CFR mode
- Integer indexing (incl. negative) content-compared against iteration
- ``repeat_each_frame`` random access and composition with slicing and CFR
- Leading-gap videos (first PTS well past 0): no phantom output frames
- File-like sources: interleaved iterators must not corrupt each other
- Float output dtype conversion semantics
"""

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from framepump import VideoFrames

DATA_DIR = Path(__file__).parent / 'data'
CFR_VIDEOS = ['short.mp4', 'short.mkv', 'variable_fps.mp4', 'ntsc_film.mp4']
CFR_MODES = [True, 12.0, 60.0]
SMALL = (72, 128)  # decode at reduced size to keep the full-decode cache cheap


@pytest.fixture(scope='module')
def full_decode():
    """Cache of full decodes: (video_name, cfr_mode) -> list of frames."""
    cache = {}

    def get(name, cfr):
        key = (name, cfr)
        if key not in cache:
            frames = VideoFrames(str(DATA_DIR / name), constant_framerate=cfr).resized(SMALL)
            cache[key] = list(frames)
        return cache[key]

    return get


def _view(name, cfr):
    return VideoFrames(str(DATA_DIR / name), constant_framerate=cfr).resized(SMALL)


def _slices(n):
    return [
        slice(3, 6),
        slice(0, 5),
        slice(2, min(20, n), 2),
        slice(1, None, 2),
        slice(max(0, n - 3), n + 5),
        slice(n // 2, None),
        slice(0, None, 40),
        slice(1, None, 40),
    ]


class TestCfrWindowedReads:
    @pytest.mark.parametrize('name', CFR_VIDEOS)
    @pytest.mark.parametrize('cfr', CFR_MODES)
    def test_windowed_matches_full_decode(self, full_decode, name, cfr):
        full = full_decode(name, cfr)
        frames = _view(name, cfr)
        assert len(frames) == len(full)
        for sl in _slices(len(full)):
            expected = full[sl]
            got = list(frames[sl])
            assert len(got) == len(expected), f'{name} cfr={cfr} slice={sl}'
            for i, (g, e) in enumerate(zip(got, expected)):
                assert np.array_equal(g, e), f'{name} cfr={cfr} slice={sl} frame {i}'

    @pytest.mark.parametrize('name', CFR_VIDEOS)
    @pytest.mark.parametrize('cfr', CFR_MODES)
    def test_len_matches_iteration(self, full_decode, name, cfr):
        full = full_decode(name, cfr)
        assert len(_view(name, cfr)) == len(full)

    @pytest.mark.parametrize('name', CFR_VIDEOS)
    def test_int_indexing_matches_iteration(self, full_decode, name):
        full = full_decode(name, True)
        frames = _view(name, True)
        n = len(full)
        for idx in {0, 1, n // 2, n - 1, -1, -n}:
            assert np.array_equal(frames[idx], full[idx]), f'{name} idx={idx}'

    @pytest.mark.parametrize('name', CFR_VIDEOS)
    def test_chained_slicing(self, full_decode, name):
        full = full_decode(name, True)
        frames = _view(name, True)
        expected = full[1:][::2][:5]
        got = list(frames[1:][::2][:5])
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            assert np.array_equal(g, e)


class TestRepeatEachFrame:
    def _check_random_access(self, view):
        expected = list(view)
        n = len(view)
        assert n == len(expected)
        indices = {0, 1, n // 3, n // 2, n - 1, -1, -2, -n}
        for idx in indices:
            assert np.array_equal(view[idx], expected[idx]), f'idx={idx}'
        for bad in (n, n + 5, -n - 1):
            with pytest.raises(IndexError):
                view[bad]

    def test_plain_repeat_random_access(self):
        frames = _view('short.mp4', False)
        self._check_random_access(frames.repeat_each_frame(3))

    def test_sliced_then_repeated_random_access(self):
        frames = _view('short.mp4', False)
        self._check_random_access(frames[2:20:2].repeat_each_frame(3))

    def test_cfr_repeat_random_access(self):
        frames = _view('short.mp4', True)
        self._check_random_access(frames.repeat_each_frame(2))

    def test_cfr_dropping_repeat_random_access(self):
        frames = _view('short.mp4', 12.0)
        self._check_random_access(frames.repeat_each_frame(3))

    def test_repeat_iteration_content(self):
        frames = _view('short.mp4', False)
        source = list(frames)
        repeated = list(frames.repeat_each_frame(3))
        assert len(repeated) == 3 * len(source)
        for i, frame in enumerate(repeated):
            assert np.array_equal(frame, source[i // 3])

    def test_repeat_rejects_non_integer(self):
        frames = _view('short.mp4', False)
        with pytest.raises((TypeError, ValueError)):
            frames.repeat_each_frame(2.5)

    def test_repeat_rejects_zero(self):
        frames = _view('short.mp4', False)
        with pytest.raises(ValueError):
            frames.repeat_each_frame(0)


class TestLeadingGap:
    @pytest.fixture(scope='class')
    def shifted_video(self, tmp_path_factory):
        if shutil.which('ffmpeg') is None:
            pytest.skip('ffmpeg CLI not available')
        out = tmp_path_factory.mktemp('shifted') / 'shifted.mp4'
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-v',
                'error',
                '-i',
                str(DATA_DIR / 'short.mp4'),
                '-output_ts_offset',
                '0.2',
                '-c',
                'copy',
                str(out),
            ],
            check=True,
        )
        return str(out)

    def test_shifted_pts_does_not_change_frame_count(self, shifted_video):
        base = VideoFrames(str(DATA_DIR / 'short.mp4'), constant_framerate=True)
        shifted = VideoFrames(shifted_video, constant_framerate=True)
        n_base = sum(1 for _ in base)
        n_shifted = sum(1 for _ in shifted)
        assert len(base) == n_base
        assert len(shifted) == n_shifted
        assert n_shifted == n_base

    def test_shifted_content_matches(self, shifted_video):
        base = list(VideoFrames(str(DATA_DIR / 'short.mp4'), constant_framerate=True))
        shifted = list(VideoFrames(shifted_video, constant_framerate=True))
        for i, (b, s) in enumerate(zip(base, shifted)):
            assert np.array_equal(b, s), f'frame {i}'


class _MinimalFileObj:
    """Seekable file-like wrapper without getbuffer/getvalue (non-BytesIO path)."""

    def __init__(self, data):
        self._io = io.BytesIO(data)

    def read(self, size=-1):
        return self._io.read(size)

    def seek(self, offset, whence=0):
        return self._io.seek(offset, whence)

    def tell(self):
        return self._io.tell()


class TestFileLikeSources:
    @pytest.fixture(scope='class')
    def big_video_bytes(self, tmp_path_factory):
        # Noise compresses poorly, giving a file large enough (> a few MB) to
        # defeat libav's internal probe buffering, which masks shared-file-object
        # corruption on small files. Encoded with PyAV directly so this fixture
        # only depends on the code under test (VideoFrames).
        import av

        path = tmp_path_factory.mktemp('bytesio') / 'noise.mp4'
        rng = np.random.default_rng(0)
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('libx264', rate=30)
            stream.width, stream.height = 640, 360
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18'}
            for _ in range(60):
                arr = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        data = path.read_bytes()
        assert len(data) > 2_000_000, 'test video unexpectedly small'
        return data

    def test_interleaved_iterators_do_not_corrupt(self, big_video_bytes):
        frames = VideoFrames(io.BytesIO(big_video_bytes))
        baseline = list(VideoFrames(io.BytesIO(big_video_bytes)))
        it_a = iter(frames)
        it_b = iter(frames)
        got_a, got_b = [], []
        for _ in range(len(baseline)):
            got_a.append(next(it_a))
            got_b.append(next(it_b))
        for i, (a, b, e) in enumerate(zip(got_a, got_b, baseline)):
            assert np.array_equal(a, e), f'iterator A frame {i}'
            assert np.array_equal(b, e), f'iterator B frame {i}'

    def test_sequential_iteration_generic_fileobj(self, big_video_bytes):
        frames = VideoFrames(_MinimalFileObj(big_video_bytes))
        n_first = sum(1 for _ in frames)
        n_second = sum(1 for _ in frames)
        assert n_first == n_second == len(frames)


class TestFloatConversion:
    """Float outputs decode through the 16-bit internal path (rgb48), so all
    expectations are relative to the uint16 decode of the same video."""

    @pytest.mark.parametrize('name', ['short.mp4', '10bit.mp4'])
    def test_float16_uses_full_uint16_range(self, name):
        path = str(DATA_DIR / name)
        raw = next(iter(VideoFrames(path, dtype=np.uint16)))
        f16 = next(iter(VideoFrames(path, dtype=np.float16)))
        expected = (raw.astype(np.float32) / 65535.0).astype(np.float16)
        assert f16.dtype == np.float16
        assert np.array_equal(f16, expected)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_float32_float64_unchanged(self, dtype):
        path = str(DATA_DIR / 'short.mp4')
        raw = next(iter(VideoFrames(path, dtype=np.uint16)))
        converted = next(iter(VideoFrames(path, dtype=dtype)))
        assert converted.dtype == dtype
        assert np.allclose(converted, raw.astype(np.float64) / 65535.0, atol=1e-6)


class TestExports:
    def test_error_types_importable(self):
        from framepump import FilterConfigError, FramePumpError, IndexBuildError

        assert issubclass(IndexBuildError, FramePumpError)
        assert issubclass(FilterConfigError, FramePumpError)
