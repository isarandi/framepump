"""Negative-step (reverse) iteration.

The oracle is always the forward decode: for any slice with negative step,
iterating the view must yield exactly ``forward_frames[the_slice]``. Reverse
iteration is chunked (backward chunks decoded forward and buffered), so the
tests also shrink the chunk budget to force many chunk boundaries.
"""

from pathlib import Path

import numpy as np
import pytest

from framepump import VideoFrames

DATA_DIR = Path(__file__).parent / 'data'

REVERSE_SLICES = [
    slice(None, None, -1),
    slice(None, None, -2),
    slice(None, None, -7),
    slice(20, 5, -3),
    slice(-1, None, -1),
    slice(-3, 2, -2),
    slice(None, None, -40),  # |step| > 30: individual-seek path
]


def _check_against_forward(vf, sl):
    forward = [f.copy() for f in vf]
    got = [f for f in vf[sl]]
    want = forward[sl]
    assert len(got) == len(want), f'{sl}: {len(got)} != {len(want)}'
    for i, (a, b) in enumerate(zip(got, want)):
        assert np.array_equal(a, b), f'{sl}: frame {i} differs'


class TestReverseBasic:
    @pytest.mark.parametrize('sl', REVERSE_SLICES, ids=str)
    @pytest.mark.parametrize('name', ['short.mp4', 'short.mkv', 'variable_fps.mp4'])
    def test_reverse_matches_forward(self, name, sl):
        _check_against_forward(VideoFrames(str(DATA_DIR / name)), sl)

    def test_reverse_uint16(self):
        vf = VideoFrames(str(DATA_DIR / '10bit.mp4'), dtype=np.uint16)
        _check_against_forward(vf, slice(None, None, -1))

    def test_reverse_float32(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'), dtype=np.float32)
        _check_against_forward(vf, slice(None, None, -2))

    def test_reverse_resized(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4')).resized((64, 64))
        _check_against_forward(vf, slice(None, None, -1))


class TestReverseComposition:
    def test_slice_then_reverse(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        forward = [f.copy() for f in vf]
        got = [f for f in vf[5:20][::-1]]
        want = forward[5:20][::-1]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)

    def test_reverse_then_slice(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        forward = [f.copy() for f in vf]
        got = [f for f in vf[::-1][3:8]]
        want = forward[::-1][3:8]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)

    def test_double_reverse_streams_lazily(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        view = vf[::-1][::-1]
        frames = [f for f in view]
        assert len(frames) == 24
        assert vf._lazy.index is None, 'double reversal reduces to identity: streamable'

    def test_reverse_with_repeat(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        forward = [f.copy() for f in vf]
        got = [f for f in vf[::-1].repeat_each_frame(3)]
        want = [f for f in forward[::-1] for _ in range(3)]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)

    def test_reverse_int_indexing(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        forward = [f.copy() for f in vf]
        rev = vf[::-1]
        assert np.array_equal(rev[0], forward[-1])
        assert np.array_equal(rev[-1], forward[0])
        assert len(rev) == len(forward)

    def test_reverse_fps(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        assert vf[::-1].fps == pytest.approx(vf.fps)
        assert vf[::-2].fps == pytest.approx(vf.fps / 2)


class TestReverseCfr:
    @pytest.mark.parametrize('mode', [True, 12.0, 60.0], ids=str)
    @pytest.mark.parametrize('sl', [slice(None, None, -1), slice(None, None, -2)], ids=str)
    def test_reverse_cfr_matches_forward(self, mode, sl):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'), constant_framerate=mode)
        _check_against_forward(vf, sl)


class TestReverseChunking:
    def test_tiny_chunks_still_correct(self, monkeypatch):
        import framepump._core as core

        # Budget for ~2 frames per chunk: maximum chunk-boundary stress
        monkeypatch.setattr(core, '_REVERSE_CHUNK_BYTES', 720 * 1280 * 3 * 2)
        _check_against_forward(VideoFrames(str(DATA_DIR / 'short.mp4')), slice(None, None, -1))

    def test_chunk_bounds_respect_budget(self):
        vf = VideoFrames(str(DATA_DIR / 'short.mp4'))
        min_chunk, max_chunk, fallback = vf._reverse_chunk_bounds(np.uint8)
        assert 1 <= min_chunk <= fallback <= max_chunk <= 64


class TestReverseBrokenStreams:
    def test_reverse_timestampless_stream(self):
        vf = VideoFrames(str(DATA_DIR / 'raw25.h264'))
        _check_against_forward(vf, slice(None, None, -1))

    def test_reverse_unreliable_seek_file(self):
        # Probe-failed file: sequential-only mode; chunked reverse must
        # still reproduce the forward decode exactly
        vf = VideoFrames(str(DATA_DIR / 'unreliable_seek.ts'))
        _check_against_forward(vf, slice(None, None, -1))
