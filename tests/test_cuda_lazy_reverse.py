"""Lazy indexing and reverse iteration on VideoFramesCuda.

Mirrors the CPU-class contracts: construction and forward streaming must not
scan packets; length-dependent access builds the index once, shared across
views; negative-step slices must yield exactly the forward decode reversed.
Reverse chunks buffer owned GPU copies, so the yielded tensors must also stay
valid past the end of iteration.

GPU detection happens inside each test (fork-safe): pytest runs with
--forked, and CUDA state created at collection time breaks in the children.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / 'data'

REVERSE_SLICES = [
    slice(None, None, -1),
    slice(None, None, -2),
    slice(20, 5, -3),
    slice(-1, None, -1),
    slice(None, None, -40),  # |step| > 30: per-frame-seek path
]


def _gpu():
    """Import torch + the CUDA reader, skipping when NVDEC is unavailable."""
    import ctypes

    try:
        ctypes.CDLL('libnvcuvid.so.1')
    except OSError:
        pytest.skip('NVDEC (libnvcuvid) not available')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from framepump import VideoFramesCuda

    return torch, VideoFramesCuda


class TestCudaLazyIndex:
    def test_construction_and_streaming_are_lazy(self):
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        assert vf._lazy.index is None
        assert 'lazy' in repr(vf)
        _ = vf.fps
        frames = [torch.from_dlpack(f).clone() for f in vf[:5]]
        assert len(frames) == 5
        assert vf._lazy.index is None, 'prefix streaming must not build the index'

    def test_len_builds_shared_index(self):
        _, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        view = vf[::2]
        assert len(view) == 12
        assert vf._lazy.index is not None
        assert vf._lazy is view._lazy

    def test_streamed_content_matches_indexed(self):
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        streamed = [torch.from_dlpack(f).clone() for f in vf]
        assert torch.equal(torch.from_dlpack(vf[7]), streamed[7])


class TestCudaReverse:
    @pytest.mark.parametrize('sl', REVERSE_SLICES, ids=str)
    def test_reverse_matches_forward_uint8(self, sl):
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        forward = [torch.from_dlpack(f).clone() for f in vf]
        got = [torch.from_dlpack(f).clone() for f in vf[sl]]
        want = forward[sl]
        assert len(got) == len(want)
        for i, (a, b) in enumerate(zip(got, want)):
            assert torch.equal(a, b), f'{sl}: frame {i}'

    def test_reverse_uint16(self):
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / '10bit.mp4'), dtype=np.uint16)
        forward = [torch.from_dlpack(f).clone() for f in vf]
        got = [torch.from_dlpack(f).clone() for f in vf[::-1]]
        assert len(got) == len(forward)
        for a, b in zip(got, forward[::-1]):
            assert torch.equal(a, b)

    def test_reverse_frames_stay_valid_without_clone(self):
        # Reverse chunks yield owned buffers: unlike forward iteration's
        # shared-buffer contract, keeping the tensors must be safe
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        forward = [torch.from_dlpack(f).clone() for f in vf]
        kept = [torch.from_dlpack(f) for f in vf[::-1]]  # no .clone()
        for a, b in zip(kept, forward[::-1]):
            assert torch.equal(a, b)

    def test_slice_then_reverse(self):
        torch, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        forward = [torch.from_dlpack(f).clone() for f in vf]
        got = [torch.from_dlpack(f).clone() for f in vf[5:20][::-1]]
        want = forward[5:20][::-1]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert torch.equal(a, b)

    def test_reverse_fps(self):
        _, VideoFramesCuda = _gpu()
        vf = VideoFramesCuda(str(DATA_DIR / 'short.mp4'))
        assert vf[::-2].fps == pytest.approx(vf.fps / 2)
