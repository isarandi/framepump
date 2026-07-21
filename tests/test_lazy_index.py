"""Lazy frame-index construction.

VideoFrames must not scan the file's packets at construction; forward
streaming access (iteration, prefix/step slicing and reducible chains)
must work without ever building the index, while length-dependent access
builds it exactly once, shared across all views.
"""

import io
from pathlib import Path

import numpy as np
import pytest

from framepump import VideoDecodeError, VideoFrames

DATA_DIR = Path(__file__).parent / 'data'
SHORT = str(DATA_DIR / 'short.mp4')


def _index_built(vf):
    return vf._lazy.index is not None


class TestConstructionIsLazy:
    def test_no_index_at_construction(self):
        vf = VideoFrames(SHORT)
        assert not _index_built(vf)

    def test_metadata_without_index(self):
        vf = VideoFrames(SHORT)
        assert vf.imshape == (720, 1280)
        assert vf.fps == pytest.approx(23.976, abs=0.001)
        assert 'lazy' in repr(vf)
        assert not _index_built(vf)

    def test_slicing_and_fps_stay_symbolic(self):
        vf = VideoFrames(SHORT)
        view = vf[::2][:10]
        assert view.fps == pytest.approx(vf.fps / 2)
        assert not _index_built(vf)


class TestStreamingAccess:
    def test_full_iteration_streams(self):
        vf = VideoFrames(SHORT)
        frames = [f for f in vf]
        assert len(frames) == 24
        assert not _index_built(vf)

    def test_prefix_slice_streams(self):
        vf = VideoFrames(SHORT)
        frames = [f for f in vf[:5]]
        assert len(frames) == 5
        assert not _index_built(vf)

    def test_chained_slices_stream(self):
        vf = VideoFrames(SHORT)
        full = [f for f in vf]
        got = [f for f in vf[2:][::3][:5]]
        assert not _index_built(vf)
        want = full[2:][::3][:5]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)

    def test_streamed_content_equals_resolved_content(self):
        lazy = VideoFrames(SHORT)
        eager = VideoFrames(SHORT)
        len(eager)  # force the index; iteration then uses the seek paths
        for a, b in zip(lazy[3:15:2], eager[3:15:2]):
            assert np.array_equal(a, b)
        assert not _index_built(lazy)
        assert _index_built(eager)

    def test_large_start_builds_index_and_seeks(self, monkeypatch):
        import framepump._core as core

        monkeypatch.setattr(core, '_STREAM_MAX_SKIP', 4)
        vf = VideoFrames(SHORT)
        full = [f for f in vf]
        assert not _index_built(vf)
        got = [f for f in vf[10:13]]
        assert _index_built(vf)
        for a, b in zip(got, full[10:13]):
            assert np.array_equal(a, b)

    def test_bytesio_source_streams(self):
        data = Path(SHORT).read_bytes()
        vf = VideoFrames(io.BytesIO(data))
        frames = [f for f in vf[:4]]
        assert len(frames) == 4
        assert not _index_built(vf)


class TestIndexTriggers:
    def test_len_builds_index(self):
        vf = VideoFrames(SHORT)
        assert len(vf) == 24
        assert _index_built(vf)

    def test_int_indexing_builds_index(self):
        vf = VideoFrames(SHORT)
        _ = vf[10]
        assert _index_built(vf)

    def test_negative_slice_component_builds_index_on_iteration(self):
        vf = VideoFrames(SHORT)
        got = [f for f in vf[-5:]]
        assert _index_built(vf)
        assert len(got) == 5

    def test_cfr_iteration_builds_index(self):
        vf = VideoFrames(SHORT, constant_framerate=True)
        _ = [f for f in vf[:3]]
        assert _index_built(vf)

    def test_index_shared_across_views(self):
        vf = VideoFrames(SHORT)
        view = vf[::2]
        len(view)
        assert _index_built(vf), 'index built via a view must be visible to the parent'
        assert vf._lazy is view._lazy

    def test_list_builds_index_via_len_hint(self):
        # list() calls len() as a preallocation hint before iterating —
        # documented behavior: use a for-loop or comprehension to stay lazy.
        vf = VideoFrames(SHORT)
        frames = list(vf)
        assert len(frames) == 24
        assert _index_built(vf)


class TestLazyBrokenStreams:
    def test_streamed_iteration_of_unreliable_file_is_lazy_and_correct(self):
        path = str(DATA_DIR / 'unreliable_seek.ts')
        vf = VideoFrames(path)
        streamed = [f for f in vf]
        assert not _index_built(vf), 'pure forward streaming needs no probe'
        assert len(vf) == len(streamed), 'materialized index must match the stream'

    def test_streamed_empty_decode_still_raises(self):
        vf = VideoFrames(str(DATA_DIR / 'no_decodable_frames.mov'))
        with pytest.raises(VideoDecodeError, match='no frames'):
            _ = [f for f in vf]


class TestSeekableParameterLazy:
    def test_seekable_false_lazy_indexing(self):
        ref = VideoFrames(SHORT)
        full = [f for f in ref]
        vf = VideoFrames(SHORT, seekable=False)
        assert not _index_built(vf)
        assert np.array_equal(vf[10], full[10])
        assert _index_built(vf)
