"""Property-based tests for FrameSelection.

The oracle is plain Python list slicing: a selection built from a chain of
slices must resolve to exactly what applying those slices to a list would
give, for every length. When the selection claims to be streamable, consuming
an unknown-length iterator with islice semantics must give the same frames.
"""

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from framepump._selection import FrameSelection

_bound = st.one_of(st.none(), st.integers(min_value=-9, max_value=9))
_step = st.one_of(st.none(), st.integers(min_value=-4, max_value=4).filter(lambda x: x != 0))
_slice = st.builds(slice, _bound, _bound, _step)
_chain = st.lists(_slice, min_size=0, max_size=4)
_length = st.integers(min_value=0, max_value=40)


def _apply_chain(chain, n):
    result = list(range(n))
    for s in chain:
        result = result[s]
    return result


def _build(chain):
    sel = FrameSelection.identity()
    for s in chain:
        sel = sel.sliced(s)
    return sel


@given(_chain, _length)
@settings(max_examples=500)
def test_resolution_matches_list_slicing(chain, n):
    sel = _build(chain)
    assert list(sel.resolve(n).range) == _apply_chain(chain, n)


@given(_chain, _length)
@settings(max_examples=500)
def test_streamable_matches_islice_semantics(chain, n):
    sel = _build(chain)
    s = sel.streamable_slice
    if s is None:
        return
    start = s.start if s.start is not None else 0
    step = s.step if s.step is not None else 1
    streamed = list(itertools.islice(iter(range(n)), start, s.stop, step))
    assert streamed == _apply_chain(chain, n)


@given(_chain, _length)
@settings(max_examples=500)
def test_step_product_matches_resolved_step(chain, n):
    sel = _build(chain)
    resolved = sel.resolve(n).range
    if len(resolved) >= 2:
        assert resolved.step == sel.step_product


@given(_chain, _length, _slice)
@settings(max_examples=300)
def test_slicing_after_resolution_matches_before(chain, n, extra):
    symbolic = _build(chain).sliced(extra).resolve(n)
    resolved_first = _build(chain).resolve(n).sliced(extra)
    assert list(symbolic.range) == list(resolved_first.range)


def test_identity_is_streamable_full_slice():
    s = FrameSelection.identity().streamable_slice
    assert s == slice(None)


def test_resolved_selection_reports_not_streamable():
    sel = FrameSelection.identity().resolve(10)
    assert sel.is_resolved
    assert sel.streamable_slice is None
    assert list(sel.range) == list(range(10))


def test_symbolic_negative_step_not_streamable():
    assert FrameSelection.identity().sliced(slice(None, None, -1)).streamable_slice is None
    assert FrameSelection.identity().sliced(slice(-5, None)).streamable_slice is None


def test_double_reversal_streams_again():
    sel = FrameSelection.identity().sliced(slice(None, None, -1)).sliced(slice(None, None, -1))
    assert sel.streamable_slice == slice(None)
