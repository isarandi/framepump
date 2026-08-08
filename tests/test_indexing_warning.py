"""PerformanceWarning on the frames[i]-in-a-loop anti-pattern."""

import os
import warnings

import pytest

import framepump

VIDEO = os.path.join(os.path.dirname(__file__), 'data', 'exact_30fps.mp4')
THRESHOLD = 10  # _core._INDEXING_WARN_THRESHOLD


def test_consecutive_indexing_warns():
    v = framepump.VideoFrames(VIDEO)
    with pytest.warns(framepump.PerformanceWarning, match='consecutive forward indexed'):
        for i in range(THRESHOLD + 2):
            v[i]


def test_warns_only_once_per_instance():
    v = framepump.VideoFrames(VIDEO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i in range(2 * (THRESHOLD + 2)):
            v[i % 28]
    perf = [w for w in caught if issubclass(w.category, framepump.PerformanceWarning)]
    assert len(perf) == 1


def test_fresh_instance_warns_again():
    for _ in range(2):
        v = framepump.VideoFrames(VIDEO)
        with pytest.warns(framepump.PerformanceWarning):
            for i in range(THRESHOLD + 2):
                v[i]


def test_strided_index_loop_warns():
    """Small forward strides are the same anti-pattern (decode-through gap)."""
    v = framepump.VideoFrames(VIDEO)
    with pytest.warns(framepump.PerformanceWarning):
        for i in range(0, 2 * (THRESHOLD + 2), 2):
            v[i]


def test_scattered_access_does_not_warn():
    v = framepump.VideoFrames(VIDEO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i in [0, 25, 3, 20, 7, 27, 1, 24, 5, 22, 9, 26, 2, 21]:  # no forward run
            v[i]
    assert not [w for w in caught if issubclass(w.category, framepump.PerformanceWarning)]


def test_few_accesses_do_not_warn():
    v = framepump.VideoFrames(VIDEO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i in range(THRESHOLD - 2):
            v[i]
    assert not [w for w in caught if issubclass(w.category, framepump.PerformanceWarning)]


def test_iteration_and_gather_do_not_warn():
    v = framepump.VideoFrames(VIDEO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in v[: THRESHOLD + 5]:
            pass
        v[list(range(THRESHOLD + 5))]
        list(v.frames_at(range(THRESHOLD + 5)))
    assert not [w for w in caught if issubclass(w.category, framepump.PerformanceWarning)]
