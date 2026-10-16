"""available_backends() reports exactly the GL encoding backends that _validate_backend accepts."""

import pytest

from framepump import video_writing_gl as vwg


def test_available_backends_is_consistent_with_validation():
    backends = vwg.available_backends()
    assert set(backends) <= {'glx', 'cuda'}
    if 'cuda' in backends:
        assert 'glx' in backends  # both need libnvidia-encode; cuda adds the bindings
    if 'cuda' in backends:
        vwg._validate_backend('cuda')
    elif vwg.NvencCudaEncoder is None:
        with pytest.raises(ImportError, match=r'framepump\[cuda\]'):
            vwg._validate_backend('cuda')


def test_no_cuda_bindings_means_no_cuda_backend(monkeypatch):
    monkeypatch.setattr(vwg, 'NvencCudaEncoder', None)
    assert 'cuda' not in vwg.available_backends()
    with pytest.raises(ImportError, match=r'framepump\[cuda\]'):
        vwg._validate_backend('cuda')
