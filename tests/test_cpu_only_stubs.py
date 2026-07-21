"""CPU-only machines must get helpful errors from the CUDA-only classes.

Without the CUDA stack, JpegVideoWriterCUDA/VideoFramesCuda/CudaToGLUploader
are import-time stubs: instantiating one must raise an ImportError naming the
missing dependencies, not "'NoneType' object is not callable".
"""

import subprocess
import sys

import pytest


def test_stub_raises_helpful_import_error():
    from framepump import _make_cuda_stub

    stub = _make_cuda_stub('VideoFramesCuda', 'cuda-python and PyNvVideoCodec')
    with pytest.raises(ImportError, match='cuda-python and PyNvVideoCodec'):
        stub()


def test_import_without_cuda_gives_stubs():
    code = """
import sys

class _Blocker:
    blocked = ('cuda', 'PyNvVideoCodec')

    def find_module(self, name, path=None):
        if name in self.blocked or any(name.startswith(b + '.') for b in self.blocked):
            return self

    def load_module(self, name):
        raise ImportError(f'{name} is blocked for this test')

sys.meta_path.insert(0, _Blocker())

import framepump

for cls_name in ('JpegVideoWriterCUDA', 'VideoFramesCuda', 'CudaToGLUploader'):
    cls = getattr(framepump, cls_name)
    try:
        cls()
    except ImportError as e:
        assert 'requires' in str(e), (cls_name, str(e))
    except TypeError as e:
        raise SystemExit(f'{cls_name} gave TypeError instead of ImportError: {e}')
    else:
        raise SystemExit(f'{cls_name} unexpectedly instantiated')
print('OK')
"""
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout
