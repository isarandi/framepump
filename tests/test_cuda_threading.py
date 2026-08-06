"""Thread-safety and process-coexistence contracts for GPU decoding.

Decode sessions must work from any thread (prefetch patterns), even when a
different thread owns the process's CUDA state, and a process that used GPU
decoding must exit cleanly. Both contracts break in ways that kill or wedge
the whole interpreter (CUDA_ERROR_INVALID_CONTEXT aborts, driver-level
livelocks at teardown), so every case runs in a subprocess with a timeout.

Both were observed in real ML workloads merely importing torchvision.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / 'data'
VIDEO = DATA_DIR / 'short.mp4'


def _require_torch_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available for torch')


def _run_isolated(code: str, timeout: float = 180.0) -> None:
    """Run test code in a subprocess; crashes and hangs must not leak out."""
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f'subprocess hung (>{timeout:.0f}s): '
            f'{(e.stdout or b"").decode(errors="replace")[-2000:]}'
        ) from e
    assert proc.returncode == 0, (
        f'subprocess exited {proc.returncode}\n'
        f'stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}'
    )


_CONSUME_IN_WORKER = f'''
import threading, sys
import torch

failures = []

def consume():
    try:
        import framepump
        v = framepump.VideoFramesCuda({str(VIDEO)!r})
        for i, frame in enumerate(v):
            t = torch.from_dlpack(frame)
            assert t.is_cuda and t.dtype == torch.uint8, (t.device, t.dtype)
            assert t.shape == (720, 1280, 3), t.shape
            float(t.float().mean())
            if i >= 3:
                break
        v.close()
    except BaseException as e:  # noqa: BLE001
        failures.append(e)
        raise

MAIN_CTX_SETUP
t = threading.Thread(target=consume)
t.start()
t.join(timeout=120)
sys.exit(2 if t.is_alive() else (1 if failures else 0))
'''


class TestWorkerThreadDecoding:
    def test_consume_entirely_in_worker_thread(self):
        """Session + decode + DLPack export all in a worker thread."""
        _require_torch_cuda()
        pytest.importorskip('PyNvVideoCodec')
        _run_isolated(_CONSUME_IN_WORKER.replace('MAIN_CTX_SETUP', ''))

    def test_worker_decode_while_main_owns_cuda(self):
        """Main thread owns torch's CUDA state; worker decodes and exports.

        This is the prefetch pattern that crashed with
        CUDA_ERROR_INVALID_CONTEXT before decode sessions were bound to the
        primary context.
        """
        _require_torch_cuda()
        pytest.importorskip('PyNvVideoCodec')
        _run_isolated(
            _CONSUME_IN_WORKER.replace('MAIN_CTX_SETUP', "torch.zeros(1, device='cuda')")
        )

    def test_iterator_advanced_across_threads(self):
        """Iteration starts in the main thread, continues in a worker that
        never touches CUDA itself — the frame proxy must supply the context."""
        _require_torch_cuda()
        pytest.importorskip('PyNvVideoCodec')
        _run_isolated(f'''
import threading, sys
import torch
import framepump

v = framepump.VideoFramesCuda({str(VIDEO)!r})
it = iter(v)
t0 = torch.from_dlpack(next(it))
assert t0.is_cuda and t0.shape == (720, 1280, 3)

failures = []

def advance():
    try:
        for _ in range(3):
            t = torch.from_dlpack(next(it))
            assert t.is_cuda and t.shape == (720, 1280, 3)
            float(t.float().mean())
    except BaseException as e:  # noqa: BLE001
        failures.append(e)
        raise

t = threading.Thread(target=advance)
t.start()
t.join(timeout=120)
sys.exit(2 if t.is_alive() else (1 if failures else 0))
''')


class TestCleanProcessExit:
    def test_gpu_decode_process_exits_cleanly_with_torchvision(self):
        """A process that imports torchvision and uses VideoFrames(gpu=True)
        must finish decoding AND exit — with FFmpeg decoder threading on
        hwaccel containers, decoder teardown livelocks in the NVIDIA driver
        under exactly this coexistence (torchvision's C++ extension loaded).
        """
        _require_torch_cuda()
        pytest.importorskip('torchvision')
        code = f'''
import torchvision
import numpy as np
from framepump import VideoFrames

v = VideoFrames({str(VIDEO)!r}, gpu=True)
try:
    v[0]
except Exception:
    import sys
    sys.exit(77)  # NVDEC not available: report as skip
n = len(v)
assert n == 24, n
count = sum(1 for _ in v)
assert count == n, (count, n)
'''
        try:
            proc = subprocess.run(
                [sys.executable, '-c', code], capture_output=True, text=True, timeout=180
            )
        except subprocess.TimeoutExpired as e:
            raise AssertionError('process wedged after GPU decode (teardown livelock)') from e
        if proc.returncode == 77:
            pytest.skip('NVDEC GPU decoding not available')
        assert proc.returncode == 0, f'{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}'
