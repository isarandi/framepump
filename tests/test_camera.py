"""CameraFrames contracts: live GPU frames, latest-frame-always semantics.

Hardware-dependent: skipped unless a V4L2 camera with MJPEG support and a
CUDA-capable setup are present.
"""

import os
import time

import pytest

DEVICE = '/dev/video0'


def _require_camera():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available for torch')
    pytest.importorskip('PyNvVideoCodec')
    if not os.path.exists(DEVICE):
        pytest.skip(f'no camera at {DEVICE}')
    return torch


def test_list_cameras():
    """Camera discovery via V4L2 (no CUDA needed, but needs a camera)."""
    if not os.path.exists(DEVICE):
        pytest.skip(f'no camera at {DEVICE}')
    from framepump import list_cameras

    cams = list_cameras()
    assert any(cam.device == DEVICE for cam in cams)
    cam = next(cam for cam in cams if cam.device == DEVICE)
    assert cam.name
    assert 'MJPG' in cam.formats
    assert cam.mjpeg_modes, 'UVC camera should offer MJPEG modes'
    h, w = cam.mjpeg_modes[0].shape
    assert h > 0 and w > 0 and cam.mjpeg_modes[0].fps > 0
    # sorted largest first
    pixel_counts = [m.shape[0] * m.shape[1] for m in cam.mjpeg_modes]
    assert pixel_counts == sorted(pixel_counts, reverse=True)


def test_live_frames_and_metadata():
    torch = _require_camera()
    from framepump import CameraFrames

    with CameraFrames(DEVICE, shape=(720, 1280), fps=30) as cam:
        assert cam.imshape[0] > 0 and cam.fps > 0
        frames = 0
        for frame in cam:
            t = torch.from_dlpack(frame)
            assert t.is_cuda and t.dtype == torch.uint8
            assert tuple(t.shape) == (*cam.imshape, 3)
            assert cam.last_capture_time is not None
            frames += 1
            if frames >= 10:
                break
        assert frames == 10


def test_slow_consumer_stays_fresh():
    """A consumer far slower than the camera must skip frames, not queue."""
    torch = _require_camera()
    from framepump import CameraFrames

    with CameraFrames(DEVICE, shape=(640, 640), fps=30) as cam:
        staleness = None
        for i, frame in enumerate(cam):
            time.sleep(0.15)  # 5x slower than the camera
            torch.from_dlpack(frame).float().mean().item()
            staleness = time.monotonic() - cam.last_capture_time
            if i >= 12:
                break
        # Naive queueing would exceed a second within these 12 frames;
        # latest-frame delivery keeps staleness near the consumer's own period.
        assert staleness < 0.5, f'staleness {staleness:.2f}s — backlog not dropped'


def test_batched_adaptive_never_repeats():
    """batched(n) yields adaptive batches (k <= n), chronological, and never
    delivers the same frame twice."""
    torch = _require_camera()
    from framepump import CameraFrames

    with CameraFrames(DEVICE, shape=(640, 640), fps=30) as cam:
        all_ts, sizes = [], []
        for i, batch in enumerate(cam.batched(3)):
            t = torch.from_dlpack(batch)
            assert t.is_cuda and t.dtype == torch.uint8
            k = t.shape[0]
            assert 1 <= k <= 3
            assert tuple(t.shape[1:]) == (*cam.imshape, 3)
            assert len(cam.last_capture_times) == k
            assert cam.last_capture_time == cam.last_capture_times[-1]
            all_ts.extend(cam.last_capture_times)
            sizes.append(k)
            time.sleep(0.1)  # ~3 camera frames per consumer step
            if i >= 9:
                break
        # strictly increasing capture times within and across batches
        # = chronological and no frame ever delivered twice
        assert all(b > a for a, b in zip(all_ts, all_ts[1:]))
        # a consumer this slow must have received multi-frame batches
        assert max(sizes) >= 2


def test_batched_covers_missed_interval_evenly():
    """A slow consumer gets frames spread across the whole interval it
    missed, not a burst of near-identical latest frames."""
    torch = _require_camera()
    from framepump import CameraFrames

    with CameraFrames(DEVICE, shape=(640, 640), fps=30) as cam:
        it = cam.batched(3)
        next(it)  # anchor a first delivery
        for _ in range(5):
            time.sleep(0.3)  # ~9 camera frames pass
            batch = next(it)
            k = torch.from_dlpack(batch).shape[0]
            ts = cam.last_capture_times
            assert k == 3
            # even coverage: the batch spans most of the ~0.3 s gap
            # (the 3 newest consecutive frames would span only ~0.07 s)
            assert ts[-1] - ts[0] > 0.15
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            assert all(0.05 < g < 0.25 for g in gaps)


def test_close_is_prompt_and_repeatable():
    _require_camera()
    from framepump import CameraFrames

    cam = CameraFrames(DEVICE, shape=(640, 640), fps=30)
    next(iter(cam))
    t = time.monotonic()
    cam.close()
    assert time.monotonic() - t < 3.0
    cam.close()  # idempotent
