"""Environment report for troubleshooting, especially the GPU stack."""

from __future__ import annotations

import platform
import sys


def diagnose() -> str:
    """Print and return a report on framepump's environment.

    Covers the FFmpeg/PyAV versions in use, the NVIDIA driver and GPUs, the
    availability of each GPU feature (NVDEC decoding, VideoFramesCuda,
    NvJpegVideoWriter, GLVideoWriter, CameraFrames) with the reason when one
    is unavailable, and connected cameras. Attach the output to bug reports;
    read it when a GPU feature fails with a cryptic error.

    Returns:
        The report text (also printed to stdout).
    """
    lines = []
    add = lines.append

    from ._version import __version__

    add(f'framepump {__version__}')
    add(f'python {sys.version.split()[0]} on {platform.platform()}')

    try:
        import av

        libs = ', '.join(
            f'{name} {v[0]}.{v[1]}.{v[2]}' for name, v in sorted(av.library_versions.items())
        )
        add(f'PyAV {av.__version__} ({libs})')
        cuvid = 'h264_cuvid' in av.codecs_available
        add(f'FFmpeg CUDA decoders (h264_cuvid): {"present" if cuvid else "ABSENT"}')
    except Exception as e:
        add(f'PyAV: FAILED to import ({e!r})')

    add(_nvidia_report())

    add('GPU features:')
    for label, module in [
        ('VideoFrames(gpu=True)', None),
        ('VideoFramesCuda / frame index', 'framepump._cuda.frames'),
        ('CameraFrames', 'framepump._cuda.camera'),
        ('NvJpegVideoWriter', 'framepump.cuda_video_writer'),
        ('GLVideoWriter (NVENC)', 'framepump.video_writing_gl'),
    ]:
        if module is None:
            # Same requirement as the cuvid check above; state it in place.
            add('  VideoFrames(gpu=True): needs the FFmpeg CUDA decoders above + an NVIDIA GPU')
            continue
        try:
            __import__(module)
            add(f'  {label}: available')
        except Exception as e:
            add(f'  {label}: unavailable ({type(e).__name__}: {e})')

    try:
        import torch

        cuda = torch.cuda.is_available()
        add(f'torch {torch.__version__} (cuda available: {cuda})')
    except Exception:
        add('torch: not installed (fine unless you consume frames with PyTorch)')

    try:
        from ._v4l2 import list_cameras

        cams = list_cameras()
        if cams:
            add('cameras:')
            for cam in cams:
                best = str(cam.mjpeg_modes[0]) if cam.mjpeg_modes else 'no MJPEG modes'
                add(f'  {cam.device}: {cam.name} ({best})')
        else:
            add('cameras: none detected')
    except NotImplementedError:
        add('cameras: discovery not supported on this platform')
    except Exception as e:
        add(f'cameras: enumeration failed ({e!r})')

    report = '\n'.join(lines)
    print(report)
    return report


def _nvidia_report() -> str:
    try:
        with open('/proc/driver/nvidia/version') as f:
            driver_line = f.readline().strip()
    except OSError:
        return 'NVIDIA driver: not detected'
    parts = [f'NVIDIA driver: {driver_line}']
    try:
        from cuda.bindings import driver as cu

        cu.cuInit(0)
        err, count = cu.cuDeviceGetCount()
        for i in range(count if err == cu.CUresult.CUDA_SUCCESS else 0):
            err, dev = cu.cuDeviceGet(i)
            err, name = cu.cuDeviceGetName(128, dev)
            gpu_name = name.split(b'\0', 1)[0].decode(errors='replace')
            parts.append(f'  GPU {i}: {gpu_name}')
    except Exception as e:
        parts.append(f'  cuda-python: unavailable ({type(e).__name__}: {e})')
    return '\n'.join(parts)
