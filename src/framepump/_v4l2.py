"""Camera discovery via V4L2 ioctls. Linux-only, no dependencies beyond the stdlib."""

from __future__ import annotations

import dataclasses
import fcntl
import re
import struct
import sys
from pathlib import Path

# From linux/videodev2.h (stable kernel ABI)
_VIDIOC_QUERYCAP = 0x80685600
_VIDIOC_ENUM_FMT = 0xC0405602
_VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
_VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B
_BUF_TYPE_VIDEO_CAPTURE = 1
_CAP_VIDEO_CAPTURE = 0x00000001
_CAP_DEVICE_CAPS = 0x80000000
_FRMSIZE_TYPE_DISCRETE = 1
_FRMIVAL_TYPE_DISCRETE = 1


@dataclasses.dataclass(frozen=True)
class CameraMode:
    """One capture mode of a camera: frame size and its highest frame rate."""

    shape: tuple[int, int]
    """Frame size as (height, width)."""

    fps: float
    """Highest frame rate the camera offers at this size."""

    def __str__(self) -> str:
        h, w = self.shape
        return f'{w}x{h} @ {self.fps:g} fps'


@dataclasses.dataclass(frozen=True)
class CameraInfo:
    """A V4L2 capture device and the modes relevant to :class:`CameraFrames`."""

    device: str
    """Device path, e.g. ``'/dev/video0'`` — pass this to :class:`CameraFrames`."""

    name: str
    """Human-readable camera name reported by the driver."""

    mjpeg_modes: tuple[CameraMode, ...]
    """MJPEG capture modes (what :class:`CameraFrames` uses), largest first."""

    formats: tuple[str, ...]
    """All pixel formats the device offers (fourcc codes, e.g. ``'MJPG'``, ``'YUYV'``)."""

    def __str__(self) -> str:
        modes = ', '.join(str(m) for m in self.mjpeg_modes) or 'none'
        return f'{self.device}: {self.name}\n  MJPEG modes: {modes}\n  formats: {", ".join(self.formats)}'


def list_cameras() -> list[CameraInfo]:
    """Discover connected cameras and the capture modes they support.

    Enumerates V4L2 video-capture devices (webcams and other UVC devices) and
    queries their pixel formats, frame sizes and frame rates directly from the
    kernel. Non-capture device nodes (metadata channels) are excluded. Use the
    reported device path and an MJPEG mode's shape/fps with
    :class:`CameraFrames`:

        >>> for cam in framepump.list_cameras():
        ...     print(cam)
        /dev/video0: HD Webcam
          MJPEG modes: 1280x720 @ 30 fps, 640x480 @ 30 fps
          formats: MJPG, YUYV

    Returns:
        One :class:`CameraInfo` per usable capture device, in device order.
        Devices that cannot be opened (e.g. permissions) are skipped.

    Raises:
        NotImplementedError: On platforms other than Linux (V4L2 only for now).
    """
    if sys.platform != 'linux':
        raise NotImplementedError('Camera discovery is only supported on Linux (V4L2) for now')
    infos = []
    devices = sorted(
        (p for p in Path('/dev').glob('video*') if re.fullmatch(r'video\d+', p.name)),
        key=lambda p: int(p.name[5:]),
    )
    for dev in devices:
        info = _query_device(dev)
        if info is not None:
            infos.append(info)
    return infos


def _query_device(dev: Path) -> CameraInfo | None:
    try:
        fd = open(dev, 'rb', buffering=0)
    except OSError:
        return None
    try:
        # struct v4l2_capability: driver[16], card[32], bus_info[32],
        # version u32, capabilities u32, device_caps u32, reserved u32[3]
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
        card = bytes(buf[16:48]).split(b'\0', 1)[0].decode(errors='replace')
        capabilities, device_caps = struct.unpack_from('<II', buf, 84)
        caps = device_caps if capabilities & _CAP_DEVICE_CAPS else capabilities
        if not caps & _CAP_VIDEO_CAPTURE:
            return None

        formats = _enum_formats(fd)
        mjpeg_modes = []
        if 'MJPG' in formats:
            for h, w in _enum_frame_sizes(fd, b'MJPG'):
                fps = _max_frame_rate(fd, b'MJPG', w, h)
                mjpeg_modes.append(CameraMode(shape=(h, w), fps=fps))
        mjpeg_modes.sort(key=lambda m: m.shape[0] * m.shape[1], reverse=True)
        return CameraInfo(
            device=str(dev), name=card,
            mjpeg_modes=tuple(mjpeg_modes), formats=tuple(formats),
        )  # fmt: skip
    except OSError:
        return None
    finally:
        fd.close()


def _enum_formats(fd) -> list[str]:
    """Fourcc codes of all capture pixel formats, in driver order."""
    formats = []
    for index in range(64):
        # struct v4l2_fmtdesc: index, type, flags u32; description[32];
        # pixelformat, mbus_code u32; reserved u32[3]
        buf = bytearray(64)
        struct.pack_into('<II', buf, 0, index, _BUF_TYPE_VIDEO_CAPTURE)
        if not _try_ioctl(fd, _VIDIOC_ENUM_FMT, buf):
            break
        (pixelformat,) = struct.unpack_from('<I', buf, 44)
        formats.append(pixelformat.to_bytes(4, 'little').decode(errors='replace').strip())
    return formats


def _enum_frame_sizes(fd, fourcc: bytes) -> list[tuple[int, int]]:
    """Discrete frame sizes for a format, as (height, width) tuples."""
    sizes = []
    for index in range(128):
        # struct v4l2_frmsizeenum: index, pixel_format, type u32;
        # union { discrete {width, height} | stepwise }; reserved u32[2]
        buf = bytearray(44)
        struct.pack_into('<I4s', buf, 0, index, fourcc)
        if not _try_ioctl(fd, _VIDIOC_ENUM_FRAMESIZES, buf):
            break
        (size_type,) = struct.unpack_from('<I', buf, 8)
        if size_type != _FRMSIZE_TYPE_DISCRETE:
            break  # stepwise/continuous: no finite list to report
        w, h = struct.unpack_from('<II', buf, 12)
        sizes.append((h, w))
    return sizes


def _max_frame_rate(fd, fourcc: bytes, width: int, height: int) -> float:
    """Highest discrete frame rate offered at the given size (0.0 if unknown)."""
    best = 0.0
    for index in range(64):
        # struct v4l2_frmivalenum: index, pixel_format, width, height, type u32;
        # union { discrete v4l2_fract {numerator, denominator} | stepwise };
        # reserved u32[2]
        buf = bytearray(52)
        struct.pack_into('<I4sII', buf, 0, index, fourcc, width, height)
        if not _try_ioctl(fd, _VIDIOC_ENUM_FRAMEINTERVALS, buf):
            break
        (ival_type,) = struct.unpack_from('<I', buf, 16)
        if ival_type != _FRMIVAL_TYPE_DISCRETE:
            break
        numerator, denominator = struct.unpack_from('<II', buf, 20)
        if numerator:
            best = max(best, denominator / numerator)
    return best


def _try_ioctl(fd, request: int, buf: bytearray) -> bool:
    """Run an enumeration ioctl; False signals the end of the enumeration."""
    try:
        fcntl.ioctl(fd, request, buf)
        return True
    except OSError:
        return False
