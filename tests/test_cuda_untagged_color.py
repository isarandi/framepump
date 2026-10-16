"""VideoFramesCuda colour convention for streams without colour description.

The oracle is a written flat-colour pattern: encoders of the ffmpeg family
(the ffmpeg CLI, OpenCV's mp4v writer) convert RGB → YUV with BT.601 limited
range and write no colour tags, and libswscale decodes untagged streams
with BT.601 at any resolution. Both readers must recover the written RGB
from such files, agree with each other, keep honouring explicit tags, and
the CUDA reader's explicit ``color_space=`` override must still win.

The CUDA reader converts 8-bit sources with a kernel that reproduces
libswscale's fixed-point arithmetic, so the two readers are compared
tightly (bit-exact for H.264, IDCT drift only for MPEG-4). A CPU-only test
pins that arithmetic against libswscale itself.

GPU detection happens inside each test (fork-safe): pytest runs with
--forked, and CUDA state created at collection time breaks in the children.
"""

import shutil
import subprocess

import numpy as np
import pytest

from framepump import VideoFrames

FFMPEG = shutil.which('ffmpeg')
pytestmark = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg CLI not available')

W, H, FPS, N_FRAMES = 128, 160, 30, 12
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 128, 128), (200, 50, 120), (30, 180, 90), (90, 60, 220),
]  # fmt: skip
BAND = H // len(COLORS)
CHECK_FRAME = 5
# Band-centre recovery error tolerance (mean abs over the band centre): 4:2:0
# subsampling and quantisation cost ~1-2 levels; the wrong matrix costs > 10
# on the saturated bands.
RECOVERY_TOL = 3.0


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
    pytest.importorskip('PyNvVideoCodec')
    from framepump import VideoFramesCuda

    return torch, VideoFramesCuda


def _pattern() -> np.ndarray:
    im = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate(COLORS):
        im[k * BAND : (k + 1) * BAND, :] = c
    return im


def _band_means(frame: np.ndarray) -> np.ndarray:
    """Mean RGB of each band's centre (away from band edges and borders)."""
    return np.stack(
        [
            frame[k * BAND + 4 : (k + 1) * BAND - 4, 8:-8].reshape(-1, 3).mean(0)
            for k in range(len(COLORS))
        ]
    )


def _recovery_error(frame: np.ndarray) -> float:
    return float(np.abs(_band_means(frame) - np.array(COLORS, np.float64)).mean())


def _encode(out_path, codec: str, *, tagged: bool) -> str:
    """Encode the pattern with the ffmpeg CLI: untagged (601 conversion, no
    tags — what ffmpeg-family encoders write) or tagged BT.709 (converted
    with 709 and flagged in the stream/container)."""
    path = str(out_path / f'{codec}_{"bt709" if tagged else "untagged"}.mp4')
    enc = {
        'mpeg4': ['-c:v', 'mpeg4', '-q:v', '2'],
        'h264': ['-c:v', 'libx264', '-crf', '12'],
    }[codec]
    color = (
        ['-vf', 'scale=out_color_matrix=bt709', '-colorspace', 'bt709',
         '-color_primaries', 'bt709', '-color_trc', 'bt709']
        if tagged else []
    )  # fmt: skip
    cmd = [
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        *color, *enc, '-pix_fmt', 'yuv420p', path,
    ]  # fmt: skip
    frames = np.broadcast_to(_pattern(), (N_FRAMES, H, W, 3))
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(
            f'ffmpeg encode failed (encoder not available?): {proc.stderr.decode()[-300:]}'
        )
    return path


def _encode_cv2(out_path) -> str:
    """The OpenCV mp4v writer (what many datasets were made with)."""
    cv2 = pytest.importorskip('cv2')
    path = str(out_path / 'cv2_mp4v.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip('cv2.VideoWriter could not open an mp4v file')
    bgr = np.ascontiguousarray(_pattern()[..., ::-1])
    for _ in range(N_FRAMES):
        writer.write(bgr)
    writer.release()
    return path


def _stream_colorspace(path: str) -> int:
    import av

    with av.open(path) as container:
        return int(container.streams.video[0].codec_context.colorspace or 0)


def _cuda_frame(torch, VideoFramesCuda, path: str, **kw) -> np.ndarray:
    return torch.from_dlpack(VideoFramesCuda(path, **kw)[CHECK_FRAME]).cpu().numpy()


def _assert_parity(cpu: np.ndarray, gpu: np.ndarray, codec: str, what: str) -> None:
    diff = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    if codec == 'h264':
        # NVDEC and libavcodec decode H.264 identically; with the same RGB
        # arithmetic the frames are bit-exact
        assert (
            diff.max() == 0
        ), f'{what}: CPU/CUDA differ (mean {diff.mean():.3f}, max {diff.max()})'
    else:
        # MPEG-4: IDCT implementations differ slightly
        assert (
            diff.mean() < 0.5 and diff.max() <= 12
        ), f'{what}: CPU/CUDA differ (mean {diff.mean():.3f}, max {diff.max()})'


class TestUntagged:
    @pytest.mark.parametrize('codec', ['mpeg4', 'h264'])
    def test_both_readers_recover_pattern_and_agree(self, tmp_path, codec):
        torch, VideoFramesCuda = _gpu()
        path = _encode(tmp_path, codec, tagged=False)
        assert _stream_colorspace(path) == 2, 'test premise: the stream carries no colour tag'

        cpu = VideoFrames(path)[CHECK_FRAME]
        gpu = _cuda_frame(torch, VideoFramesCuda, path)
        assert _recovery_error(cpu) < RECOVERY_TOL, f'CPU reader: {_recovery_error(cpu):.2f}'
        assert _recovery_error(gpu) < RECOVERY_TOL, f'CUDA reader: {_recovery_error(gpu):.2f}'
        _assert_parity(cpu, gpu, codec, f'untagged {codec}')

    def test_cv2_written_file(self, tmp_path):
        torch, VideoFramesCuda = _gpu()
        path = _encode_cv2(tmp_path)
        assert _stream_colorspace(path) == 2, 'test premise: cv2 writes no colour tag'
        cpu = VideoFrames(path)[CHECK_FRAME]
        gpu = _cuda_frame(torch, VideoFramesCuda, path)
        assert _recovery_error(cpu) < RECOVERY_TOL, f'CPU reader: {_recovery_error(cpu):.2f}'
        assert _recovery_error(gpu) < RECOVERY_TOL, f'CUDA reader: {_recovery_error(gpu):.2f}'
        _assert_parity(cpu, gpu, 'mpeg4', 'cv2 mp4v')

    def test_explicit_override_wins(self, tmp_path):
        """color_space='bt709' on an untagged (601) file gives the 709 reading:
        visibly wrong colours, differing from the CPU class by the 601/709 tint."""
        torch, VideoFramesCuda = _gpu()
        path = _encode(tmp_path, 'h264', tagged=False)
        cpu = VideoFrames(path)[CHECK_FRAME]
        gpu_auto = _cuda_frame(torch, VideoFramesCuda, path)
        gpu_709 = _cuda_frame(torch, VideoFramesCuda, path, color_space='bt709')
        assert _recovery_error(gpu_709) > 2 * RECOVERY_TOL, 'override had no effect'
        assert np.abs(gpu_709.astype(np.int16) - cpu.astype(np.int16)).mean() > 3.0
        assert np.array_equal(gpu_auto, cpu)
        # and the override to the right matrix is a no-op
        gpu_601 = _cuda_frame(torch, VideoFramesCuda, path, color_space='bt601')
        assert np.array_equal(gpu_601, cpu)


class TestTagged:
    @pytest.mark.parametrize('codec', ['mpeg4', 'h264'])
    def test_bt709_tag_honoured_by_both(self, tmp_path, codec):
        torch, VideoFramesCuda = _gpu()
        path = _encode(tmp_path, codec, tagged=True)
        assert _stream_colorspace(path) == 1, 'test premise: the stream is tagged BT.709'
        cpu = VideoFrames(path)[CHECK_FRAME]
        gpu = _cuda_frame(torch, VideoFramesCuda, path)
        assert _recovery_error(cpu) < RECOVERY_TOL, f'CPU reader: {_recovery_error(cpu):.2f}'
        assert _recovery_error(gpu) < RECOVERY_TOL, f'CUDA reader: {_recovery_error(gpu):.2f}'
        _assert_parity(cpu, gpu, codec, f'tagged {codec}')


class TestSwscaleArithmetic:
    """CPU-only: the constants and integer arithmetic the CUDA kernel uses
    reproduce libswscale's yuv420p → rgb24 output bit for bit."""

    def test_constants(self):
        from framepump._cuda.kernels import sws_yuv2rgb_coeffs

        assert sws_yuv2rgb_coeffs('bt601') == {
            'y_coff': 9539, 'y_off': 128, 'vr': 13075, 'ub': 16525, 'ug': -3209, 'vg': -6660,
        }  # fmt: skip
        assert sws_yuv2rgb_coeffs('bt709') == {
            'y_coff': 9539, 'y_off': 128, 'vr': 14686, 'ub': 17305, 'ug': -1747, 'vg': -4366,
        }  # fmt: skip
        assert sws_yuv2rgb_coeffs('bt601', full_range=True) == {
            'y_coff': 8192, 'y_off': 0, 'vr': 11485, 'ub': 14516, 'ug': -2819, 'vg': -5850,
        }  # fmt: skip

    @pytest.mark.parametrize(
        'matrix, avcol_spc',
        [('bt601', 2), ('bt601', 6), ('bt709', 1), ('smpte240m', 7), ('bt2020', 9), ('fcc', 4)],
    )
    @pytest.mark.parametrize('full_range', [False, True])
    def test_matches_libswscale(self, matrix, avcol_spc, full_range):
        av = pytest.importorskip('av')
        from framepump._cuda.kernels import sws_yuv2rgb_coeffs

        w, h = 256, 128
        rng = np.random.default_rng(1)
        y = rng.integers(0, 256, (h, w), np.uint8)
        u = rng.integers(0, 256, (h // 2, w // 2), np.uint8)
        v = rng.integers(0, 256, (h // 2, w // 2), np.uint8)
        frame = av.VideoFrame(w, h, 'yuv420p')
        for plane, arr in zip(frame.planes, (y, u, v)):
            padded = np.zeros((arr.shape[0], plane.line_size), np.uint8)
            padded[:, : arr.shape[1]] = arr
            plane.update(padded.tobytes())
        frame.colorspace = avcol_spc
        frame.color_range = 2 if full_range else 1
        ref = frame.to_ndarray(format='rgb24')

        c = sws_yuv2rgb_coeffs(matrix, full_range=full_range)
        yt = (y.astype(np.int64) << 3) - c['y_off']
        ut = (np.repeat(np.repeat(u, 2, 0), 2, 1).astype(np.int64) << 3) - 1024
        vt = (np.repeat(np.repeat(v, 2, 0), 2, 1).astype(np.int64) << 3) - 1024
        yc = (yt * c['y_coff']) >> 16
        r = yc + ((vt * c['vr']) >> 16)
        g = yc + (((ut * c['ug']) >> 16) + ((vt * c['vg']) >> 16))
        b = yc + ((ut * c['ub']) >> 16)
        ours = np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
        assert np.array_equal(ours, ref)
