"""Color-correctness verification across decode paths.

Colors must come out demonstrably correct on every path, and high-bit-depth
content must never be quantized mid-pipeline. Clips are generated on the fly
with *explicitly flagged* color properties (matrix, range, transfer), and the
ffmpeg CLI serves as the reference decoder. Two kinds of assertion:

- every path agrees with the reference within its documented tolerance
  (CPU: exact; VideoFramesCuda: a few counts), and
- the flags demonstrably *matter*: the same encoded pixels flagged BT.601 vs
  BT.709 must decode to different RGB — catching any path that silently
  hardcodes one matrix.

CPU cases run wherever ffmpeg is available; GPU cases skip without hardware.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from framepump import VideoFrames

FFMPEG = shutil.which('ffmpeg')

pytestmark = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg CLI not available')

# NVDEC requires at least ~144x144 for HEVC, so stay comfortably above.
W, H, N_FRAMES = 192, 144, 4


def _test_pattern() -> np.ndarray:
    """Saturated patches + ramps: sensitive to matrix, range, and bit depth."""
    img = np.zeros((H, W, 3), np.uint8)
    patches = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 255), (128, 128, 128),
    ]  # fmt: skip
    pw = W // len(patches)
    for i, color in enumerate(patches):
        img[: H // 2, i * pw : (i + 1) * pw] = color
    ramp = np.linspace(0, 255, W, dtype=np.uint8)
    img[H // 2 :] = ramp[None, :, None]
    return img


def _test_pattern16() -> np.ndarray:
    """16-bit variant with a fine 2D gradient: genuinely more than 256 levels,
    so a hidden 8-bit stage anywhere in a 10-bit pipeline is detectable."""
    img = _test_pattern().astype(np.uint16) * 257
    yy, xx = np.mgrid[0 : H // 2, 0:W]
    n = (H // 2) * W
    fine = ((yy * W + xx) * (65535 // n)).astype(np.uint16)
    img[H // 2 :] = fine[..., None]
    return img


def _encode_clip(
    out_path: Path,
    *,
    matrix: str,
    color_range: str = 'tv',
    bit_depth: int = 8,
    transfer: str | None = None,
    primaries: str | None = None,
    convert_matrix: str | None = None,
) -> None:
    """Encode the test pattern with explicit color flags.

    ``convert_matrix`` sets the matrix actually used for the RGB→YUV
    conversion; ``matrix`` sets what the stream is *flagged* as. They differ
    only in the flags-matter tests.
    """
    if convert_matrix is None:
        convert_matrix = matrix
    # The scale filter names matrices differently than the stream-flag options.
    vf_matrix = {'bt2020nc': 'bt2020', 'smpte170m': 'bt601'}.get(convert_matrix, convert_matrix)
    rng_flag = 'limited' if color_range == 'tv' else 'full'
    vf = f'scale=out_color_matrix={vf_matrix}:out_range={rng_flag}'
    if bit_depth == 8:
        codec = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        in_fmt, pattern = 'rgb24', _test_pattern()
    else:
        codec = ['-c:v', 'libx265', '-pix_fmt', 'yuv420p10le', '-x265-params', 'log-level=none']
        in_fmt, pattern = 'rgb48le', _test_pattern16()
    cmd = [
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', in_fmt, '-s', f'{W}x{H}', '-r', '30', '-i', '-',
        '-vf', vf,
        *codec,
        '-crf', '10',
        '-colorspace', matrix,
        '-color_range', color_range,
        '-color_primaries', primaries or ('bt709' if matrix != 'bt2020nc' else 'bt2020'),
        '-color_trc', transfer or 'bt709',
        str(out_path),
    ]  # fmt: skip
    frames = np.broadcast_to(pattern, (N_FRAMES, H, W, 3))
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f'ffmpeg encode failed (codec not available?): {proc.stderr.decode()[-300:]}')


def _reference_decode(path: Path, *, bits: int = 8) -> np.ndarray:
    """Decode frame 0 to RGB with the ffmpeg CLI (the color ground truth)."""
    pix_fmt = 'rgb24' if bits == 8 else 'rgb48le'
    cmd = [
        FFMPEG, '-hide_banner', '-loglevel', 'error',
        '-i', str(path), '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-',
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode()[-300:]
    dtype = np.uint8 if bits == 8 else np.uint16
    return np.frombuffer(proc.stdout, dtype).reshape(H, W, 3)


def _gpu_available(path) -> bool:
    try:
        VideoFrames(str(path), gpu=True)[0]
        return True
    except Exception:
        return False


def _cuda_class_frame(path, *, dtype=np.uint8):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available for torch')
    pytest.importorskip('PyNvVideoCodec')
    from framepump import VideoFramesCuda

    v = VideoFramesCuda(str(path), dtype=dtype)
    try:
        frame = torch.from_dlpack(v[0])
        return frame.cpu().numpy()
    finally:
        v.close()


def _assert_close(a, b, *, max_abs, what):
    a16, b16 = a.astype(np.int32), b.astype(np.int32)
    diff = np.abs(a16 - b16)
    assert diff.max() <= max_abs, (
        f'{what}: max diff {diff.max()} > {max_abs} '
        f'(mean {diff.mean():.2f}, p99 {np.percentile(diff, 99):.1f})'
    )


def _patch_means(img: np.ndarray) -> np.ndarray:
    """Mean RGB of each color patch's central core.

    Chroma-subsampled edges differ legitimately between converters (NPP
    replicates, swscale interpolates), so matrix/range correctness is judged
    on patch interiors, where a wrong matrix shifts values by thousands
    (16-bit scale) while converter rounding stays within a few hundred.
    """
    pw = W // 8
    means = []
    for i in range(8):
        core = img[H // 8 : H // 2 - H // 8, i * pw + pw // 4 : (i + 1) * pw - pw // 4]
        means.append(core.reshape(-1, img.shape[-1]).mean(0))
    return np.array(means)


def _assert_patch_means_close(a, b, *, max_abs, what):
    diff = np.abs(_patch_means(a.astype(np.int64)) - _patch_means(b.astype(np.int64)))
    assert diff.max() <= max_abs, f'{what}: patch-mean diff {diff.max():.0f} > {max_abs}'


class TestCpuPathMatchesReference:
    @pytest.mark.parametrize('matrix', ['bt601', 'bt709'])
    def test_matrix_honored_8bit(self, tmp_path, matrix):
        ff_matrix = 'smpte170m' if matrix == 'bt601' else 'bt709'
        clip = tmp_path / f'{matrix}.mp4'
        _encode_clip(clip, matrix=ff_matrix)
        ref = _reference_decode(clip)
        ours = VideoFrames(str(clip))[0]
        _assert_close(ours, ref, max_abs=1, what=f'CPU decode vs ffmpeg CLI ({matrix})')

    def test_full_range_honored(self, tmp_path):
        clip = tmp_path / 'full_range.mp4'
        _encode_clip(clip, matrix='bt709', color_range='pc')
        ref = _reference_decode(clip)
        ours = VideoFrames(str(clip))[0]
        _assert_close(ours, ref, max_abs=1, what='CPU decode vs ffmpeg CLI (full range)')

    def test_flags_matter_601_vs_709(self, tmp_path):
        """Same YUV pixels, different flag → decodes must differ visibly."""
        clip601 = tmp_path / 'flag601.mp4'
        clip709 = tmp_path / 'flag709.mp4'
        _encode_clip(clip601, matrix='smpte170m', convert_matrix='bt601')
        _encode_clip(clip709, matrix='bt709', convert_matrix='bt601')
        a = VideoFrames(str(clip601))[0].astype(np.int32)
        b = VideoFrames(str(clip709))[0].astype(np.int32)
        assert np.abs(a - b).max() > 5, 'decoder ignored the colorspace flag'


class TestBitDepthPreservation:
    def test_10bit_no_hidden_8bit_stage(self, tmp_path):
        clip = tmp_path / 'grad10.mp4'
        _encode_clip(clip, matrix='bt709', bit_depth=10)
        ours = VideoFrames(str(clip), dtype=np.uint16)[0]
        levels = len(np.unique(ours))
        assert levels > 256, (
            f'only {levels} distinct uint16 levels — a hidden 8-bit stage '
            f'would cap this at 256'
        )
        ref = _reference_decode(clip, bits=16)
        _assert_close(ours, ref, max_abs=257, what='CPU 10-bit decode vs ffmpeg CLI')

    def test_cuda_class_10bit_no_hidden_8bit_stage(self, tmp_path):
        clip = tmp_path / 'grad10c.mp4'
        _encode_clip(clip, matrix='bt709', bit_depth=10)
        ours = _cuda_class_frame(clip, dtype=np.uint16)
        levels = len(np.unique(ours))
        assert levels > 256, f'only {levels} distinct uint16 levels on the NPP path'

    def test_cuda_class_10bit_resized_keeps_depth(self, tmp_path):
        """The GPU resize stage must not quantize high-bit-depth data."""
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available for torch')
        pytest.importorskip('PyNvVideoCodec')
        from framepump import VideoFramesCuda

        clip = tmp_path / 'grad10rs.mp4'
        _encode_clip(clip, matrix='bt709', bit_depth=10)
        v = VideoFramesCuda(str(clip), dtype=np.uint16).resized((H // 2, W // 2))
        try:
            ours = torch.from_dlpack(v[0]).cpu().numpy()
        finally:
            v.close()
        levels = len(np.unique(ours))
        assert levels > 256, f'only {levels} distinct uint16 levels after GPU resize'


class TestGpuPathsMatchCpu:
    @pytest.mark.parametrize('matrix', ['smpte170m', 'bt709'])
    def test_hwaccel_bit_identical(self, tmp_path, matrix):
        clip = tmp_path / f'hw_{matrix}.mp4'
        _encode_clip(clip, matrix=matrix)
        if not _gpu_available(clip):
            pytest.skip('NVDEC GPU decoding not available')
        cpu = VideoFrames(str(clip))[0]
        gpu = VideoFrames(str(clip), gpu=True)[0]
        np.testing.assert_array_equal(cpu, gpu, err_msg=f'gpu=True not bit-identical ({matrix})')

    @pytest.mark.parametrize('matrix', ['smpte170m', 'bt709'])
    def test_cuda_class_close_to_cpu(self, tmp_path, matrix):
        clip = tmp_path / f'cc_{matrix}.mp4'
        _encode_clip(clip, matrix=matrix)
        cpu = VideoFrames(str(clip))[0]
        ours = _cuda_class_frame(clip)
        _assert_close(ours, cpu, max_abs=4, what=f'VideoFramesCuda vs CPU ({matrix})')

    def test_cuda_class_flags_matter(self, tmp_path):
        """VideoFramesCuda must pick its matrix from the stream flags."""
        clip601 = tmp_path / 'cflag601.mp4'
        clip709 = tmp_path / 'cflag709.mp4'
        _encode_clip(clip601, matrix='smpte170m', convert_matrix='bt601')
        _encode_clip(clip709, matrix='bt709', convert_matrix='bt601')
        a = _cuda_class_frame(clip601).astype(np.int32)
        b = _cuda_class_frame(clip709).astype(np.int32)
        assert np.abs(a - b).max() > 5, 'VideoFramesCuda ignored the colorspace flag'


class TestHdrContent:
    def test_cpu_hdr_pq_bt2020(self, tmp_path):
        """PQ/BT.2020 10-bit decode: full precision, matrix honored (vs CLI)."""
        clip = tmp_path / 'hdr.mp4'
        _encode_clip(
            clip, matrix='bt2020nc', bit_depth=10, transfer='smpte2084', primaries='bt2020'
        )
        ours = VideoFrames(str(clip), dtype=np.uint16)[0]
        assert len(np.unique(ours)) > 256
        ref = _reference_decode(clip, bits=16)
        _assert_close(ours, ref, max_abs=257, what='CPU HDR decode vs ffmpeg CLI')

    def test_cuda_class_hdr_matrix(self, tmp_path):
        """BT.2020-nc content through the NPP path must use the 2020 matrix."""
        clip = tmp_path / 'hdrc.mp4'
        _encode_clip(
            clip, matrix='bt2020nc', bit_depth=10, transfer='smpte2084', primaries='bt2020'
        )
        cpu = VideoFrames(str(clip), dtype=np.uint16)[0]
        ours = _cuda_class_frame(clip, dtype=np.uint16)
        _assert_patch_means_close(ours, cpu, max_abs=700, what='VideoFramesCuda HDR vs CPU')


class TestCudaClassHighBitDepthMatrices:
    """The NPP path must pick its twist matrix and range from the stream."""

    @pytest.mark.parametrize('matrix', ['bt709', 'smpte240m'])
    def test_matrix_close_to_cpu(self, tmp_path, matrix):
        clip = tmp_path / f'hbd_{matrix}.mp4'
        _encode_clip(clip, matrix=matrix, bit_depth=10)
        cpu = VideoFrames(str(clip), dtype=np.uint16)[0]
        ours = _cuda_class_frame(clip, dtype=np.uint16)
        _assert_patch_means_close(
            ours, cpu, max_abs=700, what=f'VideoFramesCuda 10-bit {matrix} vs CPU'
        )

    def test_full_range_honored(self, tmp_path):
        clip = tmp_path / 'hbd_full.mp4'
        _encode_clip(clip, matrix='bt709', bit_depth=10, color_range='pc')
        cpu = VideoFrames(str(clip), dtype=np.uint16)[0]
        ours = _cuda_class_frame(clip, dtype=np.uint16)
        _assert_patch_means_close(
            ours, cpu, max_abs=700, what='VideoFramesCuda 10-bit full-range vs CPU'
        )

    def test_flags_matter_10bit(self, tmp_path):
        """Same 10-bit YUV flagged 601 vs 709 must decode differently."""
        clip601 = tmp_path / 'hbd_flag601.mp4'
        clip709 = tmp_path / 'hbd_flag709.mp4'
        _encode_clip(clip601, matrix='smpte170m', convert_matrix='bt601', bit_depth=10)
        _encode_clip(clip709, matrix='bt709', convert_matrix='bt601', bit_depth=10)
        a = _cuda_class_frame(clip601, dtype=np.uint16)
        b = _cuda_class_frame(clip709, dtype=np.uint16)
        diff = np.abs(_patch_means(a.astype(np.int64)) - _patch_means(b.astype(np.int64)))
        assert diff.max() > 1000, 'VideoFramesCuda NPP path ignored the colorspace flag'
