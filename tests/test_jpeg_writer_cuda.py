"""Content round-trip tests for JpegVideoWriterCUDA.

The writer decodes JPEGs with nvJPEG and encodes with NVENC entirely on the
GPU. These tests verify that the decoded video matches the JPEG content — in
particular at heights that are not multiples of 16 (e.g. 1080), where the
NVENC buffer is padded and the plane offsets must follow the padded geometry.
Corruption magnitude is content-dependent, so the unaligned-height result is
compared against a same-content control at an aligned height rather than
against absolute thresholds.

GPU tests skip gracefully without an NVIDIA GPU. CUDA is never initialized at
collection time (pytest runs with --forked, and CUDA state created in the
collection process breaks in the forked children).
"""

import io

import numpy as np
import pytest


def _gpu_available():
    try:
        import ctypes

        ctypes.CDLL('libcuda.so.1')
        ctypes.CDLL('libnvidia-encode.so.1')
        from framepump.nvjpeg.bindings import _lib

        return _lib is not None
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()
needs_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE, reason='NVIDIA GPU with nvJPEG and NVENC required'
)

# PIL subsampling codes
PIL_444 = 0
PIL_420 = 2


def frame_rgb(width, height, idx):
    """Distinct per-frame content: moving gradients plus an index-dependent hue."""
    yy, xx = np.mgrid[0:height, 0:width]
    r = ((xx + idx * 37) % 256).astype(np.uint8)
    g = ((yy + idx * 53) % 256).astype(np.uint8)
    b = np.full((height, width), (idx * 41 + 20) % 256, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def make_jpeg(width, height, idx, subsampling):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame_rgb(width, height, idx)).save(
        buf, format='JPEG', quality=95, subsampling=subsampling
    )
    return buf.getvalue()


def write_video(out_path, jpegs, chroma=None, bframes=2):
    from framepump import JpegVideoWriterCUDA
    from framepump.encoder_config import EncoderConfig

    config = EncoderConfig(bframes=bframes)
    with JpegVideoWriterCUDA(
        str(out_path), fps=30, encoder_config=config, chroma=chroma
    ) as writer:
        for jpeg in jpegs:
            writer.append_data(jpeg)


def decode_video(out_path):
    from framepump import VideoFrames

    return list(VideoFrames(str(out_path)))


def mean_diff_vs_source(frames, width, height):
    diffs = [
        np.abs(f.astype(np.int16) - frame_rgb(width, height, i).astype(np.int16)).mean()
        for i, f in enumerate(frames)
    ]
    return float(np.mean(diffs))


CHROMA_CASES = [
    pytest.param(PIL_420, None, id='420-native'),
    pytest.param(PIL_444, '420', id='444-to-420'),
    pytest.param(PIL_444, '422', id='444-to-422'),
    pytest.param(PIL_444, None, id='444-native'),
]


@needs_gpu
@pytest.mark.parametrize('bframes', [0, 2])
@pytest.mark.parametrize('jpeg_subsampling,chroma', CHROMA_CASES)
def test_roundtrip_unaligned_height_matches_control(tmp_path, jpeg_subsampling, chroma, bframes):
    """Height 1080 (padded to 1088 for NVENC) must round-trip as cleanly as 1072."""
    width, n_frames = 640, 8
    results = {}
    for tag, height in (('unaligned', 1080), ('control', 1072)):
        out_path = tmp_path / f'{tag}.mp4'
        jpegs = [make_jpeg(width, height, i, jpeg_subsampling) for i in range(n_frames)]
        write_video(out_path, jpegs, chroma=chroma, bframes=bframes)
        frames = decode_video(out_path)
        assert len(frames) == n_frames
        assert frames[0].shape == (height, width, 3)
        results[tag] = mean_diff_vs_source(frames, width, height)

    control, unaligned = results['control'], results['unaligned']
    assert control < 12.0, f'control round-trip is broken: mean diff {control}'
    assert (
        unaligned <= control * 2.0 + 1.0
    ), f'unaligned-height round-trip diff {unaligned:.2f} vs control {control:.2f}'


@needs_gpu
def test_dimension_change_raises(tmp_path):
    from framepump import JpegVideoWriterCUDA

    out_path = tmp_path / 'out.mp4'
    with pytest.raises(ValueError, match='dimensions'):
        with JpegVideoWriterCUDA(str(out_path), fps=30) as writer:
            writer.append_data(make_jpeg(320, 240, 0, PIL_420))
            writer.append_data(make_jpeg(640, 480, 1, PIL_420))
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []


@needs_gpu
def test_subsampling_change_raises(tmp_path):
    from framepump import JpegVideoWriterCUDA

    out_path = tmp_path / 'out.mp4'
    with pytest.raises(ValueError, match='subsampling'):
        with JpegVideoWriterCUDA(str(out_path), fps=30) as writer:
            writer.append_data(make_jpeg(320, 240, 0, PIL_420))
            writer.append_data(make_jpeg(320, 240, 1, PIL_444))
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []


@needs_gpu
def test_abort_leaves_no_files(tmp_path):
    """An exception mid-write must abort: no final file, no stranded temp."""
    from framepump import JpegVideoWriterCUDA

    class Boom(Exception):
        pass

    out_path = tmp_path / 'out.mp4'
    with pytest.raises(Boom):
        with JpegVideoWriterCUDA(str(out_path), fps=30) as writer:
            for i in range(3):
                writer.append_data(make_jpeg(320, 240, i, PIL_420))
            raise Boom()
    assert list(tmp_path.iterdir()) == []


@needs_gpu
def test_odd_height_420_raises(tmp_path):
    """4:2:0 output cannot represent odd display dimensions."""
    from framepump import JpegVideoWriterCUDA

    with pytest.raises(ValueError, match='even'):
        with JpegVideoWriterCUDA(str(tmp_path / 'odd.mp4'), fps=30) as writer:
            writer.append_data(make_jpeg(320, 241, 0, PIL_420))
    assert list(tmp_path.iterdir()) == []


def test_invalid_chroma_rejected():
    framepump = pytest.importorskip('framepump')
    writer_cls = getattr(framepump, 'JpegVideoWriterCUDA', None)
    if writer_cls is None:
        pytest.skip('JpegVideoWriterCUDA not available (no CUDA)')
    with pytest.raises(ValueError, match='chroma'):
        writer_cls(chroma='422p')


@needs_gpu
class TestSequenceAbortOnException:
    def test_exception_in_sequence_context_leaves_no_file(self, tmp_path):
        from framepump import JpegVideoWriterCUDA

        out = tmp_path / 'aborted.mp4'
        writer = JpegVideoWriterCUDA(gpu=0)
        with pytest.raises(RuntimeError, match='simulated'):
            with writer.start_sequence(str(out), fps=30):
                writer.append_data(make_jpeg(320, 240, 0, PIL_420))
                writer.append_data(make_jpeg(320, 240, 1, PIL_420))
                raise RuntimeError('simulated failure')
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))

    def test_corrupt_later_jpeg_raises_value_error(self, tmp_path):
        from framepump import JpegVideoWriterCUDA

        out = tmp_path / 'out.mp4'
        writer = JpegVideoWriterCUDA(str(out), fps=30)
        writer.append_data(make_jpeg(320, 240, 0, PIL_420))
        with pytest.raises(ValueError, match='Could not parse JPEG data'):
            writer.append_data(b'\xff\xd8\xff\xe0garbage-not-a-jpeg')
        writer._abort()
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))

    def test_writer_usable_after_aborted_sequence(self, tmp_path):
        from framepump import JpegVideoWriterCUDA, VideoFrames

        aborted = tmp_path / 'aborted.mp4'
        good = tmp_path / 'good.mp4'
        writer = JpegVideoWriterCUDA(gpu=0)
        with pytest.raises(RuntimeError, match='simulated'):
            with writer.start_sequence(str(aborted), fps=30):
                writer.append_data(make_jpeg(320, 240, 0, PIL_420))
                raise RuntimeError('simulated failure')
        with writer.start_sequence(str(good), fps=30):
            for i in range(4):
                writer.append_data(make_jpeg(320, 240, i, PIL_420))
        writer.close()
        assert not aborted.exists()
        assert len(VideoFrames(str(good))) == 4


@needs_gpu
class TestFailedFirstFrame:
    def test_corrupt_first_jpeg_fails_cleanly(self, tmp_path):
        from framepump import JpegVideoWriterCUDA

        out = tmp_path / 'out.mp4'
        writer = JpegVideoWriterCUDA(str(out), fps=30)
        with pytest.raises(ValueError, match='Could not parse JPEG data'):
            writer.append_data(b'\xff\xd8\xff\xe0garbage-not-a-jpeg')
        # A retry must fail with a clear error, never ZeroDivisionError or
        # nonsense geometry comparisons against a never-accepted first frame.
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            writer.append_data(make_jpeg(320, 240, 0, PIL_420))
        assert not isinstance(excinfo.value, ZeroDivisionError)
        assert '0x0' not in str(excinfo.value)
        assert not out.exists()
        assert not list(tmp_path.glob('*.tmp_*'))
