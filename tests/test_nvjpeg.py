"""Tests for the nvJPEG decoders.

Most tests require an NVIDIA GPU with the nvJPEG library; they are skipped
gracefully otherwise. The input-validation tests run without a GPU.
"""

import io

import numpy as np
import pytest


def _nvjpeg_available():
    """Check nvJPEG and CUDA driver library presence.

    Deliberately does NOT initialize CUDA: tests run under pytest --forked,
    and CUDA state created in the collection process would break in the
    forked children. Actual device initialization happens inside each test.
    """
    try:
        import ctypes

        ctypes.CDLL('libcuda.so.1')
        import cuda.bindings  # noqa: F401

        from framepump.nvjpeg.bindings import _lib

        return _lib is not None
    except Exception:
        return False


NVJPEG_AVAILABLE = _nvjpeg_available()

# Chroma subsampling constants (mirror nvjpeg/bindings.py)
CSS_444 = 0
CSS_420 = 2


def make_jpeg(width, height, seed, subsampling, noise=False):
    """Create a JPEG with distinct, seed-dependent content.

    Args:
        subsampling: PIL subsampling code (0 = 4:4:4, 2 = 4:2:0).
        noise: Random noise content (large Huffman payload, slow transfer).
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    if noise:
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        arr = np.stack(
            [
                ((xx * (seed + 1)) % 256).astype(np.uint8),
                ((yy * (seed + 2)) % 256).astype(np.uint8),
                np.full((height, width), (seed * 40) % 256, dtype=np.uint8),
            ],
            axis=-1,
        )
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=95, subsampling=subsampling)
    return buf.getvalue()


def _cuda_check(result):
    err = result[0]
    assert int(err) == 0, f'CUDA call failed: {err}'
    return result[1] if len(result) == 2 else None


def _plane_shapes(width, height, subsampling):
    if subsampling == CSS_420:
        return (height, width), (height // 2, width // 2), (height // 2, width // 2)
    return (height, width), (height, width), (height, width)


class _PhasedHarness:
    """Allocates output buffers and runs phased decodes on one stream."""

    _BUSY_BYTES = 256 * 1024 * 1024

    def __init__(self):
        from cuda.bindings import driver

        from framepump.nvjpeg import NvjpegPhasedDecoder

        self.driver = driver
        self.decoder = NvjpegPhasedDecoder(gpu=0)
        self.stream = int(_cuda_check(driver.cuStreamCreate(0)))
        self.dev_ptrs = []
        self._busy_ptr = None

    def make_stream_busy(self):
        """Enqueue tens of ms of work so subsequent stream work stays pending.

        This widens the race window: an async transfer enqueued behind this
        work is still pending while the CPU moves on to the next frame.
        """
        if self._busy_ptr is None:
            self._busy_ptr = int(_cuda_check(self.driver.cuMemAlloc(self._BUSY_BYTES)))
            self.dev_ptrs.append(self._busy_ptr)
        for _ in range(200):
            _cuda_check(
                self.driver.cuMemsetD8Async(self._busy_ptr, 0, self._BUSY_BYTES, self.stream)
            )

    def alloc_planes(self, width, height, subsampling):
        shapes = _plane_shapes(width, height, subsampling)
        ptrs = []
        for h, w in shapes:
            ptr = int(_cuda_check(self.driver.cuMemAlloc(h * w)))
            ptrs.append(ptr)
            self.dev_ptrs.append(ptr)
        return ptrs, shapes

    def decode(self, jpeg, sync_per_frame, busy_stream=False):
        """Run one full phased decode; returns (ptrs, shapes) of the output."""
        if busy_stream:
            self.make_stream_busy()
        width, height, subsampling = self.decoder.parse(jpeg)
        ptrs, shapes = self.alloc_planes(width, height, subsampling)
        pitches = [shape[1] for shape in shapes]
        self.decoder.decode_host()
        self.decoder.decode_transfer(self.stream)
        self.decoder.decode_device(*ptrs, *pitches, self.stream)
        if sync_per_frame:
            _cuda_check(self.driver.cuStreamSynchronize(self.stream))
        return ptrs, shapes

    def download(self, ptrs, shapes):
        planes = []
        for ptr, (h, w) in zip(ptrs, shapes):
            host = np.empty((h, w), dtype=np.uint8)
            _cuda_check(self.driver.cuMemcpyDtoH(host, ptr, h * w))
            planes.append(host)
        return planes

    def close(self):
        _cuda_check(self.driver.cuStreamSynchronize(self.stream))
        for ptr in self.dev_ptrs:
            self.driver.cuMemFree(ptr)
        self.driver.cuStreamDestroy(self.stream)
        self.decoder.close()


def _run_phased(jpegs, sync_per_frame, busy_stream=False):
    """Decode all JPEGs with one decoder; returns per-frame YUV planes."""
    harness = _PhasedHarness()
    try:
        outputs = [harness.decode(jpeg, sync_per_frame, busy_stream) for jpeg in jpegs]
        _cuda_check(harness.driver.cuStreamSynchronize(harness.stream))
        return [harness.download(ptrs, shapes) for ptrs, shapes in outputs]
    finally:
        harness.close()


@pytest.mark.skipif(not NVJPEG_AVAILABLE, reason='nvJPEG or CUDA device not available')
class TestPhasedPipelined:
    """The documented pipelining pattern must give identical results."""

    def test_pipelined_matches_sequential_444(self):
        jpegs = [make_jpeg(640, 480, seed, subsampling=0) for seed in range(6)]
        reference = _run_phased(jpegs, sync_per_frame=True)
        pipelined = _run_phased(jpegs, sync_per_frame=False)
        for i, (ref_planes, pipe_planes) in enumerate(zip(reference, pipelined)):
            for name, ref, pipe in zip('YUV', ref_planes, pipe_planes):
                np.testing.assert_array_equal(pipe, ref, err_msg=f'frame {i} plane {name} differs')

    def test_pipelined_matches_sequential_noise(self):
        # Large noisy JPEGs: slow host decode and big transfers widen the
        # window in which unsynchronized buffer reuse corrupts frames.
        jpegs = [make_jpeg(1280, 960, seed, subsampling=0, noise=True) for seed in range(8)]
        reference = _run_phased(jpegs, sync_per_frame=True)
        pipelined = _run_phased(jpegs, sync_per_frame=False)
        for i, (ref_planes, pipe_planes) in enumerate(zip(reference, pipelined)):
            for name, ref, pipe in zip('YUV', ref_planes, pipe_planes):
                np.testing.assert_array_equal(pipe, ref, err_msg=f'frame {i} plane {name} differs')

    def test_pipelined_matches_sequential_420(self):
        jpegs = [make_jpeg(640, 480, seed, subsampling=2) for seed in range(5)]
        reference = _run_phased(jpegs, sync_per_frame=True)
        pipelined = _run_phased(jpegs, sync_per_frame=False)
        for i, (ref_planes, pipe_planes) in enumerate(zip(reference, pipelined)):
            for name, ref, pipe in zip('YUV', ref_planes, pipe_planes):
                np.testing.assert_array_equal(pipe, ref, err_msg=f'frame {i} plane {name} differs')

    def test_pipelined_with_busy_stream(self):
        # Adversarial scheduling: each frame's async transfer is enqueued
        # behind bulk stream work, so it is still pending while the next
        # frame's parse/host decode runs. Unsynchronized buffer reuse
        # corrupts frames under these conditions.
        jpegs = [make_jpeg(640, 480, seed, subsampling=0) for seed in range(6)]
        reference = _run_phased(jpegs, sync_per_frame=True)
        pipelined = _run_phased(jpegs, sync_per_frame=False, busy_stream=True)
        for i, (ref_planes, pipe_planes) in enumerate(zip(reference, pipelined)):
            for name, ref, pipe in zip('YUV', ref_planes, pipe_planes):
                np.testing.assert_array_equal(pipe, ref, err_msg=f'frame {i} plane {name} differs')

    def test_parse_metadata_alternating(self):
        """Parsed dimensions/subsampling stay correct across slot reuse."""
        from framepump.nvjpeg import NvjpegPhasedDecoder

        specs = [(640, 480, 0), (320, 240, 2), (512, 384, 0), (256, 128, 2), (640, 480, 2)]
        jpegs = [make_jpeg(w, h, i, subsampling=s) for i, (w, h, s) in enumerate(specs)]
        with NvjpegPhasedDecoder(gpu=0) as decoder:
            for (w, h, s), jpeg in zip(specs, jpegs):
                width, height, subsampling = decoder.parse(jpeg)
                expected_css = CSS_420 if s == 2 else CSS_444
                assert (width, height, subsampling) == (w, h, expected_css)
                assert decoder.parsed_width == w
                assert decoder.parsed_height == h
                assert decoder.parsed_subsampling == expected_css


@pytest.mark.skipif(not NVJPEG_AVAILABLE, reason='nvJPEG or CUDA device not available')
class TestSimpleDecoder:
    """NvjpegDecoder simple API."""

    def test_decode_yuv_matches_pil(self):
        from cuda.bindings import driver
        from PIL import Image

        from framepump.nvjpeg import NvjpegDecoder

        jpeg = make_jpeg(640, 480, 3, subsampling=0)
        pil_yuv = np.asarray(Image.open(io.BytesIO(jpeg)).convert('YCbCr'))

        with NvjpegDecoder(gpu=0) as decoder:
            width, height, _, subsampling = decoder.get_image_info(jpeg)
            assert (width, height, subsampling) == (640, 480, CSS_444)
            ptrs = [int(_cuda_check(driver.cuMemAlloc(height * width))) for _ in range(3)]
            try:
                decoder.decode_yuv_into(jpeg, *ptrs, y_pitch=width)
                _cuda_check(driver.cuCtxSynchronize())
                planes = []
                for ptr in ptrs:
                    host = np.empty((height, width), dtype=np.uint8)
                    _cuda_check(driver.cuMemcpyDtoH(host, ptr, height * width))
                    planes.append(host)
            finally:
                for ptr in ptrs:
                    driver.cuMemFree(ptr)

        for i, name in enumerate('YUV'):
            diff = np.abs(planes[i].astype(np.int16) - pil_yuv[..., i].astype(np.int16))
            assert diff.mean() < 2.0, f'{name} plane mean abs diff {diff.mean():.2f}'

    def test_ndarray_input(self):
        from framepump.nvjpeg import NvjpegDecoder

        jpeg = make_jpeg(320, 240, 5, subsampling=0)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        with NvjpegDecoder(gpu=0) as decoder:
            assert decoder.get_image_info(arr)[:2] == (320, 240)


class TestInputValidation:
    """JPEG input validation happens before any native call (L13)."""

    @pytest.fixture(autouse=True)
    def _need_module(self):
        pytest.importorskip('cuda.bindings')

    def test_wrong_dtype_rejected(self):
        from framepump.nvjpeg.decoder import _get_data_ptr_and_size

        with pytest.raises(ValueError, match='uint8'):
            _get_data_ptr_and_size(np.zeros(100, dtype=np.float32))

    def test_non_contiguous_rejected(self):
        from framepump.nvjpeg.decoder import _get_data_ptr_and_size

        arr = np.zeros(100, dtype=np.uint8)[::2]
        with pytest.raises(ValueError, match='contiguous'):
            _get_data_ptr_and_size(arr)

    def test_bytes_accepted(self):
        from framepump.nvjpeg.decoder import _get_data_ptr_and_size

        ptr, size = _get_data_ptr_and_size(b'\xff\xd8\xff\xe0')
        assert size == 4
        assert ptr is not None

    def test_multidim_uint8_size_is_bytes(self):
        from framepump.nvjpeg.decoder import _get_data_ptr_and_size

        arr = np.zeros((16, 16), dtype=np.uint8)
        ptr, size = _get_data_ptr_and_size(arr)
        assert size == 256
