"""CUDA context ownership convention tests.

framepump never leaves a different CUDA context current than it found:
components retain the device's primary context and push it only for the
duration of their own calls, owned buffers carry their owning context so
deleters can free from any thread, and every retain has exactly one release.
"""

import gc
import io
import threading
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / 'data'


def _cuda():
    """Import the driver and require a working CUDA device (lazy, fork-safe)."""
    driver = pytest.importorskip('cuda.bindings.driver')
    (err,) = driver.cuInit(0)
    if int(err) != 0:
        pytest.skip(f'CUDA not available: {err}')
    err, count = driver.cuDeviceGetCount()
    if int(err) != 0 or count == 0:
        pytest.skip('No CUDA device available')
    return driver


def _current_ctx(driver) -> int:
    err, ctx = driver.cuCtxGetCurrent()
    assert int(err) == 0
    return int(ctx) if ctx is not None else 0


def _make_jpeg(width=64, height=48, subsampling=2) -> bytes:
    Image = pytest.importorskip('PIL.Image')
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=95, subsampling=subsampling)
    return buf.getvalue()


class TestNvjpegContextNeutral:
    def test_simple_decoder_leaves_no_ctx(self):
        driver = _cuda()
        pytest.importorskip('framepump.nvjpeg')
        from framepump.nvjpeg import NvjpegDecoder

        assert _current_ctx(driver) == 0
        try:
            decoder = NvjpegDecoder(gpu=0)
        except ImportError:
            pytest.skip('nvJPEG not available')
        assert _current_ctx(driver) == 0, 'constructor left a context current'
        assert decoder.get_image_info(_make_jpeg())[:2] == (64, 48)
        assert _current_ctx(driver) == 0, 'get_image_info left a context current'
        decoder.close()
        assert _current_ctx(driver) == 0, 'close left a context current'

    def test_phased_decoder_leaves_no_ctx(self):
        driver = _cuda()
        pytest.importorskip('framepump.nvjpeg')
        from framepump.nvjpeg import NvjpegPhasedDecoder

        assert _current_ctx(driver) == 0
        try:
            decoder = NvjpegPhasedDecoder(gpu=0)
        except ImportError:
            pytest.skip('nvJPEG not available')
        assert _current_ctx(driver) == 0, 'constructor left a context current'
        decoder.parse(_make_jpeg())
        decoder.decode_host()
        assert _current_ctx(driver) == 0, 'CPU stages left a context current'
        decoder.close()
        assert _current_ctx(driver) == 0, 'close left a context current'

    def test_decoder_preserves_caller_ctx(self):
        driver = _cuda()
        pytest.importorskip('framepump.nvjpeg')
        from framepump.nvjpeg import NvjpegDecoder

        err, device = driver.cuDeviceGet(0)
        assert int(err) == 0
        err, ctx = driver.cuDevicePrimaryCtxRetain(device)
        assert int(err) == 0
        (err,) = driver.cuCtxSetCurrent(ctx)
        assert int(err) == 0
        try:
            with NvjpegDecoder(gpu=0) as decoder:
                decoder.get_image_info(_make_jpeg())
                assert _current_ctx(driver) == int(ctx)
            assert _current_ctx(driver) == int(ctx), 'close switched the caller context'
        finally:
            driver.cuCtxSetCurrent(None)
            driver.cuDevicePrimaryCtxRelease(device)


class TestWriterEncoderContextNeutral:
    def test_jpeg_writer_leaves_no_ctx(self, tmp_path):
        driver = _cuda()
        framepump = pytest.importorskip('framepump')
        if not hasattr(framepump, 'JpegVideoWriterCUDA'):
            pytest.skip('JpegVideoWriterCUDA not available')

        out = tmp_path / 'ctx.mp4'
        jpeg = _make_jpeg(256, 128)  # NVENC has a minimum frame size (~145x49)
        assert _current_ctx(driver) == 0
        writer = framepump.JpegVideoWriterCUDA(str(out), fps=30)
        for _ in range(3):
            writer.append_data(jpeg)
            assert _current_ctx(driver) == 0, 'append_data left a context current'
        writer.close()
        assert _current_ctx(driver) == 0, 'close left a context current'

        frames = framepump.VideoFrames(str(out))
        assert len(list(frames)) == 3

    def test_cuda_encoder_construct_close_leaves_no_ctx(self):
        driver = _cuda()
        try:
            from framepump.nvenc.cuda_encoder import NvencCudaEncoder
        except ImportError:
            pytest.skip('NVENC CUDA encoder not available')

        assert _current_ctx(driver) == 0
        try:
            # NVENC has a minimum frame size (~145x49)
            encoder = NvencCudaEncoder(256, 128, gpu=0)
        except Exception as e:
            pytest.skip(f'NVENC session not available: {e}')
        assert _current_ctx(driver) == 0, 'constructor left a context current'
        encoder.close()
        assert _current_ctx(driver) == 0, 'close left a context current'


class TestVideoFramesCuda:
    def test_hbd_iteration_ctx_neutral_without_torch(self):
        driver = _cuda()
        pytest.importorskip('PyNvVideoCodec')
        from framepump import VideoFramesCuda

        assert _current_ctx(driver) == 0
        frames = VideoFramesCuda(str(DATA_DIR / '10bit.mp4'), dtype=np.uint16)
        for i, _frame in enumerate(frames):
            if i >= 2:
                break
        frames.close()
        assert _current_ctx(driver) == 0, 'NPP pipeline left a context current'

    def test_hbd_content_matches_cpu(self):
        _cuda()
        pytest.importorskip('PyNvVideoCodec')
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch CUDA not available')
        from framepump import VideoFrames, VideoFramesCuda

        path = str(DATA_DIR / '10bit.mp4')
        cpu = [f for _, f in zip(range(3), VideoFrames(path, dtype=np.uint16))]

        frames = VideoFramesCuda(path, dtype=np.uint16)
        gpu = []
        for i in range(3):
            gpu.append(torch.from_dlpack(frames[i]).cpu().numpy())
        frames.close()

        for i, (a, b) in enumerate(zip(cpu, gpu)):
            assert a.shape == b.shape
            diff = np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()
            # GPU (NPP color twist) and CPU (swscale) round differently; a
            # wrong plane pitch would shear the image and blow far past this.
            assert diff < 1500, f'frame {i}: mean abs diff {diff:.0f} (16-bit scale)'

    def test_indexed_frame_freed_from_other_thread(self):
        driver = _cuda()
        pytest.importorskip('PyNvVideoCodec')
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch CUDA not available')
        from framepump import VideoFramesCuda

        frames = VideoFramesCuda(str(DATA_DIR / '10bit.mp4'), dtype=np.uint16)
        tensor = torch.from_dlpack(frames[2])
        frames.close()  # buffer must outlive the reader's own retain

        main_ctx = _current_ctx(driver)
        holder = [tensor]
        del tensor
        errors = []

        def drop():
            # The DLPack deleter runs here, on a thread with no current
            # context; it must push the buffer's owning context itself.
            try:
                holder.clear()
                gc.collect()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t = threading.Thread(target=drop)
        t.start()
        t.join(timeout=30)
        assert not t.is_alive(), 'deleter thread hung'
        assert not errors, f'deleter raised: {errors}'
        assert _current_ctx(driver) == main_ctx, 'main-thread context changed'
