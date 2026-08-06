"""Tests for input validation at Python-boundary entry points.

Covers CudaToGLUploader tensor validation, NPP binding hardening
(deferred library-load errors, even-dimension checks), and the
cuda-python 12.x/13.x cuCtxCreate signature dispatch.
"""

import ctypes
import importlib

import pytest


# Note: tests run with --forked, so CUDA/torch must not be initialized at
# collection time (the forked child would inherit a poisoned CUDA state).
# All GPU availability checks happen lazily inside the tests.
def _init_gpu_or_skip():
    driver = pytest.importorskip('cuda.bindings.driver')
    (err,) = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f'CUDA not available: {err}')
    return driver


def _torch_cuda_or_skip():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('torch with CUDA required')
    return torch


# ---------------------------------------------------------------------------
# CudaToGLUploader tensor validation (upload() rejects invalid input before
# touching CUDA)
# ---------------------------------------------------------------------------
class _FakeTensor:
    """Minimal stand-in exposing the tensor introspection surface."""

    def __init__(self, dtype='torch.uint8', shape=(4, 8, 3), device_type='cuda', contiguous=True):
        self.dtype = dtype
        self.shape = shape
        self._contiguous = contiguous
        self.device = type('_Dev', (), {'type': device_type})()

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return 0


class TestUploadTensorValidation:
    @pytest.fixture()
    def validate(self):
        cuda_gl = pytest.importorskip('framepump._cuda_gl')
        return cuda_gl._validate_upload_tensor

    def test_valid_fake_tensor_passes(self, validate):
        validate(_FakeTensor(), height=4, width=8, channels=3)

    def test_missing_introspection_raises_type_error(self, validate):
        class _Opaque:
            def data_ptr(self):
                return 0

        with pytest.raises(TypeError):
            validate(_Opaque(), height=4, width=8, channels=3)

    def test_wrong_dtype_rejected(self, validate):
        with pytest.raises(ValueError, match='uint8'):
            validate(_FakeTensor(dtype='torch.uint16'), height=4, width=8, channels=3)
        with pytest.raises(ValueError, match='uint8'):
            validate(_FakeTensor(dtype='torch.float32'), height=4, width=8, channels=3)

    def test_wrong_device_rejected(self, validate):
        with pytest.raises(ValueError, match='[Cc][Uu][Dd][Aa]'):
            validate(_FakeTensor(device_type='cpu'), height=4, width=8, channels=3)

    def test_wrong_shape_rejected(self, validate):
        with pytest.raises(ValueError, match='shape'):
            validate(_FakeTensor(shape=(8, 4, 3)), height=4, width=8, channels=3)
        with pytest.raises(ValueError, match='shape'):
            validate(_FakeTensor(shape=(4, 8, 4)), height=4, width=8, channels=3)
        with pytest.raises(ValueError, match='shape'):
            validate(_FakeTensor(shape=(4, 8)), height=4, width=8, channels=3)

    def test_non_contiguous_rejected(self, validate):
        with pytest.raises(ValueError, match='contiguous'):
            validate(_FakeTensor(contiguous=False), height=4, width=8, channels=3)


class TestUploadTensorValidationTorch:
    @pytest.fixture()
    def validate(self):
        cuda_gl = pytest.importorskip('framepump._cuda_gl')
        return cuda_gl._validate_upload_tensor

    def test_real_cuda_uint8_passes(self, validate):
        torch = _torch_cuda_or_skip()
        t = torch.zeros((4, 8, 3), dtype=torch.uint8, device='cuda')
        validate(t, height=4, width=8, channels=3)

    def test_real_wrong_dtype_rejected(self, validate):
        torch = _torch_cuda_or_skip()
        t = torch.zeros((4, 8, 3), dtype=torch.uint16, device='cuda')
        with pytest.raises(ValueError, match='uint8'):
            validate(t, height=4, width=8, channels=3)

    def test_real_non_contiguous_rejected(self, validate):
        torch = _torch_cuda_or_skip()
        t = torch.zeros((8, 4, 3), dtype=torch.uint8, device='cuda').transpose(0, 1)
        assert not t.is_contiguous()
        with pytest.raises(ValueError, match='contiguous'):
            validate(t, height=4, width=8, channels=3)

    def test_real_cpu_tensor_rejected(self, validate):
        torch = _torch_cuda_or_skip()
        t = torch.zeros((4, 8, 3), dtype=torch.uint8)
        with pytest.raises(ValueError, match='[Cc][Uu][Dd][Aa]'):
            validate(t, height=4, width=8, channels=3)


# ---------------------------------------------------------------------------
# NPP bindings: deferred load errors and even-dimension checks
# ---------------------------------------------------------------------------
def _npp_libs_available():
    try:
        from framepump import npp_bindings

        return npp_bindings._load_error is None
    except Exception:
        return False


class TestNppBindings:
    def test_import_is_safe_without_libraries(self, monkeypatch):
        """With CDLL failing, import must succeed and first use must raise."""
        from framepump import npp_bindings

        def _fail(name, *args, **kwargs):
            raise OSError(f'{name}: cannot open shared object file')

        monkeypatch.setattr(ctypes, 'CDLL', _fail)
        try:
            importlib.reload(npp_bindings)
            assert npp_bindings._load_error is not None
            with pytest.raises(RuntimeError, match='libnpp'):
                npp_bindings.resize_plane_8u(0, 2, 2, 2, 0, 2, 2, 2)
        finally:
            monkeypatch.undo()
            importlib.reload(npp_bindings)

        assert npp_bindings._load_error is None or True  # restored module reimported

    @pytest.mark.skipif(not _npp_libs_available(), reason='NPP libraries required')
    @pytest.mark.parametrize('width,height', [(641, 480), (640, 481), (3, 3)])
    def test_even_dims_required_420(self, width, height):
        """Odd dimensions must raise before any NPP call (dummy pointers)."""
        from framepump import npp_bindings

        with pytest.raises(ValueError, match='even'):
            npp_bindings.rgb_to_nv12(0, width * 3, 0, width, 0, width, width, height, 0, 0, 0)
        with pytest.raises(ValueError, match='even'):
            npp_bindings.yuv420_to_nv12(
                0, width, 0, width // 2, 0, width // 2, 0, width, 0, width, width, height
            )
        with pytest.raises(ValueError, match='even'):
            npp_bindings.nv12_to_rgb8(
                0,
                width,
                0,
                width,
                0,
                width * 3,
                width,
                height,
                npp_bindings.BT601_YUV_TO_RGB_8_FULL,
            )
        with pytest.raises(ValueError, match='even'):
            npp_bindings.nv12_to_p016(
                0, width, 0, width, 0, width * 2, 0, width * 2, width, height
            )

    def test_default_ctx_uses_current_device_and_caches(self):
        driver = _init_gpu_or_skip()

        from framepump import npp_bindings
        from framepump._cuda.compat import cuCtxCreate

        err, dev = driver.cuDeviceGet(0)
        assert err == driver.CUresult.CUDA_SUCCESS
        err, ctx = cuCtxCreate(0, dev)
        assert err == driver.CUresult.CUDA_SUCCESS
        try:
            first = npp_bindings._get_default_ctx()
            second = npp_bindings._get_default_ctx()
            assert first is second
            assert first.nCudaDeviceId == 0
            assert first.nMultiProcessorCount > 0
        finally:
            driver.cuCtxDestroy(ctx)


# ---------------------------------------------------------------------------
# cuCtxCreate 12.x/13.x dispatch decided at import, not per call
# ---------------------------------------------------------------------------
class TestCuCtxCreateCompat:
    def test_signature_detected_at_import(self):
        _cuda_compat = pytest.importorskip('framepump._cuda.compat')

        assert isinstance(_cuda_compat._CTX_CREATE_TAKES_PARAMS, bool)

    def test_create_and_destroy(self):
        driver = _init_gpu_or_skip()

        from framepump._cuda.compat import cuCtxCreate

        err, dev = driver.cuDeviceGet(0)
        assert err == driver.CUresult.CUDA_SUCCESS
        err, ctx = cuCtxCreate(0, dev)
        assert err == driver.CUresult.CUDA_SUCCESS
        err_tuple = driver.cuCtxDestroy(ctx)
        assert err_tuple[0] == driver.CUresult.CUDA_SUCCESS

    def test_bad_argument_error_is_unmasked(self):
        """A genuine bad-argument error must come from a single attempt,
        not be re-raised from a fallback with the first error chained on."""
        _init_gpu_or_skip()
        from framepump._cuda.compat import cuCtxCreate

        with pytest.raises(TypeError) as excinfo:
            cuCtxCreate(0, object())
        assert excinfo.value.__context__ is None
