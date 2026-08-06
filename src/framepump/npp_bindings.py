"""ctypes bindings for NPP (NVIDIA Performance Primitives) color conversion."""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_size_t, c_uint, c_uint16, c_void_p, Structure

# ---------------------------------------------------------------------------
# Libraries (import stays safe without them; first use raises, see _require_npp)
# ---------------------------------------------------------------------------
try:
    _nppicc = ctypes.CDLL('libnppicc.so')  # color conversion
    _nppidei = ctypes.CDLL('libnppidei.so')  # data exchange and initialization
    _nppial = ctypes.CDLL('libnppial.so')  # arithmetic and logical
    _nppig = ctypes.CDLL('libnppig.so')  # geometry transforms (resize)
    _load_error: OSError | None = None
except OSError as _e:
    _nppicc = _nppidei = _nppial = _nppig = None
    _load_error = _e


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
class NppiSize(Structure):
    """NPP size structure."""

    _fields_ = [
        ('width', c_int),
        ('height', c_int),
    ]


class NppiRect(Structure):
    """NPP rectangle structure (ROI for resize)."""

    _fields_ = [
        ('x', c_int),
        ('y', c_int),
        ('width', c_int),
        ('height', c_int),
    ]


class NppStreamContext(Structure):
    """NPP stream context (passed by value to _Ctx functions)."""

    _fields_ = [
        ('hStream', c_void_p),
        ('nCudaDeviceId', c_int),
        ('nMultiProcessorCount', c_int),
        ('nMaxThreadsPerMultiProcessor', c_int),
        ('nMaxThreadsPerBlock', c_int),
        ('nSharedMemPerBlock', c_size_t),
        ('nCudaDevAttrComputeCapabilityMajor', c_int),
        ('nCudaDevAttrComputeCapabilityMinor', c_int),
        ('nStreamFlags', c_uint),
        ('nReserved0', c_int),
    ]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NPP_SUCCESS = 0
# Interpolation modes from nppdefs.h. SUPER (area averaging) requires both
# scale factors to be < 1 and returns NPP_RESIZE_FACTOR_ERROR otherwise.
NPPI_INTER_LINEAR = 2
NPPI_INTER_SUPER = 8
NPPI_INTER_LANCZOS = 16

# Luma coefficients (Kr, Kb) per ITU-T H.273; mirrors FFmpeg's authoritative
# table in libavutil/csp.c. Every colorspace that is a plain matrix reduces to
# one of these five coefficient pairs (AVCOL_SPC_BT470BG and _SMPTE170M share
# the classic BT.601 numbers). The remaining enum entries (BT2020_CL, ICtCp,
# SMPTE 2085, YCgCo, chroma-derived) are not expressible as a single color
# twist and are not supported here.
LUMA_COEFFICIENTS: dict[str, tuple[float, float]] = {
    'bt709': (0.2126, 0.0722),
    'bt601': (0.299, 0.114),
    'fcc': (0.30, 0.11),
    'smpte240m': (0.212, 0.087),
    'bt2020': (0.2627, 0.0593),  # non-constant luminance
}


def make_yuv_to_rgb_twist(
    matrix: str, *, full_range: bool = False, bits: int = 16
) -> list[list[float]]:
    """Build a 3x4 NPP color-twist matrix for YUV→RGB conversion.

    Twist format for nppiColorTwist32f:
      dst[i] = M[i][0]*Y + M[i][1]*Cb + M[i][2]*Cr + M[i][3]
    on ``bits``-scaled samples (Cb/Cr centered at half scale). Limited range
    expands studio swing (luma × 255/219 with a 16-per-8-bits offset, chroma
    × 255/224); full range converts as-is.

    Args:
        matrix: A key of ``LUMA_COEFFICIENTS``.
        full_range: Whether the source YUV uses full (JPEG) range.
        bits: Sample scale of the input/output (8 or 16).
    """
    kr, kb = LUMA_COEFFICIENTS[matrix]
    kg = 1.0 - kr - kb
    coeffs = (
        (0.0, 2.0 * (1.0 - kr)),  # R: Cb, Cr
        (-2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg),  # G
        (2.0 * (1.0 - kb), 0.0),  # B
    )
    if full_range:
        y_scale, c_scale, y_offset = 1.0, 1.0, 0.0
    else:
        y_scale, c_scale = 255.0 / 219.0, 255.0 / 224.0
        y_offset = 16.0 * (1 << (bits - 8))
    half = float((1 << bits) // 2)
    rows = []
    for c_cb, c_cr in coeffs:
        b, c = c_cb * c_scale, c_cr * c_scale
        rows.append([y_scale, b, c, -(y_scale * y_offset + b * half + c * half)])
    return rows


# BT.709 / BT.601 limited-range YUV→RGB color twist matrices (16-bit scale).
#
# Standard twist format for nppiColorTwist32f:
#   dst[i] = M[i][0]*Y + M[i][1]*Cb + M[i][2]*Cr + M[i][3]
#
# The 4th column absorbs the source offsets (Y-16*256, Cb-128*256, Cr-128*256):
#   M[i][3] = -(M[i][0]*4096 + M[i][1]*32768 + M[i][2]*32768)

BT709_YUV_TO_RGB_16: list[list[float]] = [
    [1.164384, 0.000000, 1.792741, -63513.853952],
    [1.164384, -0.213249, -0.532909, 19680.788480],
    [1.164384, 2.112402, 0.000000, -73988.505600],
]

BT601_YUV_TO_RGB_16: list[list[float]] = [
    [1.164384, 0.000000, 1.596027, -57067.929600],
    [1.164384, -0.391762, -0.812968, 34707.275776],
    [1.164384, 2.017232, 0.000000, -70869.975040],
]

# Full-range (JPEG) YUV→RGB color twist matrices (16-bit scale).
#
# Full-range: Y spans 0-255 (no offset), Cb/Cr centered at 128.
#   M[i][3] = -(M[i][1]*32768 + M[i][2]*32768)

BT709_YUV_TO_RGB_16_FULL: list[list[float]] = [
    [1.0, 0.000000, 1.574800, -51603.046400],
    [1.0, -0.187300, -0.468100, 21476.147200],
    [1.0, 1.855600, 0.000000, -60804.300800],
]

BT601_YUV_TO_RGB_16_FULL: list[list[float]] = [
    [1.0, 0.000000, 1.402000, -45940.736000],
    [1.0, -0.344136, -0.714136, 34677.456896],
    [1.0, 1.772000, 0.000000, -58064.896000],
]

# Full-range (JPEG) YUV→RGB color twist matrices (8-bit scale).
#   M[i][3] = -(M[i][1]*128 + M[i][2]*128)

BT709_YUV_TO_RGB_8_FULL: list[list[float]] = [
    [1.0, 0.000000, 1.574800, -201.574400],
    [1.0, -0.187300, -0.468100, 83.891200],
    [1.0, 1.855600, 0.000000, -237.516800],
]

BT601_YUV_TO_RGB_8_FULL: list[list[float]] = [
    [1.0, 0.000000, 1.402000, -179.456000],
    [1.0, -0.344136, -0.714136, 135.458816],
    [1.0, 1.772000, 0.000000, -226.816000],
]

# Full-range YUV→RGB twist for 8-bit NV12 zero-extended to uint16.
# Input Y/Cb/Cr are 0-255 stored as uint16; output is RGB uint16 in 0-65535.
# Coefficients are the full-range values × 257 (to scale 0-255 → 0-65535).
#   M[i][3] = -(M[i][1]*128 + M[i][2]*128) * 257

BT709_NV12_8U_TO_RGB16_FULL: list[list[float]] = [
    [257.0, 0.000000, 404.723600, -51804.620800],
    [257.0, -48.136100, -120.301700, 21560.038400],
    [257.0, 476.889200, 0.000000, -61041.817600],
]

BT601_NV12_8U_TO_RGB16_FULL: list[list[float]] = [
    [257.0, 0.000000, 360.314000, -46120.192000],
    [257.0, -88.442952, -183.532952, 34812.915712],
    [257.0, 455.404000, 0.000000, -58291.712000],
]

_TwistRow = c_float * 4
_TwistMatrix = _TwistRow * 3


# ---------------------------------------------------------------------------
# 8-bit function bindings (_Ctx variants — compatible with CUDA 12.x and 13.x)
# ---------------------------------------------------------------------------

if _load_error is None:
    # nppiRGBToYCbCr420_8u_C3P3R_Ctx
    _nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx.argtypes = [
        c_void_p,
        c_int,
        c_void_p * 3,
        c_int * 3,
        NppiSize,
        NppStreamContext,
    ]
    _nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx.restype = c_int

    # nppiYCbCr420_8u_P3P2R_Ctx
    _nppicc.nppiYCbCr420_8u_P3P2R_Ctx.argtypes = [
        c_void_p * 3,
        c_int * 3,
        c_void_p,
        c_int,
        c_void_p,
        c_int,
        NppiSize,
        NppStreamContext,
    ]
    _nppicc.nppiYCbCr420_8u_P3P2R_Ctx.restype = c_int

    # nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx
    _nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx.argtypes = [
        c_void_p * 2,  # pSrc[2]: Y ptr, UV ptr
        c_int * 2,  # aSrcStep[2]: Y pitch, UV pitch (bytes)
        c_void_p,  # pDst: packed RGB8
        c_int,  # nDstStep (bytes)
        NppiSize,  # oSizeROI
        _TwistMatrix,  # aTwist[3][4]
        NppStreamContext,  # nppStreamCtx (by value)
    ]
    _nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx.restype = c_int

    # nppiResize_8u_C1R_Ctx (geometry library — single-channel 8-bit resize)
    _nppig.nppiResize_8u_C1R_Ctx.argtypes = [
        c_void_p,
        c_int,
        NppiSize,
        NppiRect,  # src ptr, src pitch, src size, src ROI
        c_void_p,
        c_int,
        NppiSize,
        NppiRect,  # dst ptr, dst pitch, dst size, dst ROI
        c_int,  # interpolation mode
        NppStreamContext,
    ]
    _nppig.nppiResize_8u_C1R_Ctx.restype = c_int

    # nppiResize_8u/16u_C3R_Ctx (packed 3-channel resize, same layout as C1R)
    for _resize_fn in (_nppig.nppiResize_8u_C3R_Ctx, _nppig.nppiResize_16u_C3R_Ctx):
        _resize_fn.argtypes = [
            c_void_p,
            c_int,
            NppiSize,
            NppiRect,  # src ptr, src pitch, src size, src ROI
            c_void_p,
            c_int,
            NppiSize,
            NppiRect,  # dst ptr, dst pitch, dst size, dst ROI
            c_int,  # interpolation mode
            NppStreamContext,
        ]
        _resize_fn.restype = c_int

    # nppiConvert_16u32f_C3R_Ctx (packed RGB uint16 → float32)
    _nppidei.nppiConvert_16u32f_C3R_Ctx.argtypes = [
        c_void_p,
        c_int,
        c_void_p,
        c_int,
        NppiSize,
        NppStreamContext,
    ]
    _nppidei.nppiConvert_16u32f_C3R_Ctx.restype = c_int

    # nppiConvert_32f16f_C3R_Ctx (packed RGB float32 → float16)
    _nppidei.nppiConvert_32f16f_C3R_Ctx.argtypes = [
        c_void_p,
        c_int,
        c_void_p,
        c_int,
        NppiSize,
        c_int,  # NppRoundMode
        NppStreamContext,
    ]
    _nppidei.nppiConvert_32f16f_C3R_Ctx.restype = c_int

    # nppiMulC_32f_C3IR_Ctx (in-place per-channel constant multiply)
    _nppial.nppiMulC_32f_C3IR_Ctx.argtypes = [
        c_float * 3,  # aConstants
        c_void_p,
        c_int,
        NppiSize,
        NppStreamContext,
    ]
    _nppial.nppiMulC_32f_C3IR_Ctx.restype = c_int

    # nppiResize_32f_C3R_Ctx (packed float32 resize, same layout as 8u)
    _nppig.nppiResize_32f_C3R_Ctx.argtypes = [
        c_void_p,
        c_int,
        NppiSize,
        NppiRect,
        c_void_p,
        c_int,
        NppiSize,
        NppiRect,
        c_int,
        NppStreamContext,
    ]
    _nppig.nppiResize_32f_C3R_Ctx.restype = c_int

    # nppiScale_8u32f_C3R_Ctx (uint8 → float32 scaled to [nMin, nMax])
    _nppidei.nppiScale_8u32f_C3R_Ctx.argtypes = [
        c_void_p,
        c_int,
        c_void_p,
        c_int,
        NppiSize,
        c_float,  # nMin
        c_float,  # nMax
        NppStreamContext,
    ]
    _nppidei.nppiScale_8u32f_C3R_Ctx.restype = c_int

    # nppiConvert_32f8u/16u_C3R_Ctx (float32 → integer with rounding)
    for _fnconv in (_nppidei.nppiConvert_32f8u_C3R_Ctx, _nppidei.nppiConvert_32f16u_C3R_Ctx):
        _fnconv.argtypes = [
            c_void_p,
            c_int,
            c_void_p,
            c_int,
            NppiSize,
            c_int,  # NppRoundMode
            NppStreamContext,
        ]
        _fnconv.restype = c_int


# ---------------------------------------------------------------------------
# 16-bit function bindings
# ---------------------------------------------------------------------------

if _load_error is None:
    # nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx
    # P016 (NV12 16-bit, 2 planes) -> packed RGB16
    _nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx.argtypes = [
        c_void_p * 2,  # pSrc[2]: Y ptr, UV ptr
        c_int * 2,  # aSrcStep[2]: Y pitch, UV pitch (bytes)
        c_void_p,  # pDst: packed RGB16
        c_int,  # nDstStep (bytes)
        NppiSize,  # oSizeROI
        _TwistMatrix,  # aTwist[3][4]
        NppStreamContext,  # nppStreamCtx (by value)
    ]
    _nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx.restype = c_int

    # nppiColorTwist32f_16u_C3IR_Ctx
    # In-place color twist on packed 16-bit 3-channel data
    _nppicc.nppiColorTwist32f_16u_C3IR_Ctx.argtypes = [
        c_void_p,  # pSrcDst
        c_int,  # nSrcDstStep (bytes)
        NppiSize,  # oSizeROI
        _TwistMatrix,  # aTwist[3][4]
        NppStreamContext,  # nppStreamCtx
    ]
    _nppicc.nppiColorTwist32f_16u_C3IR_Ctx.restype = c_int

    # nppiCopy_16u_P3C3R_Ctx
    # Interleave 3 planar 16-bit channels into packed (H,W,3)
    _nppidei.nppiCopy_16u_P3C3R_Ctx.argtypes = [
        c_void_p * 3,  # aSrc[3]
        c_int,  # nSrcStep (bytes, same for all planes)
        c_void_p,  # pDst: packed output
        c_int,  # nDstStep (bytes)
        NppiSize,  # oSizeROI
        NppStreamContext,  # nppStreamCtx
    ]
    _nppidei.nppiCopy_16u_P3C3R_Ctx.restype = c_int

    # nppiConvert_8u16u_C3R_Ctx
    # Widen uint8 -> uint16 (zero-extend: 128 -> 128)
    _nppidei.nppiConvert_8u16u_C3R_Ctx.argtypes = [
        c_void_p,
        c_int,  # pSrc, nSrcStep
        c_void_p,
        c_int,  # pDst, nDstStep
        NppiSize,  # oSizeROI
        NppStreamContext,  # nppStreamCtx
    ]
    _nppidei.nppiConvert_8u16u_C3R_Ctx.restype = c_int

    # nppiConvert_8u16u_C1R_Ctx
    # Widen single-channel uint8 -> uint16 (zero-extend)
    _nppidei.nppiConvert_8u16u_C1R_Ctx.argtypes = [
        c_void_p,
        c_int,  # pSrc, nSrcStep
        c_void_p,
        c_int,  # pDst, nDstStep
        NppiSize,  # oSizeROI
        NppStreamContext,  # nppStreamCtx
    ]
    _nppidei.nppiConvert_8u16u_C1R_Ctx.restype = c_int

    # nppiMulC_16u_C3IRSfs_Ctx
    # In-place multiply 3-channel uint16 by per-channel constants
    _nppial.nppiMulC_16u_C3IRSfs_Ctx.argtypes = [
        c_uint16 * 3,  # aConstants[3] (Npp16u)
        c_void_p,  # pSrcDst
        c_int,  # nSrcDstStep (bytes)
        NppiSize,  # oSizeROI
        c_int,  # nScaleFactor
        NppStreamContext,  # nppStreamCtx
    ]
    _nppial.nppiMulC_16u_C3IRSfs_Ctx.restype = c_int


# ---------------------------------------------------------------------------
# NppStreamContext builder
# ---------------------------------------------------------------------------
def make_npp_stream_context(gpu_id: int, stream: int = 0) -> NppStreamContext:
    """Build an NppStreamContext for the given GPU and CUDA stream.

    Uses cuda.bindings.driver to query device attributes.
    """
    from cuda.bindings import driver

    err, device = driver.cuDeviceGet(gpu_id)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'cuDeviceGet({gpu_id}) failed: {err}')

    def _attr(attr):
        err, val = driver.cuDeviceGetAttribute(attr, device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuDeviceGetAttribute({attr}) failed: {err}')
        return val

    A = driver.CUdevice_attribute
    ctx = NppStreamContext()
    ctx.hStream = stream
    ctx.nCudaDeviceId = gpu_id
    ctx.nMultiProcessorCount = _attr(A.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    ctx.nMaxThreadsPerMultiProcessor = _attr(A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    ctx.nMaxThreadsPerBlock = _attr(A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    ctx.nSharedMemPerBlock = _attr(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    ctx.nCudaDevAttrComputeCapabilityMajor = _attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    ctx.nCudaDevAttrComputeCapabilityMinor = _attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    ctx.nStreamFlags = 0
    ctx.nReserved0 = 0
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_twist(matrix: list[list[float]]) -> _TwistMatrix:
    """Convert a 3x4 Python list to a ctypes aTwist[3][4] array."""
    return _TwistMatrix(*(_TwistRow(*row) for row in matrix))


def _check(status: int, name: str) -> None:
    if status != NPP_SUCCESS:
        raise RuntimeError(f'NPP {name} failed with status {status}')


def _require_npp() -> None:
    if _load_error is not None:
        raise RuntimeError(
            'NPP libraries (libnpp*) could not be loaded; GPU color conversion '
            f'is unavailable: {_load_error}'
        )


def _require_even_dims(width: int, height: int, what: str) -> None:
    if width % 2 or height % 2:
        raise ValueError(
            f'{what} requires even width and height (4:2:0 chroma is subsampled '
            f'2x2), got {width}x{height}'
        )


_default_ctx_cache: dict[int, NppStreamContext] = {}


def _get_default_ctx() -> NppStreamContext:
    """Context for the current CUDA device, default stream (cached per device).

    The returned struct is shared — callers must not mutate it. To use a
    different stream, pass an explicit ``make_npp_stream_context`` result.
    """
    from cuda.bindings import driver

    err, dev = driver.cuCtxGetDevice()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            'No current CUDA context; create one first or pass an explicit '
            f'NppStreamContext (cuCtxGetDevice failed: {err})'
        )
    device_id = int(dev)
    ctx = _default_ctx_cache.get(device_id)
    if ctx is None:
        ctx = make_npp_stream_context(device_id)
        _default_ctx_cache[device_id] = ctx
    return ctx


# ---------------------------------------------------------------------------
# High-level conversion functions (8-bit, existing)
# ---------------------------------------------------------------------------
def yuv420_to_nv12(
    y_ptr: int,
    y_pitch: int,
    cb_ptr: int,
    cb_pitch: int,
    cr_ptr: int,
    cr_pitch: int,
    nv12_y_ptr: int,
    nv12_y_pitch: int,
    nv12_uv_ptr: int,
    nv12_uv_pitch: int,
    width: int,
    height: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert YUV420 planar (3 planes) to NV12 (2 planes) on GPU."""
    _require_npp()
    _require_even_dims(width, height, 'yuv420_to_nv12')
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 3)(y_ptr, cb_ptr, cr_ptr)
    src_steps = (c_int * 3)(y_pitch, cb_pitch, cr_pitch)
    status = _nppicc.nppiYCbCr420_8u_P3P2R_Ctx(
        src_ptrs,
        src_steps,
        nv12_y_ptr,
        nv12_y_pitch,
        nv12_uv_ptr,
        nv12_uv_pitch,
        size,
        ctx,
    )
    _check(status, 'YCbCr420 to NV12')


def rgb_to_nv12(
    rgb_ptr: int,
    rgb_pitch: int,
    nv12_y_ptr: int,
    nv12_y_pitch: int,
    nv12_uv_ptr: int,
    nv12_uv_pitch: int,
    width: int,
    height: int,
    temp_y_ptr: int,
    temp_cb_ptr: int,
    temp_cr_ptr: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert RGB to NV12 on GPU (two-step via YCbCr420 intermediate)."""
    _require_npp()
    _require_even_dims(width, height, 'rgb_to_nv12')
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)

    # Step 1: RGB -> YCbCr420 (3 planes)
    dst_ptrs = (c_void_p * 3)(temp_y_ptr, temp_cb_ptr, temp_cr_ptr)
    dst_steps = (c_int * 3)(width, width // 2, width // 2)
    status = _nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx(
        rgb_ptr,
        rgb_pitch,
        dst_ptrs,
        dst_steps,
        size,
        ctx,
    )
    _check(status, 'RGB to YCbCr420')

    # Step 2: YCbCr420 (3 planes) -> NV12 (2 planes)
    src_ptrs = (c_void_p * 3)(temp_y_ptr, temp_cb_ptr, temp_cr_ptr)
    src_steps = (c_int * 3)(width, width // 2, width // 2)
    status = _nppicc.nppiYCbCr420_8u_P3P2R_Ctx(
        src_ptrs,
        src_steps,
        nv12_y_ptr,
        nv12_y_pitch,
        nv12_uv_ptr,
        nv12_uv_pitch,
        size,
        ctx,
    )
    _check(status, 'YCbCr420 to NV12')


def nv12_to_rgb8(
    y_ptr: int,
    y_pitch: int,
    uv_ptr: int,
    uv_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    width: int,
    height: int,
    twist: list[list[float]],
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert NV12 to packed RGB uint8 with a color twist matrix."""
    _require_npp()
    _require_even_dims(width, height, 'nv12_to_rgb8')
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 2)(y_ptr, uv_ptr)
    src_steps = (c_int * 2)(y_pitch, uv_pitch)
    status = _nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
        src_ptrs,
        src_steps,
        dst_ptr,
        dst_pitch,
        size,
        _make_twist(twist),
        ctx,
    )
    _check(status, 'NV12 to RGB8 ColorTwist')


def nv12_to_p016(
    y_ptr: int,
    y_pitch: int,
    uv_ptr: int,
    uv_pitch: int,
    dst_y_ptr: int,
    dst_y_pitch: int,
    dst_uv_ptr: int,
    dst_uv_pitch: int,
    width: int,
    height: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Zero-extend NV12 (8-bit) planes to P016 (16-bit) format.

    Widens both Y and UV planes from uint8 to uint16 in-place.
    The UV plane is treated as single-channel with width equal to the luma
    width (since NV12 UV is interleaved U0V0U1V1... = width bytes per row).
    """
    _require_npp()
    _require_even_dims(width, height, 'nv12_to_p016')
    if ctx is None:
        ctx = _get_default_ctx()

    # Widen Y plane
    y_size = NppiSize(width, height)
    status = _nppidei.nppiConvert_8u16u_C1R_Ctx(
        y_ptr,
        y_pitch,
        dst_y_ptr,
        dst_y_pitch,
        y_size,
        ctx,
    )
    _check(status, 'Convert Y 8u16u')

    # Widen UV plane (interleaved UV: width bytes per row, height/2 rows)
    uv_size = NppiSize(width, height // 2)
    status = _nppidei.nppiConvert_8u16u_C1R_Ctx(
        uv_ptr,
        uv_pitch,
        dst_uv_ptr,
        dst_uv_pitch,
        uv_size,
        ctx,
    )
    _check(status, 'Convert UV 8u16u')


def resize_plane_8u(
    src_ptr: int,
    src_pitch: int,
    src_w: int,
    src_h: int,
    dst_ptr: int,
    dst_pitch: int,
    dst_w: int,
    dst_h: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Resize a single-channel 8-bit plane on GPU using area averaging."""
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    src_size = NppiSize(src_w, src_h)
    dst_size = NppiSize(dst_w, dst_h)
    src_roi = NppiRect(0, 0, src_w, src_h)
    dst_roi = NppiRect(0, 0, dst_w, dst_h)
    status = _nppig.nppiResize_8u_C1R_Ctx(
        src_ptr,
        src_pitch,
        src_size,
        src_roi,
        dst_ptr,
        dst_pitch,
        dst_size,
        dst_roi,
        # Area averaging for true 2D downsampling; NPP rejects SUPER when one
        # axis keeps its size (the 4:2:2 chroma geometry), so use LINEAR there
        # (a 2-tap average at the exact 2:1 horizontal ratio).
        NPPI_INTER_SUPER if dst_w < src_w and dst_h < src_h else NPPI_INTER_LINEAR,
        ctx,
    )
    _check(status, 'Resize 8u C1R')


NPP_RND_NEAR = 0  # NppRoundMode: round to nearest, ties to even


def resize_rgb(
    src_ptr: int,
    src_pitch: int,
    src_w: int,
    src_h: int,
    dst_ptr: int,
    dst_pitch: int,
    dst_w: int,
    dst_h: int,
    *,
    bits: int = 8,
    ctx: NppStreamContext | None = None,
) -> None:
    """Resize a packed 3-channel RGB image on GPU (8- or 16-bit samples).

    Area averaging (SUPER) for true 2D downscaling; LANCZOS otherwise
    (NPP rejects SUPER unless both dimensions shrink).
    """
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    fn = {
        8: _nppig.nppiResize_8u_C3R_Ctx,
        16: _nppig.nppiResize_16u_C3R_Ctx,
        32: _nppig.nppiResize_32f_C3R_Ctx,
    }[bits]
    status = fn(
        src_ptr,
        src_pitch,
        NppiSize(src_w, src_h),
        NppiRect(0, 0, src_w, src_h),
        dst_ptr,
        dst_pitch,
        NppiSize(dst_w, dst_h),
        NppiRect(0, 0, dst_w, dst_h),
        NPPI_INTER_SUPER if dst_w < src_w and dst_h < src_h else NPPI_INTER_LANCZOS,
        ctx,
    )
    _check(status, f'Resize {bits}u C3R')


def rgb16_to_float01(
    src_ptr: int,
    src_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    w: int,
    h: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert packed RGB uint16 to float32 scaled to [0, 1]."""
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(w, h)
    status = _nppidei.nppiConvert_16u32f_C3R_Ctx(src_ptr, src_pitch, dst_ptr, dst_pitch, size, ctx)
    _check(status, 'Convert 16u32f C3R')
    inv = 1.0 / 65535.0
    status = _nppial.nppiMulC_32f_C3IR_Ctx((c_float * 3)(inv, inv, inv), dst_ptr, dst_pitch, size, ctx)
    _check(status, 'MulC 32f C3IR')


def rgb8_to_float01(
    src_ptr: int,
    src_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    w: int,
    h: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert packed RGB uint8 to float32 scaled to [0, 1]."""
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    status = _nppidei.nppiScale_8u32f_C3R_Ctx(
        src_ptr, src_pitch, dst_ptr, dst_pitch, NppiSize(w, h), 0.0, 1.0, ctx
    )
    _check(status, 'Scale 8u32f C3R')


def float01_to_uint(
    src_ptr: int,
    src_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    w: int,
    h: int,
    *,
    bits: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert packed RGB float32 in [0, 1] to uint8/uint16 (rounding).

    Scales the source buffer in place (it is a scratch buffer by contract).
    """
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(w, h)
    maxval = float((1 << bits) - 1)
    status = _nppial.nppiMulC_32f_C3IR_Ctx(
        (c_float * 3)(maxval, maxval, maxval), src_ptr, src_pitch, size, ctx
    )
    _check(status, 'MulC 32f C3IR')
    fn = _nppidei.nppiConvert_32f8u_C3R_Ctx if bits == 8 else _nppidei.nppiConvert_32f16u_C3R_Ctx
    status = fn(src_ptr, src_pitch, dst_ptr, dst_pitch, size, NPP_RND_NEAR, ctx)
    _check(status, f'Convert 32f{bits}u C3R')


def float32_to_float16(
    src_ptr: int,
    src_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    w: int,
    h: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert packed RGB float32 to float16 (round to nearest)."""
    _require_npp()
    if ctx is None:
        ctx = _get_default_ctx()
    status = _nppidei.nppiConvert_32f16f_C3R_Ctx(
        src_ptr, src_pitch, dst_ptr, dst_pitch, NppiSize(w, h), NPP_RND_NEAR, ctx
    )
    _check(status, 'Convert 32f16f C3R')


# ---------------------------------------------------------------------------
# High-level conversion functions (16-bit)
# ---------------------------------------------------------------------------
_IDENTITY_TWIST: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]


def p016_to_rgb16(
    y_ptr: int,
    y_pitch: int,
    uv_ptr: int,
    uv_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    width: int,
    height: int,
    twist: list[list[float]],
    ctx: NppStreamContext,
) -> None:
    """Convert P016 (NV12 16-bit) to packed RGB uint16.

    Two steps to avoid version-dependent twist semantics in the NV12-specific
    NPP function (changed between CUDA 12.x and 13.x):

    Step 1: NV12 → packed YCbCr16 (identity twist — just unpack + upsample).
    Step 2: In-place color twist on packed data (standard semantics, stable).
    """
    _require_npp()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 2)(y_ptr, uv_ptr)
    src_steps = (c_int * 2)(y_pitch, uv_pitch)

    # Step 1: NV12 → packed (Y, Cb, Cr) with identity twist
    status = _nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(
        src_ptrs,
        src_steps,
        dst_ptr,
        dst_pitch,
        size,
        _make_twist(_IDENTITY_TWIST),
        ctx,
    )
    _check(status, 'NV12 16u unpack')

    # Step 2: In-place color twist: packed YCbCr → packed RGB
    status = _nppicc.nppiColorTwist32f_16u_C3IR_Ctx(
        dst_ptr,
        dst_pitch,
        size,
        _make_twist(twist),
        ctx,
    )
    _check(status, 'ColorTwist P016 to RGB')


def yuv444_16bit_to_rgb16(
    y_ptr: int,
    u_ptr: int,
    v_ptr: int,
    plane_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    width: int,
    height: int,
    twist: list[list[float]],
    ctx: NppStreamContext,
) -> None:
    """Convert YUV444_16Bit (3 planes) to packed RGB uint16.

    Step 1: Interleave 3 planar channels into packed (H,W,3).
    Step 2: In-place color twist on the packed buffer.

    The destination buffer is used for both steps (interleave target,
    then in-place twist), so it must be pre-allocated.
    """
    _require_npp()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 3)(y_ptr, u_ptr, v_ptr)

    # Step 1: Planar YUV -> packed YUV
    status = _nppidei.nppiCopy_16u_P3C3R_Ctx(
        src_ptrs,
        plane_pitch,
        dst_ptr,
        dst_pitch,
        size,
        ctx,
    )
    _check(status, 'Copy P3C3 (interleave)')

    # Step 2: In-place color twist: packed YUV -> packed RGB
    status = _nppicc.nppiColorTwist32f_16u_C3IR_Ctx(
        dst_ptr,
        dst_pitch,
        size,
        _make_twist(twist),
        ctx,
    )
    _check(status, 'ColorTwist C3IR')


def rgb8_to_rgb16(
    src_ptr: int,
    src_pitch: int,
    dst_ptr: int,
    dst_pitch: int,
    width: int,
    height: int,
    ctx: NppStreamContext,
) -> None:
    """Scale packed RGB uint8 to packed RGB uint16 (0-255 -> 0-65535).

    Step 1: Zero-extend uint8 -> uint16 (128 -> 128).
    Step 2: Multiply by 257 to fill the full uint16 range (128 -> 32896).
    This matches FFmpeg's rgb24->rgb48 conversion behavior.
    """
    _require_npp()
    size = NppiSize(width, height)

    # Step 1: Widen 8u -> 16u
    status = _nppidei.nppiConvert_8u16u_C3R_Ctx(
        src_ptr,
        src_pitch,
        dst_ptr,
        dst_pitch,
        size,
        ctx,
    )
    _check(status, 'Convert 8u16u')

    # Step 2: Multiply by 257 (in-place) to scale 0-255 -> 0-65535
    constants = (c_uint16 * 3)(257, 257, 257)
    status = _nppial.nppiMulC_16u_C3IRSfs_Ctx(
        constants,
        dst_ptr,
        dst_pitch,
        size,
        0,
        ctx,
    )
    _check(status, 'MulC 16u scale')
