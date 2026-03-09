"""ctypes bindings for NPP (NVIDIA Performance Primitives) color conversion."""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_size_t, c_uint, c_uint16, c_void_p, Structure

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
_nppicc = ctypes.CDLL('libnppicc.so')    # color conversion
_nppidei = ctypes.CDLL('libnppidei.so')  # data exchange and initialization
_nppial = ctypes.CDLL('libnppial.so')    # arithmetic and logical
_nppig = ctypes.CDLL('libnppig.so')      # geometry transforms (resize)


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
NPPI_INTER_SUPER = 16  # area-averaging downsample (box filter for integer ratios)

# BT.709 / BT.601 limited-range YUV→RGB color twist matrices (16-bit scale).
#
# Standard twist format for nppiColorTwist32f:
#   dst[i] = M[i][0]*Y + M[i][1]*Cb + M[i][2]*Cr + M[i][3]
#
# The 4th column absorbs the source offsets (Y-16*256, Cb-128*256, Cr-128*256):
#   M[i][3] = -(M[i][0]*4096 + M[i][1]*32768 + M[i][2]*32768)

BT709_YUV_TO_RGB_16: list[list[float]] = [
    [1.164384,  0.000000,  1.792741, -63513.853952],
    [1.164384, -0.213249, -0.532909,  19680.788480],
    [1.164384,  2.112402,  0.000000, -73988.505600],
]

BT601_YUV_TO_RGB_16: list[list[float]] = [
    [1.164384,  0.000000,  1.596027, -57067.929600],
    [1.164384, -0.391762, -0.812968,  34707.275776],
    [1.164384,  2.017232,  0.000000, -70869.975040],
]

# Full-range (JPEG) YUV→RGB color twist matrices (16-bit scale).
#
# Full-range: Y spans 0-255 (no offset), Cb/Cr centered at 128.
#   M[i][3] = -(M[i][1]*32768 + M[i][2]*32768)

BT709_YUV_TO_RGB_16_FULL: list[list[float]] = [
    [1.0,  0.000000,  1.574800, -51603.046400],
    [1.0, -0.187300, -0.468100,  21476.147200],
    [1.0,  1.855600,  0.000000, -60804.300800],
]

BT601_YUV_TO_RGB_16_FULL: list[list[float]] = [
    [1.0,  0.000000,  1.402000, -45940.736000],
    [1.0, -0.344136, -0.714136,  34677.456896],
    [1.0,  1.772000,  0.000000, -58064.896000],
]

# Full-range (JPEG) YUV→RGB color twist matrices (8-bit scale).
#   M[i][3] = -(M[i][1]*128 + M[i][2]*128)

BT709_YUV_TO_RGB_8_FULL: list[list[float]] = [
    [1.0,  0.000000,  1.574800, -201.574400],
    [1.0, -0.187300, -0.468100,   83.891200],
    [1.0,  1.855600,  0.000000, -237.516800],
]

BT601_YUV_TO_RGB_8_FULL: list[list[float]] = [
    [1.0,  0.000000,  1.402000, -179.456000],
    [1.0, -0.344136, -0.714136,  135.458816],
    [1.0,  1.772000,  0.000000, -226.816000],
]

# Full-range YUV→RGB twist for 8-bit NV12 zero-extended to uint16.
# Input Y/Cb/Cr are 0-255 stored as uint16; output is RGB uint16 in 0-65535.
# Coefficients are the full-range values × 257 (to scale 0-255 → 0-65535).
#   M[i][3] = -(M[i][1]*128 + M[i][2]*128) * 257

BT709_NV12_8U_TO_RGB16_FULL: list[list[float]] = [
    [257.0,    0.000000, 404.723600, -51804.620800],
    [257.0,  -48.136100, -120.301700,  21560.038400],
    [257.0,  476.889200,    0.000000, -61041.817600],
]

BT601_NV12_8U_TO_RGB16_FULL: list[list[float]] = [
    [257.0,    0.000000,  360.314000, -46120.192000],
    [257.0,  -88.442952, -183.532952,  34812.915712],
    [257.0,  455.404000,    0.000000, -58291.712000],
]

_TwistRow = c_float * 4
_TwistMatrix = _TwistRow * 3


# ---------------------------------------------------------------------------
# 8-bit function bindings (_Ctx variants — compatible with CUDA 12.x and 13.x)
# ---------------------------------------------------------------------------

# nppiRGBToYCbCr420_8u_C3P3R_Ctx
_nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx.argtypes = [
    c_void_p, c_int, c_void_p * 3, c_int * 3, NppiSize, NppStreamContext,
]
_nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx.restype = c_int

# nppiYCbCr420_8u_P3P2R_Ctx
_nppicc.nppiYCbCr420_8u_P3P2R_Ctx.argtypes = [
    c_void_p * 3, c_int * 3, c_void_p, c_int, c_void_p, c_int, NppiSize,
    NppStreamContext,
]
_nppicc.nppiYCbCr420_8u_P3P2R_Ctx.restype = c_int

# nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx
_nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx.argtypes = [
    c_void_p * 2,      # pSrc[2]: Y ptr, UV ptr
    c_int * 2,         # aSrcStep[2]: Y pitch, UV pitch (bytes)
    c_void_p,          # pDst: packed RGB8
    c_int,             # nDstStep (bytes)
    NppiSize,          # oSizeROI
    _TwistMatrix,      # aTwist[3][4]
    NppStreamContext,   # nppStreamCtx (by value)
]
_nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx.restype = c_int

# nppiResize_8u_C1R_Ctx (geometry library — single-channel 8-bit resize)
_nppig.nppiResize_8u_C1R_Ctx.argtypes = [
    c_void_p, c_int, NppiSize, NppiRect,  # src ptr, src pitch, src size, src ROI
    c_void_p, c_int, NppiSize, NppiRect,  # dst ptr, dst pitch, dst size, dst ROI
    c_int,                                  # interpolation mode
    NppStreamContext,
]
_nppig.nppiResize_8u_C1R_Ctx.restype = c_int


# ---------------------------------------------------------------------------
# 16-bit function bindings
# ---------------------------------------------------------------------------

# nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx
# P016 (NV12 16-bit, 2 planes) -> packed RGB16
_nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx.argtypes = [
    c_void_p * 2,      # pSrc[2]: Y ptr, UV ptr
    c_int * 2,         # aSrcStep[2]: Y pitch, UV pitch (bytes)
    c_void_p,          # pDst: packed RGB16
    c_int,             # nDstStep (bytes)
    NppiSize,          # oSizeROI
    _TwistMatrix,      # aTwist[3][4]
    NppStreamContext,   # nppStreamCtx (by value)
]
_nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx.restype = c_int

# nppiColorTwist32f_16u_C3IR_Ctx
# In-place color twist on packed 16-bit 3-channel data
_nppicc.nppiColorTwist32f_16u_C3IR_Ctx.argtypes = [
    c_void_p,          # pSrcDst
    c_int,             # nSrcDstStep (bytes)
    NppiSize,          # oSizeROI
    _TwistMatrix,      # aTwist[3][4]
    NppStreamContext,   # nppStreamCtx
]
_nppicc.nppiColorTwist32f_16u_C3IR_Ctx.restype = c_int

# nppiCopy_16u_P3C3R_Ctx
# Interleave 3 planar 16-bit channels into packed (H,W,3)
_nppidei.nppiCopy_16u_P3C3R_Ctx.argtypes = [
    c_void_p * 3,      # aSrc[3]
    c_int,             # nSrcStep (bytes, same for all planes)
    c_void_p,          # pDst: packed output
    c_int,             # nDstStep (bytes)
    NppiSize,          # oSizeROI
    NppStreamContext,   # nppStreamCtx
]
_nppidei.nppiCopy_16u_P3C3R_Ctx.restype = c_int

# nppiConvert_8u16u_C3R_Ctx
# Widen uint8 -> uint16 (zero-extend: 128 -> 128)
_nppidei.nppiConvert_8u16u_C3R_Ctx.argtypes = [
    c_void_p, c_int,   # pSrc, nSrcStep
    c_void_p, c_int,   # pDst, nDstStep
    NppiSize,          # oSizeROI
    NppStreamContext,   # nppStreamCtx
]
_nppidei.nppiConvert_8u16u_C3R_Ctx.restype = c_int

# nppiConvert_8u16u_C1R_Ctx
# Widen single-channel uint8 -> uint16 (zero-extend)
_nppidei.nppiConvert_8u16u_C1R_Ctx.argtypes = [
    c_void_p, c_int,   # pSrc, nSrcStep
    c_void_p, c_int,   # pDst, nDstStep
    NppiSize,          # oSizeROI
    NppStreamContext,   # nppStreamCtx
]
_nppidei.nppiConvert_8u16u_C1R_Ctx.restype = c_int

# nppiMulC_16u_C3IRSfs_Ctx
# In-place multiply 3-channel uint16 by per-channel constants
_nppial.nppiMulC_16u_C3IRSfs_Ctx.argtypes = [
    c_uint16 * 3,      # aConstants[3] (Npp16u)
    c_void_p,          # pSrcDst
    c_int,             # nSrcDstStep (bytes)
    NppiSize,          # oSizeROI
    c_int,             # nScaleFactor
    NppStreamContext,   # nppStreamCtx
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
    ctx.nMaxThreadsPerMultiProcessor = _attr(
        A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
    )
    ctx.nMaxThreadsPerBlock = _attr(A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    ctx.nSharedMemPerBlock = _attr(
        A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    )
    ctx.nCudaDevAttrComputeCapabilityMajor = _attr(
        A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    )
    ctx.nCudaDevAttrComputeCapabilityMinor = _attr(
        A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    )
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


_default_npp_ctx: NppStreamContext | None = None


def _get_default_ctx() -> NppStreamContext:
    """Lazily create a default NppStreamContext (GPU 0, default stream)."""
    global _default_npp_ctx
    if _default_npp_ctx is None:
        _default_npp_ctx = make_npp_stream_context(0)
    return _default_npp_ctx


# ---------------------------------------------------------------------------
# High-level conversion functions (8-bit, existing)
# ---------------------------------------------------------------------------
def yuv420_to_nv12(
    y_ptr: int, y_pitch: int,
    cb_ptr: int, cb_pitch: int,
    cr_ptr: int, cr_pitch: int,
    nv12_y_ptr: int, nv12_y_pitch: int,
    nv12_uv_ptr: int, nv12_uv_pitch: int,
    width: int, height: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert YUV420 planar (3 planes) to NV12 (2 planes) on GPU."""
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 3)(y_ptr, cb_ptr, cr_ptr)
    src_steps = (c_int * 3)(y_pitch, cb_pitch, cr_pitch)
    status = _nppicc.nppiYCbCr420_8u_P3P2R_Ctx(
        src_ptrs, src_steps,
        nv12_y_ptr, nv12_y_pitch, nv12_uv_ptr, nv12_uv_pitch, size, ctx,
    )
    _check(status, 'YCbCr420 to NV12')


def rgb_to_nv12(
    rgb_ptr: int, rgb_pitch: int,
    nv12_y_ptr: int, nv12_y_pitch: int,
    nv12_uv_ptr: int, nv12_uv_pitch: int,
    width: int, height: int,
    temp_y_ptr: int, temp_cb_ptr: int, temp_cr_ptr: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert RGB to NV12 on GPU (two-step via YCbCr420 intermediate)."""
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)

    # Step 1: RGB -> YCbCr420 (3 planes)
    dst_ptrs = (c_void_p * 3)(temp_y_ptr, temp_cb_ptr, temp_cr_ptr)
    dst_steps = (c_int * 3)(width, width // 2, width // 2)
    status = _nppicc.nppiRGBToYCbCr420_8u_C3P3R_Ctx(
        rgb_ptr, rgb_pitch, dst_ptrs, dst_steps, size, ctx,
    )
    _check(status, 'RGB to YCbCr420')

    # Step 2: YCbCr420 (3 planes) -> NV12 (2 planes)
    src_ptrs = (c_void_p * 3)(temp_y_ptr, temp_cb_ptr, temp_cr_ptr)
    src_steps = (c_int * 3)(width, width // 2, width // 2)
    status = _nppicc.nppiYCbCr420_8u_P3P2R_Ctx(
        src_ptrs, src_steps,
        nv12_y_ptr, nv12_y_pitch, nv12_uv_ptr, nv12_uv_pitch, size, ctx,
    )
    _check(status, 'YCbCr420 to NV12')


def nv12_to_rgb8(
    y_ptr: int, y_pitch: int,
    uv_ptr: int, uv_pitch: int,
    dst_ptr: int, dst_pitch: int,
    width: int, height: int,
    twist: list[list[float]],
    ctx: NppStreamContext | None = None,
) -> None:
    """Convert NV12 to packed RGB uint8 with a color twist matrix."""
    if ctx is None:
        ctx = _get_default_ctx()
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 2)(y_ptr, uv_ptr)
    src_steps = (c_int * 2)(y_pitch, uv_pitch)
    status = _nppicc.nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
        src_ptrs, src_steps,
        dst_ptr, dst_pitch, size,
        _make_twist(twist), ctx,
    )
    _check(status, 'NV12 to RGB8 ColorTwist')


def nv12_to_p016(
    y_ptr: int, y_pitch: int,
    uv_ptr: int, uv_pitch: int,
    dst_y_ptr: int, dst_y_pitch: int,
    dst_uv_ptr: int, dst_uv_pitch: int,
    width: int, height: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Zero-extend NV12 (8-bit) planes to P016 (16-bit) format.

    Widens both Y and UV planes from uint8 to uint16 in-place.
    The UV plane is treated as single-channel with width equal to the luma
    width (since NV12 UV is interleaved U0V0U1V1... = width bytes per row).
    """
    if ctx is None:
        ctx = _get_default_ctx()

    # Widen Y plane
    y_size = NppiSize(width, height)
    status = _nppidei.nppiConvert_8u16u_C1R_Ctx(
        y_ptr, y_pitch, dst_y_ptr, dst_y_pitch, y_size, ctx,
    )
    _check(status, 'Convert Y 8u16u')

    # Widen UV plane (interleaved UV: width bytes per row, height/2 rows)
    uv_size = NppiSize(width, height // 2)
    status = _nppidei.nppiConvert_8u16u_C1R_Ctx(
        uv_ptr, uv_pitch, dst_uv_ptr, dst_uv_pitch, uv_size, ctx,
    )
    _check(status, 'Convert UV 8u16u')


def resize_plane_8u(
    src_ptr: int, src_pitch: int, src_w: int, src_h: int,
    dst_ptr: int, dst_pitch: int, dst_w: int, dst_h: int,
    ctx: NppStreamContext | None = None,
) -> None:
    """Resize a single-channel 8-bit plane on GPU using area averaging."""
    if ctx is None:
        ctx = _get_default_ctx()
    src_size = NppiSize(src_w, src_h)
    dst_size = NppiSize(dst_w, dst_h)
    src_roi = NppiRect(0, 0, src_w, src_h)
    dst_roi = NppiRect(0, 0, dst_w, dst_h)
    status = _nppig.nppiResize_8u_C1R_Ctx(
        src_ptr, src_pitch, src_size, src_roi,
        dst_ptr, dst_pitch, dst_size, dst_roi,
        NPPI_INTER_SUPER, ctx,
    )
    _check(status, 'Resize 8u C1R')


_INTERLEAVE_UV_PTX = b'''\
.version 9.1
.target sm_52
.address_size 64
.visible .entry interleave_uv(
 .param .u64 p0, .param .u32 p1, .param .u64 p2, .param .u32 p3,
 .param .u64 p4, .param .u32 p5, .param .u32 p6, .param .u32 p7)
{
 .reg .pred %p<4>; .reg .b16 %rs<3>; .reg .b32 %r<18>; .reg .b64 %rd<13>;
 ld.param.u64 %rd1,[p0]; ld.param.u32 %r3,[p1]; ld.param.u64 %rd2,[p2];
 ld.param.u32 %r4,[p3]; ld.param.u64 %rd3,[p4]; ld.param.u32 %r5,[p5];
 ld.param.u32 %r6,[p6]; ld.param.u32 %r7,[p7];
 mov.u32 %r8,%ntid.x; mov.u32 %r9,%ctaid.x; mov.u32 %r10,%tid.x;
 mad.lo.s32 %r1,%r9,%r8,%r10;
 mov.u32 %r11,%ntid.y; mov.u32 %r12,%ctaid.y; mov.u32 %r13,%tid.y;
 mad.lo.s32 %r2,%r12,%r11,%r13;
 setp.ge.s32 %p1,%r1,%r6; setp.ge.s32 %p2,%r2,%r7; or.pred %p3,%p1,%p2;
 @%p3 bra $done;
 cvta.to.global.u64 %rd4,%rd1;
 mad.lo.s32 %r14,%r2,%r3,%r1; cvt.s64.s32 %rd5,%r14;
 add.s64 %rd6,%rd4,%rd5; ld.global.u8 %rs1,[%rd6];
 shl.b32 %r15,%r1,1; mad.lo.s32 %r16,%r2,%r5,%r15;
 cvt.s64.s32 %rd7,%r16; cvta.to.global.u64 %rd8,%rd3;
 add.s64 %rd9,%rd8,%rd7; st.global.u8 [%rd9],%rs1;
 mad.lo.s32 %r17,%r2,%r4,%r1; cvt.s64.s32 %rd10,%r17;
 cvta.to.global.u64 %rd11,%rd2; add.s64 %rd12,%rd11,%rd10;
 ld.global.u8 %rs2,[%rd12]; st.global.u8 [%rd9+1],%rs2;
$done: ret;
}
'''

_interleave_module = None
_interleave_func = None


def _get_interleave_func():
    global _interleave_module, _interleave_func
    if _interleave_func is None:
        from cuda.bindings import driver
        err, mod = driver.cuModuleLoadData(_INTERLEAVE_UV_PTX)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to load interleave PTX: {err}')
        _interleave_module = mod
        err, func = driver.cuModuleGetFunction(mod, b'interleave_uv')
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get interleave_uv function: {err}')
        _interleave_func = func
    return _interleave_func


def interleave_uv(
    u_ptr: int, u_pitch: int,
    v_ptr: int, v_pitch: int,
    uv_ptr: int, uv_pitch: int,
    chroma_width: int, chroma_height: int,
    stream: int = 0,
) -> None:
    """Interleave separate U and V planes into a single UV plane (for NV16).

    Each output row is: U0 V0 U1 V1 ... U(w-1) V(w-1).
    """
    import ctypes
    from cuda.bindings import driver
    func = _get_interleave_func()
    block_x, block_y = 32, 8
    grid_x = (chroma_width + block_x - 1) // block_x
    grid_y = (chroma_height + block_y - 1) // block_y

    # Pack kernel arguments
    args = (
        ctypes.c_uint64(u_ptr), ctypes.c_int32(u_pitch),
        ctypes.c_uint64(v_ptr), ctypes.c_int32(v_pitch),
        ctypes.c_uint64(uv_ptr), ctypes.c_int32(uv_pitch),
        ctypes.c_int32(chroma_width), ctypes.c_int32(chroma_height),
    )
    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    err, = driver.cuLaunchKernel(
        func, grid_x, grid_y, 1, block_x, block_y, 1, 0, stream, arg_ptrs, 0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'interleave_uv kernel launch failed: {err}')


# ---------------------------------------------------------------------------
# High-level conversion functions (16-bit)
# ---------------------------------------------------------------------------
_IDENTITY_TWIST: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]


def p016_to_rgb16(
    y_ptr: int, y_pitch: int,
    uv_ptr: int, uv_pitch: int,
    dst_ptr: int, dst_pitch: int,
    width: int, height: int,
    twist: list[list[float]],
    ctx: NppStreamContext,
) -> None:
    """Convert P016 (NV12 16-bit) to packed RGB uint16.

    Two steps to avoid version-dependent twist semantics in the NV12-specific
    NPP function (changed between CUDA 12.x and 13.x):

    Step 1: NV12 → packed YCbCr16 (identity twist — just unpack + upsample).
    Step 2: In-place color twist on packed data (standard semantics, stable).
    """
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 2)(y_ptr, uv_ptr)
    src_steps = (c_int * 2)(y_pitch, uv_pitch)

    # Step 1: NV12 → packed (Y, Cb, Cr) with identity twist
    status = _nppicc.nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(
        src_ptrs, src_steps,
        dst_ptr, dst_pitch, size,
        _make_twist(_IDENTITY_TWIST),
        ctx,
    )
    _check(status, 'NV12 16u unpack')

    # Step 2: In-place color twist: packed YCbCr → packed RGB
    status = _nppicc.nppiColorTwist32f_16u_C3IR_Ctx(
        dst_ptr, dst_pitch, size, _make_twist(twist), ctx,
    )
    _check(status, 'ColorTwist P016 to RGB')


def yuv444_16bit_to_rgb16(
    y_ptr: int, u_ptr: int, v_ptr: int, plane_pitch: int,
    dst_ptr: int, dst_pitch: int,
    width: int, height: int,
    twist: list[list[float]],
    ctx: NppStreamContext,
) -> None:
    """Convert YUV444_16Bit (3 planes) to packed RGB uint16.

    Step 1: Interleave 3 planar channels into packed (H,W,3).
    Step 2: In-place color twist on the packed buffer.

    The destination buffer is used for both steps (interleave target,
    then in-place twist), so it must be pre-allocated.
    """
    size = NppiSize(width, height)
    src_ptrs = (c_void_p * 3)(y_ptr, u_ptr, v_ptr)

    # Step 1: Planar YUV -> packed YUV
    status = _nppidei.nppiCopy_16u_P3C3R_Ctx(
        src_ptrs, plane_pitch,
        dst_ptr, dst_pitch,
        size, ctx,
    )
    _check(status, 'Copy P3C3 (interleave)')

    # Step 2: In-place color twist: packed YUV -> packed RGB
    status = _nppicc.nppiColorTwist32f_16u_C3IR_Ctx(
        dst_ptr, dst_pitch, size, _make_twist(twist), ctx,
    )
    _check(status, 'ColorTwist C3IR')


def rgb8_to_rgb16(
    src_ptr: int, src_pitch: int,
    dst_ptr: int, dst_pitch: int,
    width: int, height: int,
    ctx: NppStreamContext,
) -> None:
    """Scale packed RGB uint8 to packed RGB uint16 (0-255 -> 0-65535).

    Step 1: Zero-extend uint8 -> uint16 (128 -> 128).
    Step 2: Multiply by 257 to fill the full uint16 range (128 -> 32896).
    This matches FFmpeg's rgb24->rgb48 conversion behavior.
    """
    size = NppiSize(width, height)

    # Step 1: Widen 8u -> 16u
    status = _nppidei.nppiConvert_8u16u_C3R_Ctx(
        src_ptr, src_pitch, dst_ptr, dst_pitch, size, ctx,
    )
    _check(status, 'Convert 8u16u')

    # Step 2: Multiply by 257 (in-place) to scale 0-255 -> 0-65535
    constants = (c_uint16 * 3)(257, 257, 257)
    status = _nppial.nppiMulC_16u_C3IRSfs_Ctx(
        constants, dst_ptr, dst_pitch, size, 0, ctx,
    )
    _check(status, 'MulC 16u scale')
