"""Hand-written PTX kernels for operations NPP has no primitive for.

CUmodule/CUfunction handles are only valid inside the context they were
loaded in, so the per-kernel caches are keyed by the current context id.
"""

from __future__ import annotations

import ctypes

_INTERLEAVE_UV_PTX = b"""\
.version 7.0
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
"""

# CUmodule/CUfunction handles are only valid inside the context they were
# loaded in, so the cache is keyed by the current context. A plain global
# handle would go stale once that context is destroyed (e.g. between two
# sequentially used writers in one process).
_interleave_funcs: dict[int, object] = {}


def _get_interleave_func():
    from cuda.bindings import driver

    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
        raise RuntimeError('interleave_uv requires a current CUDA context')
    # cuCtxGetId is unique per context *incarnation*: a new context reusing a
    # destroyed context's address must not hit the stale cached CUfunction.
    # (CUDA 12+ driver; fall back to the address on older drivers.)
    if hasattr(driver, 'cuCtxGetId'):
        err, ctx_id = driver.cuCtxGetId(ctx)
        key = int(ctx_id) if err == driver.CUresult.CUDA_SUCCESS else int(ctx)
    else:
        key = int(ctx)
    func = _interleave_funcs.get(key)
    if func is None:
        err, mod = driver.cuModuleLoadData(_INTERLEAVE_UV_PTX)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to load interleave PTX: {err}')
        err, func = driver.cuModuleGetFunction(mod, b'interleave_uv')
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get interleave_uv function: {err}')
        _interleave_funcs[key] = func
    return func


def interleave_uv(
    u_ptr: int,
    u_pitch: int,
    v_ptr: int,
    v_pitch: int,
    uv_ptr: int,
    uv_pitch: int,
    chroma_width: int,
    chroma_height: int,
    stream: int = 0,
) -> None:
    """Interleave separate U and V planes into a single UV plane (for NV16).

    Each output row is: U0 V0 U1 V1 ... U(w-1) V(w-1).
    """
    from cuda.bindings import driver

    func = _get_interleave_func()
    block_x, block_y = 32, 8
    grid_x = (chroma_width + block_x - 1) // block_x
    grid_y = (chroma_height + block_y - 1) // block_y

    # Pack kernel arguments
    args = (
        ctypes.c_uint64(u_ptr),
        ctypes.c_int32(u_pitch),
        ctypes.c_uint64(v_ptr),
        ctypes.c_int32(v_pitch),
        ctypes.c_uint64(uv_ptr),
        ctypes.c_int32(uv_pitch),
        ctypes.c_int32(chroma_width),
        ctypes.c_int32(chroma_height),
    )
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    (err,) = driver.cuLaunchKernel(
        func, grid_x, grid_y, 1, block_x, block_y, 1, 0, stream, arg_ptrs, 0
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'interleave_uv kernel launch failed: {err}')


# Exact IEC 61966-2-1 sRGB transfer curves (piecewise: linear toe + power
# segment), applied in place on flat float32 data. NPP has no per-channel
# piecewise primitive, so these are small CUDA kernels; pow is computed as
# ex2(e * lg2(x)) (PTX approx intrinsics, ~1e-6 relative error — far below
# any output quantization). linear_to_srgb clamps its input to [0, 1] first,
# absorbing Lanczos over/undershoot from resizing in linear light.
_SRGB_PTX = b"""\
.version 7.0
.target sm_52
.address_size 64

.visible .entry srgb_to_linear(
 .param .u64 p0, .param .u32 p1)
{
 .reg .pred %p<2>; .reg .b32 %r<6>; .reg .b64 %rd<5>; .reg .f32 %f<8>;
 ld.param.u64 %rd1, [p0];
 ld.param.u32 %r1, [p1];
 mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
 mad.lo.s32 %r5, %r3, %r2, %r4;
 setp.ge.s32 %p1, %r5, %r1;
 @%p1 bra DONE1;
 cvta.to.global.u64 %rd2, %rd1;
 mul.wide.s32 %rd3, %r5, 4;
 add.s64 %rd4, %rd2, %rd3;
 ld.global.f32 %f1, [%rd4];
 mul.f32 %f2, %f1, 0f3D9E8391;
 add.f32 %f3, %f1, 0f3D6147AE;
 mul.f32 %f4, %f3, 0f3F72A76E;
 lg2.approx.f32 %f5, %f4;
 mul.f32 %f5, %f5, 0f4019999A;
 ex2.approx.f32 %f6, %f5;
 setp.le.f32 %p1, %f1, 0f3D25AEE6;
 selp.f32 %f7, %f2, %f6, %p1;
 st.global.f32 [%rd4], %f7;
DONE1:
 ret;
}

.visible .entry linear_to_srgb(
 .param .u64 p0, .param .u32 p1)
{
 .reg .pred %p<2>; .reg .b32 %r<6>; .reg .b64 %rd<5>; .reg .f32 %f<8>;
 ld.param.u64 %rd1, [p0];
 ld.param.u32 %r1, [p1];
 mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
 mad.lo.s32 %r5, %r3, %r2, %r4;
 setp.ge.s32 %p1, %r5, %r1;
 @%p1 bra DONE2;
 cvta.to.global.u64 %rd2, %rd1;
 mul.wide.s32 %rd3, %r5, 4;
 add.s64 %rd4, %rd2, %rd3;
 ld.global.f32 %f1, [%rd4];
 max.f32 %f1, %f1, 0f00000000;
 min.f32 %f1, %f1, 0f3F800000;
 mul.f32 %f2, %f1, 0f414EB852;
 lg2.approx.f32 %f3, %f1;
 mul.f32 %f3, %f3, 0f3ED55555;
 ex2.approx.f32 %f4, %f3;
 mul.f32 %f5, %f4, 0f3F870A3D;
 sub.f32 %f6, %f5, 0f3D6147AE;
 setp.le.f32 %p1, %f1, 0f3B4D2E1C;
 selp.f32 %f7, %f2, %f6, %p1;
 st.global.f32 [%rd4], %f7;
DONE2:
 ret;
}
"""

_srgb_funcs: dict[tuple[int, bytes], object] = {}


def _get_srgb_func(name: bytes):
    from cuda.bindings import driver

    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
        raise RuntimeError('srgb_curve_inplace requires a current CUDA context')
    if hasattr(driver, 'cuCtxGetId'):
        err, ctx_id = driver.cuCtxGetId(ctx)
        ctx_key = int(ctx_id) if err == driver.CUresult.CUDA_SUCCESS else int(ctx)
    else:
        ctx_key = int(ctx)
    key = (ctx_key, name)
    func = _srgb_funcs.get(key)
    if func is None:
        err, mod = driver.cuModuleLoadData(_SRGB_PTX)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to load sRGB PTX: {err}')
        err, func = driver.cuModuleGetFunction(mod, name)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get sRGB kernel {name!r}: {err}')
        _srgb_funcs[key] = func
    return func


def srgb_curve_inplace(ptr: int, n_floats: int, *, decode: bool, stream: int = 0) -> None:
    """Apply the exact sRGB transfer in place on flat float32 data.

    ``decode=True``: encoded sRGB → linear light. ``decode=False``: linear
    light (clamped to [0, 1]) → encoded sRGB.
    """
    from cuda.bindings import driver

    func = _get_srgb_func(b'srgb_to_linear' if decode else b'linear_to_srgb')
    block = 256
    grid = (n_floats + block - 1) // block
    args = (ctypes.c_uint64(ptr), ctypes.c_int32(n_floats))
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    (err,) = driver.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, stream, arg_ptrs, 0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'sRGB curve kernel launch failed: {err}')


# ── swscale-exact YUV → RGB8 ─────────────────────────────────────────

# libswscale's inverse coefficient table (ff_yuv2rgb_coeffs: crv, cbu, cgu,
# cgv in 16.16 fixed point), keyed like npp_bindings.LUMA_COEFFICIENTS.
_SWS_INV_COEFFS: dict[str, tuple[int, int, int, int]] = {
    'bt601': (104597, 132201, 25675, 53279),
    'bt709': (117489, 138438, 13975, 34925),
    'fcc': (104448, 132798, 24759, 53109),
    'smpte240m': (117579, 136230, 16907, 35559),
    'bt2020': (110013, 140363, 12277, 42626),
}


def _c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _round_to_int16(f: int) -> int:
    return (f + (1 << 15)) >> 16


def sws_yuv2rgb_coeffs(matrix: str, *, full_range: bool = False) -> dict[str, int]:
    """The 3.13 fixed-point constants of libswscale's 8-bit YUV → RGB path.

    Replicates ``ff_yuv2rgb_c_init_tables`` (libswscale/yuv2rgb.c) with unit
    contrast/saturation and zero brightness: limited range scales luma by
    255/219 with a 16-offset, full range scales the chroma coefficients by
    224/255; every constant is then rounded to 3.13 exactly like the C code
    (``roundToInt16(x << 13)``), so the kernel below reproduces swscale's
    output bit for bit.
    """
    crv, cbu, cgu, cgv = _SWS_INV_COEFFS[matrix]
    cgu, cgv = -cgu, -cgv
    cy, oy = 1 << 16, 0
    if full_range:
        crv, cbu, cgu, cgv = (_c_div(c * 224, 255) for c in (crv, cbu, cgu, cgv))
    else:
        cy = _c_div(cy * 255, 219)
        oy = 16 << 16
    return {
        'y_coff': _round_to_int16(cy << 13),
        'y_off': _round_to_int16(oy << 3),
        'vr': _round_to_int16(crv << 13),
        'ub': _round_to_int16(cbu << 13),
        'ug': _round_to_int16(cgu << 13),
        'vg': _round_to_int16(cgv << 13),
    }


# Per pixel, exactly like swscale's x86 yuv2rgb (yuv_2_rgb.asm, RGB24):
#   yt = (Y << 3) - y_off;  ut = (U << 3) - 1024;  vt = (V << 3) - 1024
#   yc = (yt * y_coff) >> 16                     (pmulhw: floor per term)
#   R = yc + ((vt * vr) >> 16);  B = yc + ((ut * ub) >> 16)
#   G = yc + (((ut * ug) >> 16) + ((vt * vg) >> 16));  clip to 0..255
# Chroma is taken from the nearest 4:2:0 sample (2x2 replication), which is
# also what swscale's unscaled yuv420p → rgb24 converter does. The 4:4:4
# variant applies the same arithmetic per pixel.
_YUV2RGB_SWS_PTX = b"""\
.version 7.0
.target sm_52
.address_size 64
.visible .entry nv12_to_rgb8_sws(
 .param .u64 nv12_to_rgb8_sws_param_0,
 .param .u32 nv12_to_rgb8_sws_param_1,
 .param .u64 nv12_to_rgb8_sws_param_2,
 .param .u32 nv12_to_rgb8_sws_param_3,
 .param .u64 nv12_to_rgb8_sws_param_4,
 .param .u32 nv12_to_rgb8_sws_param_5,
 .param .u32 nv12_to_rgb8_sws_param_6,
 .param .u32 nv12_to_rgb8_sws_param_7,
 .param .u32 nv12_to_rgb8_sws_param_8,
 .param .u32 nv12_to_rgb8_sws_param_9,
 .param .u32 nv12_to_rgb8_sws_param_10,
 .param .u32 nv12_to_rgb8_sws_param_11,
 .param .u32 nv12_to_rgb8_sws_param_12,
 .param .u32 nv12_to_rgb8_sws_param_13
)
{
 .reg .pred %p<4>;
 .reg .b16 %rs<7>;
 .reg .b32 %r<52>;
 .reg .b64 %rd<17>;
 ld.param.u64 %rd1, [nv12_to_rgb8_sws_param_0];
 ld.param.u32 %r3, [nv12_to_rgb8_sws_param_1];
 ld.param.u64 %rd2, [nv12_to_rgb8_sws_param_2];
 ld.param.u32 %r4, [nv12_to_rgb8_sws_param_3];
 ld.param.u64 %rd3, [nv12_to_rgb8_sws_param_4];
 ld.param.u32 %r5, [nv12_to_rgb8_sws_param_5];
 ld.param.u32 %r12, [nv12_to_rgb8_sws_param_6];
 ld.param.u32 %r13, [nv12_to_rgb8_sws_param_7];
 ld.param.u32 %r6, [nv12_to_rgb8_sws_param_8];
 ld.param.u32 %r7, [nv12_to_rgb8_sws_param_9];
 ld.param.u32 %r8, [nv12_to_rgb8_sws_param_10];
 ld.param.u32 %r9, [nv12_to_rgb8_sws_param_11];
 ld.param.u32 %r10, [nv12_to_rgb8_sws_param_12];
 ld.param.u32 %r11, [nv12_to_rgb8_sws_param_13];
 mov.u32 %r14, %ntid.x;
 mov.u32 %r15, %ctaid.x;
 mov.u32 %r16, %tid.x;
 mad.lo.s32 %r1, %r15, %r14, %r16;
 mov.u32 %r17, %ntid.y;
 mov.u32 %r18, %ctaid.y;
 mov.u32 %r19, %tid.y;
 mad.lo.s32 %r2, %r18, %r17, %r19;
 setp.ge.s32 %p1, %r1, %r12;
 setp.ge.s32 %p2, %r2, %r13;
 or.pred %p3, %p1, %p2;
 @%p3 bra $L__BB0_2;
 cvta.to.global.u64 %rd4, %rd1;
 mad.lo.s32 %r20, %r2, %r3, %r1;
 cvt.s64.s32 %rd5, %r20;
 add.s64 %rd6, %rd4, %rd5;
 ld.global.nc.u8 %rs1, [%rd6];
 shr.s32 %r21, %r2, 1;
 mul.lo.s32 %r22, %r21, %r4;
 cvt.s64.s32 %rd7, %r22;
 and.b32 %r23, %r1, -2;
 cvt.s64.s32 %rd8, %r23;
 add.s64 %rd9, %rd7, %rd8;
 cvta.to.global.u64 %rd10, %rd2;
 add.s64 %rd11, %rd10, %rd9;
 ld.global.nc.u8 %rs2, [%rd11];
 ld.global.nc.u8 %rs3, [%rd11+1];
 mul.wide.u16 %r24, %rs1, 8;
 sub.s32 %r25, %r24, %r7;
 mul.wide.u16 %r26, %rs2, 8;
 add.s32 %r27, %r26, -1024;
 mul.wide.u16 %r28, %rs3, 8;
 add.s32 %r29, %r28, -1024;
 mul.lo.s32 %r30, %r25, %r6;
 shr.s32 %r31, %r30, 16;
 mul.lo.s32 %r32, %r29, %r8;
 shr.s32 %r33, %r32, 16;
 add.s32 %r34, %r33, %r31;
 mul.lo.s32 %r35, %r27, %r10;
 shr.s32 %r36, %r35, 16;
 mul.lo.s32 %r37, %r29, %r11;
 shr.s32 %r38, %r37, 16;
 add.s32 %r39, %r36, %r31;
 add.s32 %r40, %r39, %r38;
 mul.lo.s32 %r41, %r27, %r9;
 shr.s32 %r42, %r41, 16;
 add.s32 %r43, %r42, %r31;
 max.s32 %r44, %r34, 0;
 min.s32 %r45, %r44, 255;
 max.s32 %r46, %r40, 0;
 min.s32 %r47, %r46, 255;
 max.s32 %r48, %r43, 0;
 min.s32 %r49, %r48, 255;
 mul.lo.s32 %r50, %r2, %r5;
 cvt.s64.s32 %rd12, %r50;
 mul.lo.s32 %r51, %r1, 3;
 cvt.s64.s32 %rd13, %r51;
 add.s64 %rd14, %rd12, %rd13;
 cvta.to.global.u64 %rd15, %rd3;
 add.s64 %rd16, %rd15, %rd14;
 st.global.u8 [%rd16], %r45;
 st.global.u8 [%rd16+1], %r47;
 st.global.u8 [%rd16+2], %r49;
$L__BB0_2:
 ret;
}
.visible .entry yuv444_to_rgb8_sws(
 .param .u64 yuv444_to_rgb8_sws_param_0,
 .param .u32 yuv444_to_rgb8_sws_param_1,
 .param .u64 yuv444_to_rgb8_sws_param_2,
 .param .u32 yuv444_to_rgb8_sws_param_3,
 .param .u64 yuv444_to_rgb8_sws_param_4,
 .param .u32 yuv444_to_rgb8_sws_param_5,
 .param .u64 yuv444_to_rgb8_sws_param_6,
 .param .u32 yuv444_to_rgb8_sws_param_7,
 .param .u32 yuv444_to_rgb8_sws_param_8,
 .param .u32 yuv444_to_rgb8_sws_param_9,
 .param .u32 yuv444_to_rgb8_sws_param_10,
 .param .u32 yuv444_to_rgb8_sws_param_11,
 .param .u32 yuv444_to_rgb8_sws_param_12,
 .param .u32 yuv444_to_rgb8_sws_param_13,
 .param .u32 yuv444_to_rgb8_sws_param_14,
 .param .u32 yuv444_to_rgb8_sws_param_15
)
{
 .reg .pred %p<4>;
 .reg .b16 %rs<7>;
 .reg .b32 %r<52>;
 .reg .b64 %rd<19>;
 ld.param.u64 %rd1, [yuv444_to_rgb8_sws_param_0];
 ld.param.u32 %r3, [yuv444_to_rgb8_sws_param_1];
 ld.param.u64 %rd2, [yuv444_to_rgb8_sws_param_2];
 ld.param.u32 %r4, [yuv444_to_rgb8_sws_param_3];
 ld.param.u64 %rd3, [yuv444_to_rgb8_sws_param_4];
 ld.param.u32 %r5, [yuv444_to_rgb8_sws_param_5];
 ld.param.u64 %rd4, [yuv444_to_rgb8_sws_param_6];
 ld.param.u32 %r6, [yuv444_to_rgb8_sws_param_7];
 ld.param.u32 %r13, [yuv444_to_rgb8_sws_param_8];
 ld.param.u32 %r14, [yuv444_to_rgb8_sws_param_9];
 ld.param.u32 %r7, [yuv444_to_rgb8_sws_param_10];
 ld.param.u32 %r8, [yuv444_to_rgb8_sws_param_11];
 ld.param.u32 %r9, [yuv444_to_rgb8_sws_param_12];
 ld.param.u32 %r10, [yuv444_to_rgb8_sws_param_13];
 ld.param.u32 %r11, [yuv444_to_rgb8_sws_param_14];
 ld.param.u32 %r12, [yuv444_to_rgb8_sws_param_15];
 mov.u32 %r15, %ntid.x;
 mov.u32 %r16, %ctaid.x;
 mov.u32 %r17, %tid.x;
 mad.lo.s32 %r1, %r16, %r15, %r17;
 mov.u32 %r18, %ntid.y;
 mov.u32 %r19, %ctaid.y;
 mov.u32 %r20, %tid.y;
 mad.lo.s32 %r2, %r19, %r18, %r20;
 setp.ge.s32 %p1, %r1, %r13;
 setp.ge.s32 %p2, %r2, %r14;
 or.pred %p3, %p1, %p2;
 @%p3 bra $L__BB1_2;
 cvta.to.global.u64 %rd5, %rd1;
 mad.lo.s32 %r21, %r2, %r3, %r1;
 cvt.s64.s32 %rd6, %r21;
 add.s64 %rd7, %rd5, %rd6;
 ld.global.nc.u8 %rs1, [%rd7];
 mad.lo.s32 %r22, %r2, %r4, %r1;
 cvt.s64.s32 %rd8, %r22;
 cvta.to.global.u64 %rd9, %rd2;
 add.s64 %rd10, %rd9, %rd8;
 ld.global.nc.u8 %rs2, [%rd10];
 mad.lo.s32 %r23, %r2, %r5, %r1;
 cvt.s64.s32 %rd11, %r23;
 cvta.to.global.u64 %rd12, %rd3;
 add.s64 %rd13, %rd12, %rd11;
 ld.global.nc.u8 %rs3, [%rd13];
 mul.wide.u16 %r24, %rs1, 8;
 sub.s32 %r25, %r24, %r8;
 mul.wide.u16 %r26, %rs2, 8;
 add.s32 %r27, %r26, -1024;
 mul.wide.u16 %r28, %rs3, 8;
 add.s32 %r29, %r28, -1024;
 mul.lo.s32 %r30, %r25, %r7;
 shr.s32 %r31, %r30, 16;
 mul.lo.s32 %r32, %r29, %r9;
 shr.s32 %r33, %r32, 16;
 add.s32 %r34, %r33, %r31;
 mul.lo.s32 %r35, %r27, %r11;
 shr.s32 %r36, %r35, 16;
 mul.lo.s32 %r37, %r29, %r12;
 shr.s32 %r38, %r37, 16;
 add.s32 %r39, %r36, %r31;
 add.s32 %r40, %r39, %r38;
 mul.lo.s32 %r41, %r27, %r10;
 shr.s32 %r42, %r41, 16;
 add.s32 %r43, %r42, %r31;
 max.s32 %r44, %r34, 0;
 min.s32 %r45, %r44, 255;
 max.s32 %r46, %r40, 0;
 min.s32 %r47, %r46, 255;
 max.s32 %r48, %r43, 0;
 min.s32 %r49, %r48, 255;
 mul.lo.s32 %r50, %r2, %r6;
 cvt.s64.s32 %rd14, %r50;
 mul.lo.s32 %r51, %r1, 3;
 cvt.s64.s32 %rd15, %r51;
 add.s64 %rd16, %rd14, %rd15;
 cvta.to.global.u64 %rd17, %rd4;
 add.s64 %rd18, %rd17, %rd16;
 st.global.u8 [%rd18], %r45;
 st.global.u8 [%rd18+1], %r47;
 st.global.u8 [%rd18+2], %r49;
$L__BB1_2:
 ret;
}
"""

_yuv2rgb_funcs: dict[tuple[int, bytes], object] = {}


def _get_yuv2rgb_func(name: bytes):
    from cuda.bindings import driver

    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
        raise RuntimeError('yuv_to_rgb8_sws requires a current CUDA context')
    if hasattr(driver, 'cuCtxGetId'):
        err, ctx_id = driver.cuCtxGetId(ctx)
        ctx_key = int(ctx_id) if err == driver.CUresult.CUDA_SUCCESS else int(ctx)
    else:
        ctx_key = int(ctx)
    key = (ctx_key, name)
    func = _yuv2rgb_funcs.get(key)
    if func is None:
        err, mod = driver.cuModuleLoadData(_YUV2RGB_SWS_PTX)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to load YUV→RGB PTX: {err}')
        err, func = driver.cuModuleGetFunction(mod, name)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get YUV→RGB kernel {name!r}: {err}')
        _yuv2rgb_funcs[key] = func
    return func


def yuv_to_rgb8_sws(
    planes: list[tuple[int, int]],
    dst_ptr: int,
    dst_pitch: int,
    width: int,
    height: int,
    coeffs: dict[str, int],
    *,
    stream: int = 0,
) -> None:
    """Convert 8-bit YUV to packed RGB8 with swscale's exact arithmetic.

    ``planes`` is ``[(y_ptr, y_pitch), (uv_ptr, uv_pitch)]`` for NV12 or
    ``[(y, ..), (u, ..), (v, ..)]`` for planar 4:4:4; ``coeffs`` comes from
    :func:`sws_yuv2rgb_coeffs`.
    """
    from cuda.bindings import driver

    if len(planes) == 2:
        func = _get_yuv2rgb_func(b'nv12_to_rgb8_sws')
    elif len(planes) == 3:
        func = _get_yuv2rgb_func(b'yuv444_to_rgb8_sws')
    else:
        raise ValueError(f'Expected 2 (NV12) or 3 (4:4:4) planes, got {len(planes)}')
    args = []
    for ptr, pitch in planes:
        args += [ctypes.c_uint64(ptr), ctypes.c_int32(pitch)]
    args += [ctypes.c_uint64(dst_ptr), ctypes.c_int32(dst_pitch)]
    args += [ctypes.c_int32(width), ctypes.c_int32(height)]
    args += [ctypes.c_int32(coeffs[k]) for k in ('y_coff', 'y_off', 'vr', 'ub', 'ug', 'vg')]
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    bx, by = 32, 8
    grid_x, grid_y = (width + bx - 1) // bx, (height + by - 1) // by
    (err,) = driver.cuLaunchKernel(func, grid_x, grid_y, 1, bx, by, 1, 0, stream, arg_ptrs, 0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'YUV→RGB kernel launch failed: {err}')
