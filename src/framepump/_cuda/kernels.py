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
