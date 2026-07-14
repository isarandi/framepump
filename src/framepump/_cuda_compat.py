"""CUDA driver-API helpers: version compatibility and context ownership.

Context ownership convention (applies to all CUDA code in framepump):

- framepump never leaves a different CUDA context current than it found.
  Code that needs a context pushes it via ``cuda_ctx_pushed`` for the
  duration of its own driver/library calls and pops it on exit.
- Components that need a context of their own retain the device's *primary*
  context (``retain_primary_context``) instead of creating a private one, so
  they interoperate with callers that use the primary context themselves
  (torch, cupy, other framepump components). Each retain has exactly one
  release.
- Every owned GPU resource carries its owning context (or device, for
  primary-context retain/release). Deleters that may run on arbitrary
  threads push that context before freeing and release their retain after,
  so frees cannot silently fail for lack of a current context.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager

from cuda.bindings import driver
from cuda.bindings.driver import cuCtxCreate as _cuCtxCreate


@contextmanager
def cuda_ctx_pushed(ctx):
    """Make ``ctx`` current for the duration of the block, then restore.

    Uses the driver's context stack, so the caller's current context (or the
    absence of one) is exactly restored on exit, on any thread.
    """
    (err,) = driver.cuCtxPushCurrent(ctx)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'Failed to push CUDA context: {err}')
    try:
        yield
    finally:
        driver.cuCtxPopCurrent()


def retain_primary_context(gpu: int):
    """Initialize CUDA and retain the primary context of device ``gpu``.

    Does not make the context current. The caller must balance with
    ``cuDevicePrimaryCtxRelease(device)``.

    Returns:
        Tuple of (device handle, context handle).
    """
    (err,) = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'Failed to initialize CUDA: {err}')
    err, device = driver.cuDeviceGet(gpu)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'Failed to get CUDA device {gpu}: {err}')
    err, ctx = driver.cuDevicePrimaryCtxRetain(device)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'Failed to retain primary context on device {gpu}: {err}')
    return device, ctx


def cuCtxCreate(flags, device):
    """Create CUDA context, supporting both cuda-python 12.x and 13.x APIs.

    The installed signature is detected once at import, so a genuine
    bad-argument TypeError from the driver binding surfaces unmasked
    instead of triggering a confusing retry with the other signature.

    Args:
        flags: Context creation flags (e.g., 0 for default).
        device: CUDA device handle.

    Returns:
        Tuple of (error_code, context).
    """
    if _CTX_CREATE_TAKES_PARAMS:
        # cuda-python 13+: cuCtxCreate(ctxCreateParams, flags, device)
        return _cuCtxCreate(None, flags, device)
    # cuda-python 12.x: cuCtxCreate(flags, device)
    return _cuCtxCreate(flags, device)


def _detect_ctx_create_takes_params() -> bool:
    # Checking for driver.CUctxCreateParams would be wrong: 12.x bindings
    # already ship that type while keeping the two-argument signature.
    try:
        parameters = inspect.signature(_cuCtxCreate).parameters
    except (TypeError, ValueError):
        doc = (_cuCtxCreate.__doc__ or '').strip()
        first_line = doc.splitlines()[0] if doc else ''
        return first_line.count(',') >= 2
    return len(parameters) >= 3


_CTX_CREATE_TAKES_PARAMS = _detect_ctx_create_takes_params()
