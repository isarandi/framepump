"""Compatibility layer for cuda-python 12.x and 13.x API differences."""

from __future__ import annotations

import inspect

from cuda.bindings.driver import cuCtxCreate as _cuCtxCreate


def resolve_gpu_device(gpu: bool | int) -> int:
    """Resolve a gpu parameter to a CUDA device ordinal.

    Args:
        gpu: True for auto-detect (device 0), or an explicit device ordinal.

    Returns:
        Device ordinal (int).
    """
    if gpu is True:
        return 0
    return int(gpu)


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
