"""Compatibility layer for cuda-python 12.x and 13.x API differences."""

from __future__ import annotations

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

    Args:
        flags: Context creation flags (e.g., 0 for default).
        device: CUDA device handle.

    Returns:
        Tuple of (error_code, context).
    """
    try:
        # cuda-python 13+: cuCtxCreate(ctxCreateParams, flags, device)
        return _cuCtxCreate(None, flags, device)
    except TypeError:
        # cuda-python 12.x: cuCtxCreate(flags, device)
        return _cuCtxCreate(flags, device)
