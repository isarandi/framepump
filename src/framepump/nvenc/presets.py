"""Shared presets and constants for NVENC encoders."""

from fractions import Fraction

from .bindings import NV_ENC_PARAMS_RC_VBR

# File extensions for each output mode
RAW_EXTENSIONS = {'.h264', '.264', '.avc', '.hevc', '.h265', '.265'}
CONTAINER_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}

# Default preset uses VBR with targetQuality (CQ mode)
DEFAULT_PRESET = {
    'gop_length': 250,
    'rate_control': NV_ENC_PARAMS_RC_VBR,
}


def float_to_rational(fps: float) -> Fraction:
    """Convert float fps to Fraction for NVENC frameRateNum/Den."""
    return Fraction(fps).limit_denominator(10000)