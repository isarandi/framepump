"""Encoder configuration for video writing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

X264_TO_NVENC = {
    'ultrafast': 'p1',
    'superfast': 'p1',
    'veryfast': 'p2',
    'faster': 'p3',
    'fast': 'p3',
    'medium': 'p4',
    'slow': 'p5',
    'slower': 'p6',
    'veryslow': 'p7',
}

NVENC_TO_X264 = {
    'p1': 'veryfast',
    'p2': 'faster',
    'p3': 'fast',
    'p4': 'medium',
    'p5': 'slow',
    'p6': 'slower',
    'p7': 'veryslow',
}


@dataclass(frozen=True)
class EncoderConfig:
    """Video encoder settings.

    Attributes:
        crf: Constant Rate Factor. Lower = better quality, larger files.
            Range 0-51, default 15. Visually lossless ~17-18.
        preset: Encoder effort preset.
            - NVENC (GPU): 'p1' (fastest) to 'p7' (slowest, best compression)
            - libx264 (CPU): 'ultrafast', 'veryfast', 'fast', 'medium',
                            'slow', 'slower', 'veryslow'
            Auto-translated between NVENC and libx264 names.
            Default: None (uses encoder default)
        bframes: Number of B-frames. More = better compression for static
            content. Range 0-4, default 2.
        gop: Group of Pictures size (keyframe interval). Larger = better
            compression but slower seeking. Default 250.
        codec: 'h264' or 'hevc'. HEVC is ~30% smaller but slower to decode.
    """

    crf: int = 15
    preset: str | None = None
    bframes: int = 2
    gop: int = 250
    codec: Literal['h264', 'hevc'] = 'h264'

    def with_overrides(self, **kwargs) -> EncoderConfig:
        """Return a new config with the given overrides."""
        return replace(self, **kwargs)

    def resolve_preset(self, gpu: bool | int) -> str | None:
        """Resolve preset name for the target encoder (NVENC or libx264)."""
        if self.preset is None:
            return None

        if gpu:
            # Target is NVENC, translate libx264 names if needed
            if self.preset in NVENC_TO_X264:
                return self.preset  # Already NVENC format
            return X264_TO_NVENC.get(self.preset, self.preset)
        else:
            # Target is libx264, translate NVENC names if needed
            if self.preset in X264_TO_NVENC:
                return self.preset  # Already libx264 format
            return NVENC_TO_X264.get(self.preset, self.preset)

    def get_codec_name(self, gpu: bool | int) -> str:
        """Get the FFmpeg codec name."""
        if gpu:
            return 'hevc_nvenc' if self.codec == 'hevc' else 'h264_nvenc'
        else:
            return 'libx265' if self.codec == 'hevc' else 'libx264'

    def build_options(self, gpu: bool | int) -> dict[str, str]:
        """Build FFmpeg encoder options dict."""
        preset = self.resolve_preset(gpu)
        options = {}

        if gpu:
            options['rc'] = 'vbr'
            options['cq'] = str(self.crf)
            if type(gpu) is int:
                options['gpu'] = str(gpu)
        else:
            options['crf'] = str(self.crf)

        if preset is not None:
            options['preset'] = preset

        options['bf'] = str(self.bframes)
        options['g'] = str(self.gop)

        return options