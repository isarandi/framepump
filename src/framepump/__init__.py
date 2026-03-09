"""High-performance video processing built on PyAV.

This library provides lazy, sliceable video frame access via VideoFrames
and threaded video writing via VideoWriter. For GPU encoding, GLVideoWriter
offers zero-copy OpenGL texture to video encoding using NVENC.
"""

from ._core import (
    VideoFrames,
    get_duration,
    get_fps,
    has_audio,
    num_frames,
    video_extents,
)
from ._pyav import FramePumpError, NoAudioStreamError, VideoDecodeError, VideoEncodeError
from .encoder_config import EncoderConfig

from .video_writing import AbstractVideoWriter, VideoWriter, trim_video, video_audio_mux
from .video_writing_gl import GLVideoWriter

try:
    from .cuda_video_writer import JpegVideoWriterCUDA
except ImportError:
    JpegVideoWriterCUDA = None

try:
    from ._cuda_frames import VideoFramesCuda
except ImportError:
    VideoFramesCuda = None

try:
    from ._cuda_gl import CudaToGLUploader
except ImportError:
    CudaToGLUploader = None

try:
    from ._version import __version__
except ImportError:
    __version__ = 'unknown'

__all__ = [
    'VideoFrames',
    'FramePumpError',
    'VideoDecodeError',
    'VideoEncodeError',
    'NoAudioStreamError',
    'AbstractVideoWriter',
    'VideoWriter',
    'GLVideoWriter',
    'JpegVideoWriterCUDA',
    'VideoFramesCuda',
    'CudaToGLUploader',
    'EncoderConfig',
    'get_fps',
    'get_duration',
    'num_frames',
    'video_extents',
    'trim_video',
    'video_audio_mux',
    'has_audio',
    '__version__',
]

# Set __module__ to this module for sphinx-codeautolink to resolve references.
# Preserve original module in _module_original_ for source code links.
for _name in __all__:
    if _name == '__version__':
        continue
    _obj = globals()[_name]
    if hasattr(_obj, '__module__'):
        _obj._module_original_ = _obj.__module__
        _obj.__module__ = __name__
