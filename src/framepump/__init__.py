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
from ._pyav import (
    FilterConfigError,
    FramePumpError,
    IndexBuildError,
    NoAudioStreamError,
    NoVideoStreamError,
    UnsupportedCodecError,
    VideoDecodeError,
    VideoEncodeError,
)
from .encoder_config import EncoderConfig

from .video_writing import AbstractVideoWriter, VideoWriter, trim_video, video_audio_mux
from .video_writing_gl import GLVideoWriter


def _make_cuda_stub(class_name: str, requirements: str) -> type:
    """Placeholder for a CUDA-only class on machines without the CUDA stack.

    Instantiating it raises a helpful ImportError instead of the bare
    "'NoneType' object is not callable" a None placeholder would give.
    """

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f'{class_name} requires {requirements}. ' f'Install the CUDA dependencies to use it.'
        )

    return type(class_name, (), {'__init__': __init__, '__doc__': f'{class_name} (unavailable)'})


try:
    from .cuda_video_writer import JpegVideoWriterCUDA
except ImportError:
    JpegVideoWriterCUDA = _make_cuda_stub(
        'JpegVideoWriterCUDA', 'cuda-python and the nvJPEG/NVENC libraries (CUDA toolkit)'
    )

try:
    from ._cuda_frames import VideoFramesCuda
except ImportError:
    VideoFramesCuda = _make_cuda_stub('VideoFramesCuda', 'cuda-python and PyNvVideoCodec')

try:
    from ._cuda_gl import CudaToGLUploader
except ImportError:
    CudaToGLUploader = _make_cuda_stub('CudaToGLUploader', 'cuda-python')

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
    'NoVideoStreamError',
    'UnsupportedCodecError',
    'IndexBuildError',
    'FilterConfigError',
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
