from .encoder import NvencEncoder, EncodedPacket
from .exceptions import (
    NvencError,
    NvencNotAvailable,
    TextureFormatError,
    EncoderNotInitialized,
)

# CUDA-path encoder (optional, requires additional dependencies)
try:
    from .cuda_encoder import NvencCudaEncoder
    HAS_CUDA_PATH = True
except ImportError:
    NvencCudaEncoder = None
    HAS_CUDA_PATH = False

__all__ = [
    'NvencEncoder',
    'NvencCudaEncoder',
    'EncodedPacket',
    'HAS_CUDA_PATH',
    'NvencError',
    'NvencNotAvailable',
    'TextureFormatError',
    'EncoderNotInitialized',
]