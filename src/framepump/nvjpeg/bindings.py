"""ctypes bindings for nvJPEG library."""

import ctypes
from ctypes import c_int, c_void_p, c_size_t, c_uint, POINTER, Structure

# Load library
try:
    _lib = ctypes.CDLL('libnvjpeg.so')
except OSError:
    _lib = None

# Constants
NVJPEG_MAX_COMPONENT = 4

# nvjpegStatus_t
NVJPEG_STATUS_SUCCESS = 0
NVJPEG_STATUS_NOT_INITIALIZED = 1
NVJPEG_STATUS_INVALID_PARAMETER = 2
NVJPEG_STATUS_BAD_JPEG = 3
NVJPEG_STATUS_JPEG_NOT_SUPPORTED = 4
NVJPEG_STATUS_ALLOCATOR_FAILURE = 5
NVJPEG_STATUS_EXECUTION_FAILED = 6
NVJPEG_STATUS_ARCH_MISMATCH = 7
NVJPEG_STATUS_INTERNAL_ERROR = 8
NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9
NVJPEG_STATUS_INCOMPLETE_BITSTREAM = 10

# nvjpegOutputFormat_t
NVJPEG_OUTPUT_UNCHANGED = 0
NVJPEG_OUTPUT_YUV = 1
NVJPEG_OUTPUT_Y = 2
NVJPEG_OUTPUT_RGB = 3
NVJPEG_OUTPUT_BGR = 4
NVJPEG_OUTPUT_RGBI = 5  # Interleaved RGB
NVJPEG_OUTPUT_BGRI = 6  # Interleaved BGR

# nvjpegChromaSubsampling_t
NVJPEG_CSS_444 = 0
NVJPEG_CSS_422 = 1
NVJPEG_CSS_420 = 2
NVJPEG_CSS_440 = 3
NVJPEG_CSS_411 = 4
NVJPEG_CSS_410 = 5
NVJPEG_CSS_GRAY = 6
NVJPEG_CSS_UNKNOWN = -1


class nvjpegImage_t(Structure):
    """Output image descriptor."""
    _fields_ = [
        ('channel', c_void_p * NVJPEG_MAX_COMPONENT),  # Device pointers
        ('pitch', c_size_t * NVJPEG_MAX_COMPONENT),
    ]


# nvjpegBackend_t - decoder implementations
NVJPEG_BACKEND_DEFAULT = 0
NVJPEG_BACKEND_HYBRID = 1  # CPU Huffman, GPU IDCT (best for phased)
NVJPEG_BACKEND_GPU_HYBRID = 2  # GPU Huffman, GPU IDCT

# Opaque handles
nvjpegHandle_t = c_void_p
nvjpegJpegState_t = c_void_p
nvjpegDecodeParams_t = c_void_p
nvjpegJpegStream_t = c_void_p
nvjpegJpegDecoder_t = c_void_p
nvjpegBufferPinned_t = c_void_p
nvjpegBufferDevice_t = c_void_p

if _lib is not None:
    # Function prototypes
    _lib.nvjpegCreateSimple.argtypes = [POINTER(nvjpegHandle_t)]
    _lib.nvjpegCreateSimple.restype = c_int

    _lib.nvjpegDestroy.argtypes = [nvjpegHandle_t]
    _lib.nvjpegDestroy.restype = c_int

    _lib.nvjpegJpegStateCreate.argtypes = [nvjpegHandle_t, POINTER(nvjpegJpegState_t)]
    _lib.nvjpegJpegStateCreate.restype = c_int

    _lib.nvjpegJpegStateDestroy.argtypes = [nvjpegJpegState_t]
    _lib.nvjpegJpegStateDestroy.restype = c_int

    _lib.nvjpegGetImageInfo.argtypes = [
        nvjpegHandle_t,
        POINTER(ctypes.c_ubyte),  # data
        c_size_t,  # length
        POINTER(c_int),  # nComponents
        POINTER(c_int),  # subsampling
        POINTER(c_int),  # widths (array)
        POINTER(c_int),  # heights (array)
    ]
    _lib.nvjpegGetImageInfo.restype = c_int

    _lib.nvjpegDecodeParamsCreate.argtypes = [nvjpegHandle_t, POINTER(nvjpegDecodeParams_t)]
    _lib.nvjpegDecodeParamsCreate.restype = c_int

    _lib.nvjpegDecodeParamsDestroy.argtypes = [nvjpegDecodeParams_t]
    _lib.nvjpegDecodeParamsDestroy.restype = c_int

    _lib.nvjpegDecodeParamsSetOutputFormat.argtypes = [nvjpegDecodeParams_t, c_int]
    _lib.nvjpegDecodeParamsSetOutputFormat.restype = c_int

    _lib.nvjpegDecode.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegState_t,
        POINTER(ctypes.c_ubyte),  # data
        c_size_t,  # length
        c_int,  # output_format (nvjpegOutputFormat_t)
        POINTER(nvjpegImage_t),  # destination
        c_void_p,  # stream (cudaStream_t)
    ]
    _lib.nvjpegDecode.restype = c_int

    # Newer API with decode params
    _lib.nvjpegDecodeJpeg.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegState_t,
        nvjpegDecodeParams_t,
        POINTER(ctypes.c_ubyte),  # data
        c_size_t,  # length
        POINTER(nvjpegImage_t),  # destination
        c_void_p,  # stream
    ]
    _lib.nvjpegDecodeJpeg.restype = c_int

    # ========================================================================
    # Phased Decoding API (for async pipeline)
    # ========================================================================

    # JPEG stream (holds parsed JPEG data)
    _lib.nvjpegJpegStreamCreate.argtypes = [nvjpegHandle_t, POINTER(nvjpegJpegStream_t)]
    _lib.nvjpegJpegStreamCreate.restype = c_int

    _lib.nvjpegJpegStreamDestroy.argtypes = [nvjpegJpegStream_t]
    _lib.nvjpegJpegStreamDestroy.restype = c_int

    _lib.nvjpegJpegStreamParse.argtypes = [
        nvjpegHandle_t,
        POINTER(ctypes.c_ubyte),  # data
        c_size_t,  # length
        c_int,  # save_metadata
        c_int,  # save_stream
        nvjpegJpegStream_t,
    ]
    _lib.nvjpegJpegStreamParse.restype = c_int

    # Decoder handle (different from state - for phased API)
    _lib.nvjpegDecoderCreate.argtypes = [
        nvjpegHandle_t,
        c_int,  # nvjpegBackend_t
        POINTER(nvjpegJpegDecoder_t),
    ]
    _lib.nvjpegDecoderCreate.restype = c_int

    _lib.nvjpegDecoderDestroy.argtypes = [nvjpegJpegDecoder_t]
    _lib.nvjpegDecoderDestroy.restype = c_int

    _lib.nvjpegDecoderStateCreate.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegDecoder_t,
        POINTER(nvjpegJpegState_t),
    ]
    _lib.nvjpegDecoderStateCreate.restype = c_int

    # Phased decode functions
    _lib.nvjpegDecodeJpegHost.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegDecoder_t,
        nvjpegJpegState_t,
        nvjpegDecodeParams_t,
        nvjpegJpegStream_t,
    ]
    _lib.nvjpegDecodeJpegHost.restype = c_int

    _lib.nvjpegDecodeJpegTransferToDevice.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegDecoder_t,
        nvjpegJpegState_t,
        nvjpegJpegStream_t,
        c_void_p,  # cudaStream_t
    ]
    _lib.nvjpegDecodeJpegTransferToDevice.restype = c_int

    _lib.nvjpegDecodeJpegDevice.argtypes = [
        nvjpegHandle_t,
        nvjpegJpegDecoder_t,
        nvjpegJpegState_t,
        POINTER(nvjpegImage_t),  # destination
        c_void_p,  # cudaStream_t
    ]
    _lib.nvjpegDecodeJpegDevice.restype = c_int

    # Get stream info (for getting dimensions after parse)
    _lib.nvjpegJpegStreamGetFrameDimensions.argtypes = [
        nvjpegJpegStream_t,
        POINTER(c_uint),  # width
        POINTER(c_uint),  # height
    ]
    _lib.nvjpegJpegStreamGetFrameDimensions.restype = c_int

    _lib.nvjpegJpegStreamGetChromaSubsampling.argtypes = [
        nvjpegJpegStream_t,
        POINTER(c_int),  # subsampling
    ]
    _lib.nvjpegJpegStreamGetChromaSubsampling.restype = c_int

    # ====================================================================
    # Internal Buffers (required for phased decoding)
    # ====================================================================

    # Pinned (page-locked) host buffer for CPU→GPU transfer
    _lib.nvjpegBufferPinnedCreate.argtypes = [
        nvjpegHandle_t,
        c_void_p,  # pinned_allocator (NULL for default)
        POINTER(nvjpegBufferPinned_t),
    ]
    _lib.nvjpegBufferPinnedCreate.restype = c_int

    _lib.nvjpegBufferPinnedDestroy.argtypes = [nvjpegBufferPinned_t]
    _lib.nvjpegBufferPinnedDestroy.restype = c_int

    # Device buffer for intermediate GPU storage
    _lib.nvjpegBufferDeviceCreate.argtypes = [
        nvjpegHandle_t,
        c_void_p,  # device_allocator (NULL for default)
        POINTER(nvjpegBufferDevice_t),
    ]
    _lib.nvjpegBufferDeviceCreate.restype = c_int

    _lib.nvjpegBufferDeviceDestroy.argtypes = [nvjpegBufferDevice_t]
    _lib.nvjpegBufferDeviceDestroy.restype = c_int

    # Attach buffers to decoder state
    _lib.nvjpegStateAttachPinnedBuffer.argtypes = [
        nvjpegJpegState_t,
        nvjpegBufferPinned_t,
    ]
    _lib.nvjpegStateAttachPinnedBuffer.restype = c_int

    _lib.nvjpegStateAttachDeviceBuffer.argtypes = [
        nvjpegJpegState_t,
        nvjpegBufferDevice_t,
    ]
    _lib.nvjpegStateAttachDeviceBuffer.restype = c_int


def nvjpeg_status_message(status, context=''):
    """Convert nvJPEG status code to error message."""
    messages = {
        NVJPEG_STATUS_SUCCESS: 'Success',
        NVJPEG_STATUS_NOT_INITIALIZED: 'Not initialized',
        NVJPEG_STATUS_INVALID_PARAMETER: 'Invalid parameter',
        NVJPEG_STATUS_BAD_JPEG: 'Bad JPEG',
        NVJPEG_STATUS_JPEG_NOT_SUPPORTED: 'JPEG not supported',
        NVJPEG_STATUS_ALLOCATOR_FAILURE: 'Allocator failure',
        NVJPEG_STATUS_EXECUTION_FAILED: 'Execution failed',
        NVJPEG_STATUS_ARCH_MISMATCH: 'Architecture mismatch',
        NVJPEG_STATUS_INTERNAL_ERROR: 'Internal error',
        NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: 'Implementation not supported',
        NVJPEG_STATUS_INCOMPLETE_BITSTREAM: 'Incomplete bitstream',
    }
    msg = messages.get(status, f'Unknown error {status}')
    if context:
        return f'{context}: {msg}'
    return msg