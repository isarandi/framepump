"""nvJPEG GPU JPEG decoder."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_int, c_size_t, c_ubyte
from typing import Union

import numpy as np
from cuda.bindings import driver

from .._cuda_compat import cuCtxCreate
from .bindings import (
    _lib,
    nvjpegHandle_t,
    nvjpegJpegState_t,
    nvjpegDecodeParams_t,
    nvjpegJpegStream_t,
    nvjpegJpegDecoder_t,
    nvjpegBufferPinned_t,
    nvjpegBufferDevice_t,
    nvjpegImage_t,
    nvjpeg_status_message,
    NVJPEG_STATUS_SUCCESS,
    NVJPEG_OUTPUT_YUV,
    NVJPEG_MAX_COMPONENT,
    NVJPEG_BACKEND_HYBRID,
)

# Type for JPEG data - numpy array or bytes
JpegData = Union[np.ndarray, bytes]
BytePointer = POINTER(c_ubyte)


def _get_data_ptr(data: JpegData) -> BytePointer:
    """Get ctypes pointer to data without copying."""
    if isinstance(data, np.ndarray):
        return data.ctypes.data_as(POINTER(ctypes.c_ubyte))
    else:
        # For bytes, we need to wrap in numpy array (zero-copy via buffer protocol)
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr.ctypes.data_as(POINTER(ctypes.c_ubyte))


class NvjpegDecoder:
    """GPU JPEG decoder using nvJPEG.

    Decodes JPEG data directly into user-provided CUDA device buffers (zero-copy).

    Example:
        >>> decoder = NvjpegDecoder()
        >>> width, height = decoder.decode_yuv_into(jpeg_bytes, y_ptr, u_ptr, v_ptr, ...)
    """

    def __init__(self, gpu: int = 0):
        if _lib is None:
            raise ImportError(
                'nvJPEG library not available. '
                'Ensure libnvjpeg.so is installed (part of CUDA toolkit).'
            )
        self._handle = nvjpegHandle_t()
        self._state = nvjpegJpegState_t()
        self._params = nvjpegDecodeParams_t()
        self._closed = False
        self._cuda_ctx = None

        # Initialize CUDA
        driver.cuInit(0)
        err, device = driver.cuDeviceGet(gpu)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to get CUDA device {gpu}: {err}')
        err, self._cuda_ctx = cuCtxCreate(0, device)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Failed to create CUDA context: {err}')

        # Initialize nvJPEG
        self._check(
            _lib.nvjpegCreateSimple(byref(self._handle)),
            'Failed to create nvJPEG handle',
        )
        self._check(
            _lib.nvjpegJpegStateCreate(self._handle, byref(self._state)),
            'Failed to create JPEG state',
        )
        self._check(
            _lib.nvjpegDecodeParamsCreate(self._handle, byref(self._params)),
            'Failed to create decode params',
        )
        self._check(
            _lib.nvjpegDecodeParamsSetOutputFormat(self._params, NVJPEG_OUTPUT_YUV),
            'Failed to set output format',
        )

    def _cleanup_partial(self) -> None:
        """Clean up partially initialized nvJPEG resources."""
        if self._params:
            _lib.nvjpegDecodeParamsDestroy(self._params)
            self._params = nvjpegDecodeParams_t()
        if self._state:
            _lib.nvjpegJpegStateDestroy(self._state)
            self._state = nvjpegJpegState_t()
        if self._handle:
            _lib.nvjpegDestroy(self._handle)
            self._handle = nvjpegHandle_t()

    def _check(self, status: int, msg: str) -> None:
        """Check nvJPEG status, cleanup and raise on error."""
        if status != NVJPEG_STATUS_SUCCESS:
            self._cleanup_partial()
            if self._cuda_ctx:
                driver.cuCtxDestroy(self._cuda_ctx)
                self._cuda_ctx = None
            raise RuntimeError(nvjpeg_status_message(status, msg))

    def get_image_info(self, jpeg_data: JpegData) -> tuple[int, int, int, int]:
        """Get image dimensions and chroma subsampling from JPEG data.

        Returns:
            Tuple of (width, height, num_components, subsampling)
            where subsampling is one of NVJPEG_CSS_* constants:
            - 0: 4:4:4
            - 1: 4:2:2
            - 2: 4:2:0
            - 6: grayscale
        """
        n_components = c_int()
        subsampling = c_int()
        widths = (c_int * NVJPEG_MAX_COMPONENT)()
        heights = (c_int * NVJPEG_MAX_COMPONENT)()

        status = _lib.nvjpegGetImageInfo(
            self._handle,
            _get_data_ptr(jpeg_data),
            c_size_t(len(jpeg_data)),
            byref(n_components),
            byref(subsampling),
            widths,
            heights,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to get image info'))

        return widths[0], heights[0], n_components.value, subsampling.value

    def decode_yuv_into(
        self,
        jpeg_data: JpegData,
        y_ptr: int,
        u_ptr: int,
        v_ptr: int,
        y_pitch: int,
        u_pitch: int | None = None,
        v_pitch: int | None = None,
        stream: int | None = None,
    ) -> tuple[int, int]:
        """Decode JPEG to YUV directly into user-provided buffers (zero-copy).

        Args:
            jpeg_data: JPEG file data as numpy array or bytes.
            y_ptr: CUDA device pointer for Y plane.
            u_ptr: CUDA device pointer for U plane.
            v_ptr: CUDA device pointer for V plane.
            y_pitch: Pitch (stride) for Y plane in bytes.
            u_pitch: Pitch for U plane. If None, uses y_pitch (for YUV444).
            v_pitch: Pitch for V plane. If None, uses y_pitch (for YUV444).
            stream: Optional CUDA stream for async decode. None = default stream.

        Returns:
            Tuple of (width, height) of the decoded image.
        """
        if self._closed:
            raise RuntimeError('Decoder is closed')

        if u_pitch is None:
            u_pitch = y_pitch
        if v_pitch is None:
            v_pitch = y_pitch

        width, height, _, _ = self.get_image_info(jpeg_data)

        output = nvjpegImage_t()
        output.channel[0] = y_ptr
        output.channel[1] = u_ptr
        output.channel[2] = v_ptr
        output.pitch[0] = y_pitch
        output.pitch[1] = u_pitch
        output.pitch[2] = v_pitch

        status = _lib.nvjpegDecode(
            self._handle,
            self._state,
            _get_data_ptr(jpeg_data),
            c_size_t(len(jpeg_data)),
            NVJPEG_OUTPUT_YUV,
            byref(output),
            stream,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to decode JPEG to YUV'))

        return width, height

    def close(self):
        """Release resources."""
        if self._closed:
            return
        self._closed = True

        self._cleanup_partial()

        if self._cuda_ctx:
            driver.cuCtxDestroy(self._cuda_ctx)

    def __del__(self):
        if not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class NvjpegPhasedDecoder:
    """GPU JPEG decoder using nvJPEG phased API for async pipeline.

    The phased API splits decoding into stages that can overlap:
    1. parse() - Parse JPEG headers (CPU, fast)
    2. decode_host() - Huffman decode (CPU, slow)
    3. decode_transfer() - Transfer to GPU (async)
    4. decode_device() - IDCT/color on GPU (async)

    This allows CPU work on frame N+1 while GPU processes frame N.

    Example:
        >>> decoder = NvjpegPhasedDecoder()
        >>> decoder.parse(jpeg_bytes)
        >>> decoder.decode_host()
        >>> decoder.decode_transfer(stream)  # async
        >>> decoder.decode_device(output, stream)  # async
        >>> # GPU is now working, CPU can parse next frame
    """

    def __init__(self, gpu: int | None = 0):
        """Initialize the phased nvJPEG decoder.

        Args:
            gpu: CUDA device ordinal to create a context on. If None, the caller
                is responsible for providing an existing CUDA context.
        """
        if _lib is None:
            raise ImportError(
                'nvJPEG library not available. '
                'Ensure libnvjpeg.so is installed (part of CUDA toolkit).'
            )
        self._handle = nvjpegHandle_t()
        self._decoder = nvjpegJpegDecoder_t()
        self._state = nvjpegJpegState_t()
        self._params = nvjpegDecodeParams_t()
        self._jpeg_stream = nvjpegJpegStream_t()
        self._pinned_buffer = nvjpegBufferPinned_t()
        self._device_buffer = nvjpegBufferDevice_t()
        self._closed = False
        self._cuda_ctx = None
        self._owns_cuda_ctx = gpu is not None

        # Initialize CUDA if needed
        if gpu is not None:
            driver.cuInit(0)
            err, device = driver.cuDeviceGet(gpu)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to get CUDA device {gpu}: {err}')
            err, self._cuda_ctx = cuCtxCreate(0, device)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to create CUDA context: {err}')

        # Initialize nvJPEG handle and decoder
        self._check(
            _lib.nvjpegCreateSimple(byref(self._handle)),
            'Failed to create nvJPEG handle',
        )
        self._check(
            _lib.nvjpegDecoderCreate(self._handle, NVJPEG_BACKEND_HYBRID, byref(self._decoder)),
            'Failed to create decoder',
        )
        self._check(
            _lib.nvjpegDecoderStateCreate(self._handle, self._decoder, byref(self._state)),
            'Failed to create decoder state',
        )

        # Create internal buffers for phased decoding
        self._check(
            _lib.nvjpegBufferPinnedCreate(self._handle, None, byref(self._pinned_buffer)),
            'Failed to create pinned buffer',
        )
        self._check(
            _lib.nvjpegBufferDeviceCreate(self._handle, None, byref(self._device_buffer)),
            'Failed to create device buffer',
        )

        # Attach buffers to state (required for phased decode)
        self._check(
            _lib.nvjpegStateAttachPinnedBuffer(self._state, self._pinned_buffer),
            'Failed to attach pinned buffer',
        )
        self._check(
            _lib.nvjpegStateAttachDeviceBuffer(self._state, self._device_buffer),
            'Failed to attach device buffer',
        )

        # Create decode params and set output format
        self._check(
            _lib.nvjpegDecodeParamsCreate(self._handle, byref(self._params)),
            'Failed to create decode params',
        )
        self._check(
            _lib.nvjpegDecodeParamsSetOutputFormat(self._params, NVJPEG_OUTPUT_YUV),
            'Failed to set output format',
        )

        # Create JPEG stream for parsing
        self._check(
            _lib.nvjpegJpegStreamCreate(self._handle, byref(self._jpeg_stream)),
            'Failed to create JPEG stream',
        )

        # Cache for parsed image info
        self._parsed_width = 0
        self._parsed_height = 0
        self._parsed_subsampling = -1

    def _cleanup_partial(self):
        """Clean up partially initialized resources."""
        if self._jpeg_stream:
            _lib.nvjpegJpegStreamDestroy(self._jpeg_stream)
            self._jpeg_stream = nvjpegJpegStream_t()
        if self._params:
            _lib.nvjpegDecodeParamsDestroy(self._params)
            self._params = nvjpegDecodeParams_t()
        if self._device_buffer:
            _lib.nvjpegBufferDeviceDestroy(self._device_buffer)
            self._device_buffer = nvjpegBufferDevice_t()
        if self._pinned_buffer:
            _lib.nvjpegBufferPinnedDestroy(self._pinned_buffer)
            self._pinned_buffer = nvjpegBufferPinned_t()
        if self._state:
            _lib.nvjpegJpegStateDestroy(self._state)
            self._state = nvjpegJpegState_t()
        if self._decoder:
            _lib.nvjpegDecoderDestroy(self._decoder)
            self._decoder = nvjpegJpegDecoder_t()
        if self._handle:
            _lib.nvjpegDestroy(self._handle)
            self._handle = nvjpegHandle_t()

    def _check(self, status: int, msg: str) -> None:
        """Check nvJPEG status, cleanup and raise on error."""
        if status != NVJPEG_STATUS_SUCCESS:
            self._cleanup_partial()
            if self._owns_cuda_ctx and self._cuda_ctx:
                driver.cuCtxDestroy(self._cuda_ctx)
                self._cuda_ctx = None
            raise RuntimeError(nvjpeg_status_message(status, msg))

    def parse(self, jpeg_data: JpegData) -> tuple[int, int, int]:
        """Parse JPEG headers and prepare for decoding.

        This is fast CPU work - just reads headers.

        Args:
            jpeg_data: JPEG file data.

        Returns:
            Tuple of (width, height, subsampling).
        """
        if self._closed:
            raise RuntimeError('Decoder is closed')

        status = _lib.nvjpegJpegStreamParse(
            self._handle,
            _get_data_ptr(jpeg_data),
            c_size_t(len(jpeg_data)),
            1,  # save_metadata (required for phased decoding)
            1,  # save_stream (required for phased decoding)
            self._jpeg_stream,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to parse JPEG'))

        # Get dimensions from parsed stream
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        status = _lib.nvjpegJpegStreamGetFrameDimensions(
            self._jpeg_stream, byref(width), byref(height)
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to get frame dimensions'))

        subsampling = c_int()
        status = _lib.nvjpegJpegStreamGetChromaSubsampling(
            self._jpeg_stream, byref(subsampling)
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to get chroma subsampling'))

        self._parsed_width = width.value
        self._parsed_height = height.value
        self._parsed_subsampling = subsampling.value

        return width.value, height.value, subsampling.value

    def decode_host(self) -> None:
        """Perform CPU-side decode (Huffman decoding).

        This is the slow CPU work. Call after parse().
        """
        if self._closed:
            raise RuntimeError('Decoder is closed')

        status = _lib.nvjpegDecodeJpegHost(
            self._handle,
            self._decoder,
            self._state,
            self._params,
            self._jpeg_stream,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to decode JPEG on host'))

    def decode_transfer(self, stream: int | None = None) -> None:
        """Transfer decoded data to GPU (async).

        This queues work on the CUDA stream and returns immediately.

        Args:
            stream: CUDA stream handle (None for default stream).
        """
        if self._closed:
            raise RuntimeError('Decoder is closed')

        status = _lib.nvjpegDecodeJpegTransferToDevice(
            self._handle,
            self._decoder,
            self._state,
            self._jpeg_stream,
            ctypes.c_void_p(stream) if stream else None,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to transfer to device'))

    def decode_device(
        self,
        y_ptr: int,
        u_ptr: int,
        v_ptr: int,
        y_pitch: int,
        u_pitch: int,
        v_pitch: int,
        stream: int | None = None,
    ) -> None:
        """Perform GPU-side decode (IDCT, color conversion) - async.

        This queues work on the CUDA stream and returns immediately.

        Args:
            y_ptr, u_ptr, v_ptr: CUDA device pointers for Y, U, V planes.
            y_pitch, u_pitch, v_pitch: Pitch for each plane.
            stream: CUDA stream handle.
        """
        if self._closed:
            raise RuntimeError('Decoder is closed')

        output = nvjpegImage_t()
        output.channel[0] = y_ptr
        output.channel[1] = u_ptr
        output.channel[2] = v_ptr
        output.pitch[0] = y_pitch
        output.pitch[1] = u_pitch
        output.pitch[2] = v_pitch

        status = _lib.nvjpegDecodeJpegDevice(
            self._handle,
            self._decoder,
            self._state,
            byref(output),
            ctypes.c_void_p(stream) if stream else None,
        )
        if status != NVJPEG_STATUS_SUCCESS:
            raise RuntimeError(nvjpeg_status_message(status, 'Failed to decode JPEG on device'))

    def decode_phased_into(
        self,
        jpeg_data: JpegData,
        y_ptr: int,
        u_ptr: int,
        v_ptr: int,
        y_pitch: int,
        u_pitch: int,
        v_pitch: int,
        stream: int | None = None,
    ) -> tuple[int, int, int]:
        """Full phased decode in one call - for simpler usage.

        Equivalent to: parse() + decode_host() + decode_transfer() + decode_device()
        The GPU work (transfer + device) is async on the given stream.

        Returns:
            Tuple of (width, height, subsampling).
        """
        width, height, subsampling = self.parse(jpeg_data)
        self.decode_host()
        self.decode_transfer(stream)
        self.decode_device(y_ptr, u_ptr, v_ptr, y_pitch, u_pitch, v_pitch, stream)
        return width, height, subsampling

    @property
    def parsed_width(self) -> int:
        return self._parsed_width

    @property
    def parsed_height(self) -> int:
        return self._parsed_height

    @property
    def parsed_subsampling(self) -> int:
        return self._parsed_subsampling

    def close(self):
        """Release resources."""
        if self._closed:
            return
        self._closed = True

        self._cleanup_partial()

        if self._owns_cuda_ctx and self._cuda_ctx:
            driver.cuCtxDestroy(self._cuda_ctx)

    def __del__(self):
        if not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()