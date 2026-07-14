"""nvJPEG GPU JPEG decoder."""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
from ctypes import POINTER, byref, c_int, c_size_t, c_ubyte
from typing import Union

import numpy as np
from cuda.bindings import driver

from .._cuda_compat import cuda_ctx_pushed, retain_primary_context
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

# Number of internal slots (JPEG stream + pinned buffer pairs) in the phased
# decoder. Two slots let the CPU stages of frame N+1 run while frame N's
# transfer is still in flight.
_NUM_SLOTS = 2


def _get_data_ptr_and_size(data: JpegData) -> tuple[BytePointer, int]:
    """Get a ctypes pointer to the JPEG bytes and their byte length, without copying.

    Accepts bytes-like objects or uint8 C-contiguous numpy arrays. The caller
    must keep ``data`` alive for the duration of the native call.
    """
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            raise ValueError(f'JPEG data array must have dtype uint8, got {data.dtype}')
        if not data.flags.c_contiguous:
            raise ValueError('JPEG data array must be C-contiguous')
        arr = data
    else:
        # Bytes-like objects are wrapped via the buffer protocol (zero-copy).
        arr = np.frombuffer(data, dtype=np.uint8)
    return arr.ctypes.data_as(BytePointer), arr.nbytes


class NvjpegDecoder:
    """GPU JPEG decoder using nvJPEG.

    Decodes JPEG data directly into user-provided CUDA device buffers (zero-copy).

    The decoder retains the primary CUDA context of the given device and makes
    it current only for the duration of its own calls — the caller's current
    context is never disturbed. Callers doing their own CUDA work (allocating
    the output buffers, synchronizing) must hold a context themselves, e.g.
    via ``cuDevicePrimaryCtxRetain`` on the same device or through torch.

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
        self._closed = False

        self._cuda_device, self._cuda_ctx = retain_primary_context(gpu)
        try:
            with cuda_ctx_pushed(self._cuda_ctx):
                self._check(
                    _lib.nvjpegCreateSimple(byref(self._handle)),
                    'Failed to create nvJPEG handle',
                )
                self._check(
                    _lib.nvjpegJpegStateCreate(self._handle, byref(self._state)),
                    'Failed to create JPEG state',
                )
        except Exception:
            driver.cuDevicePrimaryCtxRelease(self._cuda_device)
            self._cuda_ctx = None
            self._cuda_device = None
            raise

    def _cleanup_partial(self) -> None:
        """Clean up partially initialized nvJPEG resources."""
        if self._state:
            _lib.nvjpegJpegStateDestroy(self._state)
            self._state = nvjpegJpegState_t()
        if self._handle:
            _lib.nvjpegDestroy(self._handle)
            self._handle = nvjpegHandle_t()

    def _check(self, status: int, msg: str) -> None:
        """Check nvJPEG status, cleanup nvJPEG state and raise on error."""
        if status != NVJPEG_STATUS_SUCCESS:
            self._cleanup_partial()
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

        data_ptr, data_size = _get_data_ptr_and_size(jpeg_data)
        with cuda_ctx_pushed(self._cuda_ctx):
            status = _lib.nvjpegGetImageInfo(
                self._handle,
                data_ptr,
                c_size_t(data_size),
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

        data_ptr, data_size = _get_data_ptr_and_size(jpeg_data)
        with cuda_ctx_pushed(self._cuda_ctx):
            status = _lib.nvjpegDecode(
                self._handle,
                self._state,
                data_ptr,
                c_size_t(data_size),
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

        if self._cuda_ctx is not None:
            with cuda_ctx_pushed(self._cuda_ctx):
                self._cleanup_partial()
            driver.cuDevicePrimaryCtxRelease(self._cuda_device)
            self._cuda_ctx = None
            self._cuda_device = None
        else:
            self._cleanup_partial()

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

    JPEG streams and pinned buffers are double-buffered internally, and
    decode_transfer() synchronizes the stream before submitting new work, so
    the CPU stages of the next frame may safely run while the previous
    frame's GPU work is still in flight:

        >>> decoder = NvjpegPhasedDecoder()
        >>> for jpeg_bytes in jpegs:
        ...     decoder.parse(jpeg_bytes)
        ...     decoder.decode_host()  # overlaps previous frame's GPU work
        ...     decoder.decode_transfer(stream)  # async
        ...     decoder.decode_device(y, u, v, y_pitch, u_pitch, v_pitch, stream)  # async

    All frames must be submitted on the same CUDA stream.

    When ``gpu`` is given, the decoder retains that device's primary CUDA
    context and makes it current only for the duration of its own calls — the
    caller's current context is never disturbed. Callers doing their own CUDA
    work (creating the stream, allocating output buffers) must hold a context
    themselves, e.g. via ``cuDevicePrimaryCtxRetain`` on the same device.
    """

    def __init__(self, gpu: int | None = 0):
        """Initialize the phased nvJPEG decoder.

        Args:
            gpu: CUDA device ordinal whose primary context to retain. If None,
                the caller is responsible for providing a current CUDA context
                around every call.
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
        self._jpeg_streams = [nvjpegJpegStream_t() for _ in range(_NUM_SLOTS)]
        self._pinned_buffers = [nvjpegBufferPinned_t() for _ in range(_NUM_SLOTS)]
        self._device_buffer = nvjpegBufferDevice_t()
        # parse() rotates first, so the first frame lands in slot 0.
        self._slot = _NUM_SLOTS - 1
        self._closed = False
        self._cuda_ctx = None
        self._cuda_device = None
        self._owns_cuda_ctx = gpu is not None

        if gpu is not None:
            self._cuda_device, self._cuda_ctx = retain_primary_context(gpu)

        try:
            with self._ctx_guard():
                self._init_nvjpeg()
        except Exception:
            if self._owns_cuda_ctx and self._cuda_device is not None:
                driver.cuDevicePrimaryCtxRelease(self._cuda_device)
                self._cuda_ctx = None
                self._cuda_device = None
            raise

        # Cache for parsed image info
        self._parsed_width = 0
        self._parsed_height = 0
        self._parsed_subsampling = -1

    def _ctx_guard(self):
        """Push the owned context for the duration of a call, if any."""
        if self._owns_cuda_ctx and self._cuda_ctx is not None:
            return cuda_ctx_pushed(self._cuda_ctx)
        return nullcontext()

    def _init_nvjpeg(self) -> None:
        """Create all nvJPEG objects (under the owned context, if any)."""
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

        # Create internal buffers for phased decoding. Pinned buffers and JPEG
        # streams are per-slot so that the CPU stages of the next frame never
        # touch what the previous frame's in-flight transfer still reads.
        for pinned_buffer in self._pinned_buffers:
            self._check(
                _lib.nvjpegBufferPinnedCreate(self._handle, None, byref(pinned_buffer)),
                'Failed to create pinned buffer',
            )
        self._check(
            _lib.nvjpegBufferDeviceCreate(self._handle, None, byref(self._device_buffer)),
            'Failed to create device buffer',
        )

        # The device buffer is attached once and shared across slots; the
        # active slot's pinned buffer is (re)attached by parse().
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

        # Create JPEG streams for parsing (one per slot)
        for jpeg_stream in self._jpeg_streams:
            self._check(
                _lib.nvjpegJpegStreamCreate(self._handle, byref(jpeg_stream)),
                'Failed to create JPEG stream',
            )

    def _cleanup_partial(self):
        """Clean up partially initialized resources."""
        for jpeg_stream in self._jpeg_streams:
            if jpeg_stream:
                _lib.nvjpegJpegStreamDestroy(jpeg_stream)
        self._jpeg_streams = []
        if self._params:
            _lib.nvjpegDecodeParamsDestroy(self._params)
            self._params = nvjpegDecodeParams_t()
        if self._device_buffer:
            _lib.nvjpegBufferDeviceDestroy(self._device_buffer)
            self._device_buffer = nvjpegBufferDevice_t()
        for pinned_buffer in self._pinned_buffers:
            if pinned_buffer:
                _lib.nvjpegBufferPinnedDestroy(pinned_buffer)
        self._pinned_buffers = []
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
        """Check nvJPEG status, cleanup nvJPEG state and raise on error."""
        if status != NVJPEG_STATUS_SUCCESS:
            self._cleanup_partial()
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

        data_ptr, data_size = _get_data_ptr_and_size(jpeg_data)

        with self._ctx_guard():
            # Rotate to the next slot and attach its pinned buffer, so this
            # frame's CPU stages never touch the buffers that the previous
            # frame's in-flight transfer still reads.
            self._slot = (self._slot + 1) % _NUM_SLOTS
            jpeg_stream = self._jpeg_streams[self._slot]
            status = _lib.nvjpegStateAttachPinnedBuffer(
                self._state, self._pinned_buffers[self._slot]
            )
            if status != NVJPEG_STATUS_SUCCESS:
                raise RuntimeError(
                    nvjpeg_status_message(status, 'Failed to attach pinned buffer')
                )

            status = _lib.nvjpegJpegStreamParse(
                self._handle,
                data_ptr,
                c_size_t(data_size),
                1,  # save_metadata (required for phased decoding)
                1,  # save_stream (required for phased decoding)
                jpeg_stream,
            )
            if status != NVJPEG_STATUS_SUCCESS:
                raise RuntimeError(nvjpeg_status_message(status, 'Failed to parse JPEG'))

            # Get dimensions from parsed stream
            width = ctypes.c_uint()
            height = ctypes.c_uint()
            status = _lib.nvjpegJpegStreamGetFrameDimensions(
                jpeg_stream, byref(width), byref(height)
            )
            if status != NVJPEG_STATUS_SUCCESS:
                raise RuntimeError(
                    nvjpeg_status_message(status, 'Failed to get frame dimensions')
                )

            subsampling = c_int()
            status = _lib.nvjpegJpegStreamGetChromaSubsampling(jpeg_stream, byref(subsampling))
            if status != NVJPEG_STATUS_SUCCESS:
                raise RuntimeError(
                    nvjpeg_status_message(status, 'Failed to get chroma subsampling')
                )

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

        with self._ctx_guard():
            status = _lib.nvjpegDecodeJpegHost(
                self._handle,
                self._decoder,
                self._state,
                self._params,
                self._jpeg_streams[self._slot],
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

        with self._ctx_guard():
            # The decoder state's device-side working memory is shared across
            # slots, so the previous frame's GPU work must finish before new
            # work is submitted (this also frees the slot being reused two
            # frames later). Requires all frames to use the same stream.
            (err,) = driver.cuStreamSynchronize(stream if stream is not None else 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to synchronize stream before transfer: {err}')

            status = _lib.nvjpegDecodeJpegTransferToDevice(
                self._handle,
                self._decoder,
                self._state,
                self._jpeg_streams[self._slot],
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

        with self._ctx_guard():
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

        with self._ctx_guard():
            self._cleanup_partial()

        if self._owns_cuda_ctx and self._cuda_device is not None:
            driver.cuDevicePrimaryCtxRelease(self._cuda_device)
            self._cuda_ctx = None
            self._cuda_device = None

    def __del__(self):
        if not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
