GPU Acceleration
================

FramePump supports GPU-accelerated video decoding and encoding through several
classes, each targeting a different use case.


Choosing the Right Class
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Class
     - Direction
     - Output
     - Use when
   * - ``VideoFrames(gpu=True)``
     - Decode
     - numpy arrays (CPU)
     - You want faster decoding with a familiar numpy interface
   * - ``VideoFramesCuda``
     - Decode
     - GPU tensors (DLPack)
     - You need frames to stay on GPU for further CUDA processing
   * - ``GLVideoWriter``
     - Encode
     - MP4 file
     - You're encoding OpenGL textures to video (zero CPU transfer)
   * - ``NvJpegVideoWriter``
     - Encode
     - MP4 file
     - You have JPEG byte streams and want fully GPU-resident encoding

All GPU features require an NVIDIA GPU with appropriate driver support.


GPU-Accelerated Decoding
-------------------------

VideoFrames with gpu=True
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use GPU decoding. Frames are decoded on the GPU using
FFmpeg's CUDA hardware acceleration, then transferred to CPU as numpy arrays:

.. code-block:: python

    from framepump import VideoFrames

    frames = VideoFrames('input.mp4', gpu=True)
    for frame in frames:
        # frame is a regular numpy array, same as CPU path
        print(frame.shape, frame.dtype)

To select a specific GPU:

.. code-block:: python

    frames = VideoFrames('input.mp4', gpu=0)  # first GPU
    frames = VideoFrames('input.mp4', gpu=1)  # second GPU

**Supported codecs** are whatever the GPU's NVDEC engine handles (typically
H.264, HEVC, MPEG-1/2/4, AV1, VP8, VP9, VC-1, MJPEG — depending on GPU
generation). FFmpeg reports compatibility when the file is opened: an
unsupported codec raises :class:`~framepump.FramePumpError` with a clear
message suggesting ``gpu=False``. There is deliberately no silent fallback:
``gpu=True`` either really decodes on the GPU or fails loudly.

The decoded output is **bit-identical to CPU decoding** — you can switch
``gpu`` on and off without affecting results.

**Pipeline:**

::

    Video file → FFmpeg demuxer → NVDEC (GPU) → GPU→CPU transfer → numpy array


VideoFramesCuda
~~~~~~~~~~~~~~~

For GPU-resident processing where you don't want to pay the cost of
transferring frames to CPU. Decoded frames stay in GPU memory and are
exported via DLPack for zero-copy access from PyTorch, CuPy, etc.

.. code-block:: python

    from framepump import VideoFramesCuda

    frames = VideoFramesCuda('input.mp4')
    for decoded_frame in frames:
        # decoded_frame supports __dlpack__() for zero-copy export
        tensor = torch.from_dlpack(decoded_frame)  # no copy, shares GPU memory
        # tensor is (H, W, 3) uint8 on CUDA — valid until the next iteration step!

.. warning::

    Iteration yields **zero-copy views of reusable GPU memory**: a frame (or
    a tensor made from it) is only valid until the next iteration step, after
    which the underlying buffer is reused and its contents silently change.
    Process each frame within its step, or ``.clone()`` the tensor before
    keeping it:

    .. code-block:: python

        kept = [torch.from_dlpack(f).clone() for f in frames]   # safe
        kept = [torch.from_dlpack(f) for f in frames]           # WRONG: all
                                                                # end up with
                                                                # recycled data

    To collect many frames, prefer the batch gather below — it produces one
    independently owned buffer with no cloning needed. Single indexed frames
    (``frames[42]``) are also independently owned and safe to keep.

``VideoFramesCuda`` supports slicing, indexing, and batch gathering:

.. code-block:: python

    frames = VideoFramesCuda('input.mp4')
    subset = frames[::2][:100]
    single = frames[42]

    # numpy-style index lists produce ONE stacked (n, H, W, 3) GPU buffer:
    # a ready batch tensor straight from NVDEC, no further copies
    batch = torch.from_dlpack(frames[[10, 50, 51, 300]])

    # lazy counterpart, yielding frames in the given order
    for f in frames.frames_at(kept_indices):
        ...

GPU-side processing options mirror the CPU class: ``resized()`` (NPP resize,
with an optional ``gamma_correct=True`` mode that resamples in linear light
using the exact sRGB transfer), ``repeat_each_frame()``,
``constant_framerate`` (the same ffmpeg-parity source map as the CPU class,
so both select identical source frames), float16/float32 outputs scaled to
[0, 1], and file-like sources (BytesIO, archive members — GPU decoding of
video that never touches the filesystem). Iteration and DLPack export work
from any thread, including prefetch threads in processes where torch owns
the CUDA state.

For 10-bit video sources, use ``dtype=np.uint16`` to preserve the full
precision through an NPP (NVIDIA Performance Primitives) conversion
pipeline; the conversion matrix and range follow the stream's colorspace
flags (BT.601/709/2020 and friends).

Unlike ``VideoFrames(gpu=True)``, the output of ``VideoFramesCuda`` is **not
bit-identical** to CPU decoding: its YUV→RGB conversion runs in CUDA kernels
rather than FFmpeg's swscale, so pixel values typically differ by a few
counts. Don't mix the two classes when exact reproducibility matters.

**Pipeline:**

::

    Video file → PyAV demuxer → NVDEC (GPU) → [NPP color conversion] → GPU buffer (DLPack)


CameraFrames — live cameras
~~~~~~~~~~~~~~~~~~~~~~~~~~~

USB cameras deliver MJPEG at their real resolutions; ``CameraFrames``
decodes that stream on the GPU (NVDEC's JPEG engine) and always hands you
the **latest** captured frame — a consumer slower than the camera skips
stale frames instead of processing a growing backlog, keeping latency at
one frame interval (a naive queueing loop reaches seconds of staleness
within moments):

.. code-block:: python

    from framepump import CameraFrames

    with CameraFrames('/dev/video0', shape=(720, 1280), fps=30) as cam:
        for frame in cam:
            tensor = torch.from_dlpack(frame)  # (H, W, 3) uint8 CUDA, zero-copy
            ...  # valid until the next iteration step; .clone() to keep
            if done:
                break

Live semantics: iteration only (no ``len()``, seeking or slicing);
``cam.last_capture_time`` carries the kernel capture timestamp of the
delivered frame (monotonic clock) so staleness is always measurable.
Linux/V4L2, MJPEG cameras (essentially all UVC devices).

For a model that only reaches real-time throughput when batching,
``cam.batched(n)`` yields adaptive batches instead of single frames: each
step delivers up to ``n`` frames as one stacked ``(k, H, W, 3)`` GPU
buffer, chosen by dividing the time since the previous delivery into
``n`` equal steps and taking the nearest retained frame to each — so a
slow consumer receives frames spread evenly across the whole interval it
missed (always ending with the newest), never a burst of near-identical
latest frames, and never the same frame twice. Selections landing on the
same frame collapse, so a consumer that keeps up simply gets ``k = 1``.
The camera retains up to ``history`` undelivered frames (default two
seconds' worth) to make this possible; ``cam.last_capture_times`` holds
the k capture timestamps:

.. code-block:: python

    with CameraFrames('/dev/video0', shape=(720, 1280), fps=30) as cam:
        for batch in cam.batched(4):
            tensors = torch.from_dlpack(batch)  # (k, H, W, 3), 1 <= k <= 4
            ...

Mind that shape-specializing runtimes (TorchScript JIT, cudnn benchmark
mode) recompile for each distinct batch size — warm up every size from 1
to ``n`` before going live, or the first occurrence of each size stalls.

CudaToGLUploader
~~~~~~~~~~~~~~~~

Transfers GPU-resident frames to OpenGL textures via CUDA-GL interop,
without going through CPU:

.. code-block:: python

    from framepump import VideoFramesCuda, CudaToGLUploader

    frames = VideoFramesCuda('input.mp4')
    # gl_texture is a GL texture ID (e.g. from moderngl)
    uploader = CudaToGLUploader(gl_texture_id, width, height)

    decoded = frames[0]
    tensor = torch.from_dlpack(decoded)
    uploader.upload(tensor)  # GPU→GPU DMA, no CPU involved

This is useful for rendering decoded video frames in an OpenGL application
without any CPU round-trip.


GPU-Accelerated Encoding
-------------------------

GLVideoWriter
~~~~~~~~~~~~~

Encodes OpenGL textures directly to H.264 video using NVIDIA's NVENC hardware
encoder. Pixel data never leaves the GPU.

.. code-block:: python

    from framepump import GLVideoWriter

    with GLVideoWriter('output.mp4', fps=30) as writer:
        for _ in render_loop:
            render_to_texture(texture)
            ctx.finish()  # ensure GPU rendering is complete
            writer.append_data(texture)  # encode directly from GPU

``GLVideoWriter`` runs synchronously (no background thread) because the
OpenGL context must be current on the calling thread.

**Two encoding paths**, selectable via the ``backend`` parameter:

- **GLX path** (``backend='glx'``): GL texture → NVENC directly. Requires an
  X11-backed (GLX) OpenGL context on the NVIDIA GPU.

- **CUDA path** (``backend='cuda'``): GL texture → CUarray via CUDA-GL
  interop → NVENC. Works with EGL contexts and headless setups. Requires
  ``pip install framepump[nvenc-cuda]``.

The default ``backend='auto'`` picks the CUDA path when the ``DISPLAY``
environment variable is unset and the GLX path otherwise. That heuristic can
guess wrong — most commonly for a standalone EGL context created while
``DISPLAY`` is still set — so pass ``backend='cuda'`` explicitly when you
know your context is EGL:

.. code-block:: python

    with GLVideoWriter('output.mp4', fps=30, backend='cuda') as writer:
        ...

**Pipeline:**

::

    GL texture → [CUDA-GL interop if CUDA path] → NVENC (GPU) → H.264 NALs → PyAV muxer → MP4

See :doc:`/explanation/nvenc-zero-copy` for a detailed explanation.

Hybrid AMD+NVIDIA machines
~~~~~~~~~~~~~~~~~~~~~~~~~~

On machines where an AMD or Intel GPU drives the display and the NVIDIA card
is a compute device (common laptop and SFF-workstation setup), GLX contexts
land on the display GPU by default, and ``GLVideoWriter`` fails with
*"Current OpenGL context is on a non-NVIDIA GPU"*. Two remedies:

- **Windowed (GLX):** route OpenGL to the NVIDIA GPU via PRIME render
  offload before creating the GL context::

      export __NV_PRIME_RENDER_OFFLOAD=1
      export __GLX_VENDOR_LIBRARY_NAME=nvidia

- **Headless (EGL):** create an EGL context on the NVIDIA device and use
  ``backend='cuda'`` (or unset ``DISPLAY`` and rely on ``backend='auto'``).
  For batch/offline rendering this is the recommended setup.


VideoWriter with GPU Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regular :class:`~framepump.VideoWriter` also supports GPU encoding by
passing ``gpu=True``. Unlike ``GLVideoWriter``, the input is numpy arrays
(CPU), and encoding uses PyAV's ``h264_nvenc`` codec:

.. code-block:: python

    from framepump import VideoWriter

    with VideoWriter('output.mp4', fps=30, gpu=True) as writer:
        for frame in frames:
            writer.append_data(frame)  # numpy array → GPU encode

This is simpler than ``GLVideoWriter`` but involves a CPU→GPU transfer for
each frame. It's useful when your frames are already in numpy arrays and you
want faster encoding than libx264.


NvJpegVideoWriter
~~~~~~~~~~~~~~~~~~~

A specialized writer for the case when the input data consists of frames in JPEG
format. The entire path from JPEG bytes to H.264 video stays on the GPU (decoding via NVJPEG and encoding via NVENC).

Common use cases:

- **Camera / network streams** that deliver JPEG frames.
- **Dataset conversion**: many datasets distribute video as directories of JPEG
  frames. Converting them to video files saves disk space, and
  :class:`~framepump.VideoFrames` provides fast, frame-accurate random access,
  making video files a practical replacement for frame directories.

.. code-block:: python

    from framepump import NvJpegVideoWriter

    with NvJpegVideoWriter('output.mp4', fps=30, gpu=0) as writer:
        for jpeg_bytes in camera_stream:
            writer.append_data(jpeg_bytes)  # bytes object

The decoder uses a phased pipeline for throughput: while frame N is being
encoded by NVENC, frame N+1's JPEG is being decoded by nvJPEG. Two GPU
buffers alternate (ping-pong) so decode and encode overlap.

Supports both 4:2:0 and 4:4:4 chroma subsampling — auto-detected from the
first JPEG frame, or set explicitly:

.. code-block:: python

    writer = NvJpegVideoWriter('output.mp4', fps=30, gpu=0, chroma='444')

**Pipeline:**

::

    JPEG bytes → nvJPEG decode (GPU) → YUV buffer (GPU) → NVENC encode (GPU) → H.264 → PyAV mux → MP4


Encoder Configuration
---------------------

All writers accept an :class:`~framepump.EncoderConfig` for fine-grained
control:

.. code-block:: python

    from framepump import EncoderConfig, VideoWriter

    config = EncoderConfig(
        crf=18,       # quality (0–51, lower = better, default 15)
        gop=120,      # keyframe interval (default 250)
        bframes=2,    # B-frames for compression (default 2)
        codec='h264', # 'h264' or 'hevc'
        preset='p5',  # NVENC: 'p1'–'p7'; libx264: 'ultrafast'–'veryslow'
    )

    with VideoWriter('output.mp4', fps=30, gpu=True, encoder_config=config) as w:
        ...

Preset names are auto-translated between NVENC and libx264 — you can use
either naming convention regardless of the target encoder.


Requirements
------------

- **GPU decoding** (``gpu=True``): NVIDIA GPU with driver installed; the
  standard PyAV wheels include NVDEC support
- **VideoFramesCuda**: ``pip install framepump[cuda]`` (requires
  ``PyNvVideoCodec``)
- **GLVideoWriter (GLX)**: NVIDIA GPU, X11 display, ``libnvidia-encode.so``
- **GLVideoWriter (headless)**: ``pip install framepump[nvenc-cuda]``
  (``cuda-python``)
- **NvJpegVideoWriter**: ``pip install framepump[cuda]`` (``cuda-python``,
  nvJPEG)
