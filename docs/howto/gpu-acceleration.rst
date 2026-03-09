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
   * - ``JpegVideoWriterCUDA``
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

**Supported codecs:** H.264, HEVC, MPEG-1/2/4, AV1, VP8, VP9, VC-1, MJPEG.
Unsupported codecs or container formats (e.g. FLV) raise
:class:`~framepump.FramePumpError` with a clear message — FramePump probes the
file before opening with hardware acceleration to avoid FFmpeg crashes.

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
        # tensor is (H, W, 3) uint8 on CUDA

``VideoFramesCuda`` supports the same slicing and indexing as ``VideoFrames``:

.. code-block:: python

    frames = VideoFramesCuda('input.mp4')
    subset = frames[::2][:100]
    single = frames[42]

For 10-bit video sources, use ``dtype=np.uint16`` to preserve the full
precision through an NPP (NVIDIA Performance Primitives) conversion pipeline.

**Pipeline:**

::

    Video file → PyNvDemuxer → NVDEC (GPU) → [NPP color conversion] → GPU buffer (DLPack)


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

**Two encoding paths (auto-selected):**

- **GLX path** (X11 display available): GL texture → NVENC directly.
  Selected when the ``DISPLAY`` environment variable is set.

- **CUDA path** (headless/EGL): GL texture → CUarray via CUDA-GL interop →
  NVENC. Selected when ``DISPLAY`` is not set. Requires
  ``pip install framepump[nvenc-cuda]``.

**Pipeline:**

::

    GL texture → [CUDA-GL interop if headless] → NVENC (GPU) → H.264 NALs → PyAV muxer → MP4

See :doc:`/explanation/nvenc-zero-copy` for a detailed explanation.


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


JpegVideoWriterCUDA
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

    from framepump import JpegVideoWriterCUDA

    with JpegVideoWriterCUDA('output.mp4', fps=30, gpu=0) as writer:
        for jpeg_bytes in camera_stream:
            writer.append_data(jpeg_bytes)  # bytes object

The decoder uses a phased pipeline for throughput: while frame N is being
encoded by NVENC, frame N+1's JPEG is being decoded by nvJPEG. Two GPU
buffers alternate (ping-pong) so decode and encode overlap.

Supports both 4:2:0 and 4:4:4 chroma subsampling — auto-detected from the
first JPEG frame, or set explicitly:

.. code-block:: python

    writer = JpegVideoWriterCUDA('output.mp4', fps=30, gpu=0, chroma='444')

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

- **GPU decoding** (``gpu=True``): NVIDIA GPU, CUDA-enabled FFmpeg build
  (PyAV must be linked against it)
- **VideoFramesCuda**: ``pip install framepump[cuda]`` (requires
  ``PyNvVideoCodec``)
- **GLVideoWriter (GLX)**: NVIDIA GPU, X11 display, ``libnvidia-encode.so``
- **GLVideoWriter (headless)**: ``pip install framepump[nvenc-cuda]``
  (``cuda-python``)
- **JpegVideoWriterCUDA**: ``pip install framepump[cuda]`` (``cuda-python``,
  nvJPEG)
