Zero-Copy OpenGL to Video Encoding with NVENC
==============================================

Rendering video frames on the GPU is fast. And newer NVIDIA GPUs have
dedicated hardware for video encoding (NVENC) that can compress frames without
using the CPU. But getting pixel data from OpenGL to NVENC efficiently can be
tricky. This document explains how framepump achieves zero-copy
transfer from OpenGL textures to NVENC.

The naive approach to rendering to a video file looks like this:


1. GPU renders a frame to an OpenGL texture
2. CPU reads the texture back to system memory
3. Software encoder compresses the frame
4. Compressed data is written to disk

Both the readback (step 2) and the software encoding (step 3) are slow. Readback stalls the GPU
waiting for the CPU to fetch pixels over the PCIe bus. Software encoding uses CPU cycles
that could be spent elsewhere and is much slower than hardware encoding.

The goal is therefore to keep the data on the GPU throughout rendering and encoding.


The NVENC Hardware Encoder
---------------------------

NVIDIA GPUs since Kepler (2012) include dedicated video encoding hardware
called NVENC. It's separate from the CUDA cores and the graphics pipeline, so
encoding can happen in parallel with rendering or deep learning inference.

But NVENC needs to access the frame data somehow. The NVENC API offers two
device modes for this:

- **OpenGL mode**: takes OpenGL texture IDs
- **CUDA mode**: takes CUDA device pointers

Both are zero-copy in the sense that NVENC reads directly from the GPU memory of the texture or buffer. The main
difference lies in what kind of GPU context you're running in.

Direct OpenGL to NVENC
----------------------

The simplest approach is to give NVENC the OpenGL texture directly. Initialize the encoder with ``NV_ENC_DEVICE_TYPE_OPENGL``, register the
texture, and encode:

.. code-block:: python

    # Simplified from encoder.py
    nvenc.nvEncOpenEncodeSessionEx(
        device=None,  # current GL context
        deviceType=NV_ENC_DEVICE_TYPE_OPENGL
    )

    # Register the texture
    nvenc.nvEncRegisterResource(
        resourceType=NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX,
        resourceToRegister=texture_id
    )

This works well, but NVENC's OpenGL mode only works with GLX
contexts, so you need an X11 display. On a headless server or with an
EGL context (common in containerized rendering), this path isn't available.

The CUDA Intermediary Way
-------------------------

For headless rendering, we need the CUDA device mode. But we still have an
OpenGL texture, so we also need to get from GL to CUDA without copying.

With CUDA-GL interop, the same GPU memory that backs an OpenGL texture
can be accessed as a CUDA array without copying. The steps are:

.. code-block:: python

    # Register the GL texture with CUDA
    resource = cuGraphicsGLRegisterImage(
        texture_id,
        GL_TEXTURE_2D,
        CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY
    )

    # Map it to get a CUarray handle
    cuGraphicsMapResources(resource)
    cu_array = cuGraphicsSubResourceGetMappedArray(resource)

    # Now cu_array points to the same memory as the GL texture

With the CUarray in hand, we can register it with NVENC in CUDA mode:

.. code-block:: python

    nvenc.nvEncOpenEncodeSessionEx(
        device=cuda_context,
        deviceType=NV_ENC_DEVICE_TYPE_CUDA
    )

    nvenc.nvEncRegisterResource(
        resourceType=NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
        resourceToRegister=cu_array
    )

The path is: GL texture → CUarray → NVENC. But these are all just different views
of the same GPU memory. No pixel data is copied at any step.

There's bookkeeping involved such as mapping and unmapping resources, synchronizing
access, but the expensive part (moving pixels) never happens.

Why ctypes Instead of PyNvVideoCodec?
-------------------------------------

NVIDIA provides PyNvVideoCodec, a Python wrapper for their Video Codec SDK.
It handles all the low-level NVENC complexity and accepts GPU memory via
Python's ``__cuda_array_interface__`` protocol. This sounds like a perfect fit, but it doesn't work for our use case.

The problem is that ``__cuda_array_interface__`` expects a **linear device pointer**.
When you map a GL texture to CUDA via ``cuGraphicsGLRegisterImage``, you don't get
a linear pointer but a **CUarray**, which is a handle to 2D texture memory with
its own tiled layout.

PyNvVideoCodec doesn't accept a CUarray. You'd have to copy it to linear memory first
(via ``cuMemcpy2D``), then wrap that as a CuPy or PyTorch array. That copy is faster than a full readback to host memory, but it still isn't zero-copy.

NVENC's C API, however, has ``NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY``, which can
register a CUarray directly as input. PyNvVideoCodec doesn't expose this; it's
hidden behind the ``__cuda_array_interface__`` abstraction.

So framepump loads ``libnvidia-encode.so`` via ctypes and calls the C API directly.
The bindings are straightforward structures mirroring the C headers - mostly
boilerplate. But it gives us access to the CUarray input path that PyNvVideoCodec
doesn't offer.

Automatic Path Selection
------------------------

framepump picks the right encoder automatically based on whether an X11 display
is available:

.. code-block:: python

    def _is_headless():
        return not os.environ.get('DISPLAY')

    # In GLSequenceWriter._open():
    if _is_headless():
        encoder = NvencCudaEncoder(width, height, ...)  # CUDA path, works with EGL
    else:
        encoder = NvencEncoder(width, height, ...)       # OpenGL path, simpler

If ``DISPLAY`` is set, we assume GLX and use the direct OpenGL path. Otherwise,
we use the CUDA path with GL interop.

The Output Side
---------------

NVENC produces raw H.264 (or HEVC) NAL units — just the compressed video data,
no container. To get a playable file, we need to mux this into a container
format like MP4 or MKV.

framepump uses PyAV for muxing. Each encoded packet from NVENC is wrapped in an
``av.Packet`` with PTS/DTS timestamps and muxed into the output container:

.. code-block:: python

    output = av.open('output.mp4', 'w')
    stream = output.add_stream('h264', rate=fps)

    for encoded in encoder.encode(texture):
        packet = av.Packet(encoded.data)
        packet.stream = stream
        packet.pts = encoded.pts
        packet.dts = encoded.dts
        output.mux(packet)

PyAV handles container writing, audio interleaving, and all the muxing edge cases
in-process (no subprocess). The computational cost of muxing compressed packets is
negligible compared to the encoding itself.

Synchronization
---------------

There's one catch with zero-copy: you need to make sure the GPU is done
rendering before NVENC starts reading the texture. OpenGL commands are
queued asynchronously - when ``render_to_texture()`` returns, the GPU may
still be working on it.

The fix is ``ctx.finish()`` (or ``glFinish()``), which blocks until all
pending GL commands complete. Only then is it safe to hand the texture to
NVENC.

.. code-block:: python

    render_to_texture(texture)
    ctx.finish()  # wait for GPU to finish rendering
    encoder.encode(texture)  # now NVENC can safely read it

This makes the pipeline fully sequential: render, wait, encode, repeat.
Note that ``encode()`` is also blocking - it doesn't return until NVENC
has finished compressing the frame. So there's no risk of the next render
overwriting the texture while encoding is still reading it.

The sequential approach means the rendering parts of the GPU sit idle while waiting for the encoding hardware components,
and vice versa. Double-buffering (render to texture A while encoding texture B)
could overlap these operations for higher throughput.

However, NVENC's async mode (``enableEncodeAsync``) is **Windows-only**. On Linux,
``nvEncEncodePicture()`` always blocks until encoding completes. True async
pipelining would require a separate encoding thread, but OpenGL contexts are
single-threaded on Linux - encoder operations must run on the same thread that
created the GL context.

Is it worth the complexity? Timing measurements from
`PoseViz <https://github.com/isarandi/poseviz>`_ (a 3D pose visualization tool
using GLVideoWriter) show typical per-frame costs:

- Render scene: ~0.2 ms
- ``ctx.finish()``: ~0.3 ms
- NVENC encode: ~2.0 ms

Encoding dominates at roughly 6x the render time. Double-buffering would overlap
the ~0.5 ms render+sync with the 2 ms encode, saving at most 20% per frame. Given
the added complexity of managing an FBO pool and GL fences, the simple sequential
approach is the better tradeoff for most use cases.

Putting It All Together
-----------------------

The high-level API hides all of this:

.. code-block:: python

    from framepump import GLVideoWriter

    with GLVideoWriter('output.mp4', fps=30) as writer:
        for _ in render_loop:
            render_to_texture(texture)
            ctx.finish()  # ensure rendering is complete
            writer.append_data(texture)

Under the hood: path selection happens at initialization, textures get
registered and encoded via NVENC, raw H.264 flows to FFmpeg, and a playable
video file comes out. The pixel data never touches the CPU or the host memory.
