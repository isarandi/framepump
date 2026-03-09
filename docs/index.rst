FramePump
=========

A Python library for high-performance video processing, built on PyAV (Python bindings for FFmpeg's libraries).

This project provides:

- Lazy, sliceable video frame access via ``VideoFrames``
- Threaded video writing via ``VideoWriter``
- Zero-copy GPU encoding via ``GLVideoWriter`` (NVENC)
- GPU-accelerated decoding support
- High bit depth (10-bit) video support

Installation
------------

.. code-block:: bash

    pip install framepump

For zero-copy GPU encoding with headless/EGL contexts:

.. code-block:: bash

    pip install framepump[nvenc-cuda]

Quick Start
-----------

Reading Video Frames
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from framepump import VideoFrames

    # Lazy loading - only reads metadata
    frames = VideoFrames('my_video.mp4')

    # Iterate over frames
    for frame in frames:
        # frame is a numpy array of shape (height, width, 3)
        pass

    # Slice the video (lazy)
    subset = frames[:100:2]  # Every second frame of first 100

    # Resize on the fly — shape is (height, width)
    resized = frames.resized((128, 128))

Writing Videos
~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from framepump import VideoWriter

    with VideoWriter('output.mp4', fps=30) as writer:
        for i in range(100):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            writer.append_data(frame)

Zero-Copy GPU Encoding
~~~~~~~~~~~~~~~~~~~~~~

For real-time rendering, encode OpenGL textures directly to video without
CPU memory transfers:

.. code-block:: python

    from framepump import GLVideoWriter

    with GLVideoWriter('output.mp4', fps=30) as writer:
        for _ in render_loop:
            render_to_texture(texture)
            ctx.finish()  # Wait for GPU to finish rendering
            writer.append_data(texture)  # Encode directly from GPU

This uses NVIDIA's NVENC hardware encoder. See :doc:`explanation/nvenc-zero-copy`
for details on how it works.

API Reference
-------------

See the full API documentation at :mod:`framepump`.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   tutorial/index
   howto/index
   explanation/index
   API Reference <api/framepump/index>

* :ref:`genindex`