Getting Started
===============

This tutorial walks through the basics of reading and writing video with
FramePump. By the end, you'll be able to open a video, inspect its metadata,
access individual frames, and write processed frames to a new file.


Installation
------------

.. code-block:: bash

    pip install framepump


Opening a Video
---------------

The main entry point is :class:`~framepump.VideoFrames`. Creating an instance
reads only metadata — no pixel data is loaded yet.

.. code-block:: python

    from framepump import VideoFrames

    frames = VideoFrames('input.mp4')

You can inspect the video's properties immediately:

.. code-block:: python

    print(frames.fps)       # Frame rate, e.g. 29.97
    print(frames.imshape)   # (height, width) in pixels, e.g. (1080, 1920)
    print(len(frames))      # Exact frame count

The frame count is exact, not an estimate. FramePump builds a packet index
on construction by scanning the container without decoding any frames.


Iterating Over Frames
---------------------

Iterating decodes frames on the fly and yields numpy arrays:

.. code-block:: python

    for frame in frames:
        # frame.shape is (height, width, 3), dtype is uint8
        print(frame.shape, frame.dtype)
        break  # just check the first frame

Each frame is an RGB numpy array. Decoding happens lazily — only the frames
you consume are decoded.


Accessing a Single Frame
------------------------

You can index into the video like a list:

.. code-block:: python

    first = frames[0]
    last = frames[-1]
    middle = frames[len(frames) // 2]

Random access works correctly even for videos with B-frames. FramePump seeks
to the nearest safe point and decodes forward to the target frame.


Slicing
-------

Slicing is lazy — it returns a new :class:`~framepump.VideoFrames` without
reading any pixel data:

.. code-block:: python

    # Every other frame from the first 100
    subset = frames[:100:2]
    print(len(subset))  # 50

    # Slices can be chained
    smaller = frames[::4][:25]  # Every 4th frame, then take first 25

The resulting object supports the same operations: iteration, indexing,
further slicing, and resizing.


Resizing
--------

:meth:`~framepump.VideoFrames.resized` returns a new lazy instance that
decodes frames at the given resolution:

.. code-block:: python

    # shape is (height, width), following numpy convention
    small = frames.resized((128, 228))
    print(small.imshape)  # (128, 228)

    for frame in small:
        assert frame.shape == (128, 228, 3)
        break

Resizing is done in FFmpeg's filter graph, so it's fast and uses the same
high-quality scaling as the ``ffmpeg`` command-line tool.

.. note::

    :meth:`~framepump.VideoFrames.resized` takes ``(height, width)``, matching
    numpy array shape convention. This is the opposite of
    :func:`~framepump.video_extents`, which returns ``(width, height)``.


Data Types
----------

By default, frames are ``uint8`` (0–255). Other dtypes are available:

.. code-block:: python

    import numpy as np

    # 16-bit integer (preserves 10-bit source precision)
    frames_16 = VideoFrames('input.mp4', dtype=np.uint16)

    # Float (normalized to 0.0–1.0)
    frames_f32 = VideoFrames('input.mp4', dtype=np.float32)

Supported dtypes: ``uint8``, ``uint16``, ``float16``, ``float32``, ``float64``.

Videos with high bit depth (e.g. 10-bit color) retain their full precision when
decoded to ``uint16`` or any float dtype. With the default ``uint8``, the extra
precision is quantized down to 8 bits.


Writing a Video
---------------

:class:`~framepump.VideoWriter` encodes frames in a background thread,
so your main code isn't blocked by encoding:

.. code-block:: python

    import numpy as np
    from framepump import VideoWriter

    with VideoWriter('output.mp4', fps=30) as writer:
        for i in range(90):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[i*5:(i+1)*5, :] = 255  # moving white bar
            writer.append_data(frame)

The context manager waits for all frames to be encoded and closes the file.
The output file is written atomically — a temporary file is used during
encoding, then renamed to the final path on success.


Reading and Writing Together
----------------------------

A common pattern: read a video, process frames, write the result.

.. code-block:: python

    from framepump import VideoFrames, VideoWriter, get_fps

    src = VideoFrames('input.mp4')
    fps = src.fps

    with VideoWriter('grayscale.mp4', fps=fps) as writer:
        for frame in src.resized((480, 640)):
            # Convert to grayscale, keep 3 channels for H.264
            gray = frame.mean(axis=2, keepdims=True).astype(np.uint8)
            gray = np.broadcast_to(gray, frame.shape).copy()
            writer.append_data(gray)


Next Steps
----------

- :doc:`/howto/frame-accurate-processing` — slicing, frame counting, VFR handling
- :doc:`/howto/batch-video-writing` — writing multiple videos efficiently
- :doc:`/howto/gpu-acceleration` — hardware-accelerated decode and encode
