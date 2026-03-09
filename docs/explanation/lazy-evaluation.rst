Lazy Evaluation and Chaining
============================

:class:`~framepump.VideoFrames` is designed so that creating objects is cheap
and actual I/O only happens when you consume frames. This page explains what's
lazy, what triggers real work, and how operations compose.


What Happens on Construction
----------------------------

When you create a ``VideoFrames``, the constructor does two things:

1. **Opens the container** to read stream metadata (resolution, fps, duration).
2. **Builds a packet index** by scanning all packets in the file. This reads
   packet headers but does not decode any frames.

After the index is built, the file handle is closed. No pixel data has been
touched.

.. code-block:: python

    # This scans packets and builds the index, then closes the file.
    # It does NOT decode any frames.
    frames = VideoFrames('input.mp4')

The cost of construction is proportional to the number of packets (typically
a few milliseconds for a short video, up to a few hundred milliseconds for
a multi-hour file).


What Triggers I/O
-----------------

Two operations actually decode frames:

- **Iteration** (``for frame in frames``) opens a fresh file handle and
  decodes frames on the fly.
- **Integer indexing** (``frames[42]``) opens a file handle, seeks to the
  nearest safe point, decodes forward to the target frame, and closes the
  handle.

Everything else is metadata manipulation.


Slicing is O(1)
---------------

:class:`~framepump.VideoFrames` stores a Python ``range`` object that
represents which frames to decode during iteration. Slicing creates a new
instance with a modified range:

.. code-block:: python

    frames = VideoFrames('input.mp4')   # range(0, 1000)
    subset = frames[100:500:2]          # range(100, 500, 2)
    smaller = subset[:50]               # range(100, 200, 2)

No frames are decoded. The cost is one ``range.__getitem__`` call and one
shallow copy of the ``VideoFrames`` metadata.

Sliced instances share the same packet index (it's read-only, so sharing is
safe).


Cloning Operations
------------------

Several methods return a new :class:`~framepump.VideoFrames` clone with
modified parameters:

- ``frames[a:b:c]`` — narrows the frame range
- ``frames.resized((h, w))`` — sets a target resize shape
- ``frames.repeat_each_frame(n)`` — sets a repeat multiplier

These clones share the same packet index and path, but have independent
metadata. None of them trigger I/O.


How Iteration Works
-------------------

When you iterate, ``__iter__`` picks a decode strategy based on the
frame range:

1. **Start at 0, small step**: decode sequentially from the beginning.
   This is the fastest path — no seeking, just linear decode with optional
   step skip via ``islice``.

2. **Start after 0, small step**: seek once to the start position, then
   decode forward. The seek uses the packet index to find the nearest safe
   keyframe before the target.

3. **Large step (> 30)**: seek individually to each frame. When the step is
   large enough, it's faster to seek per-frame than to decode and discard
   the frames in between.

Each iteration opens a fresh file handle and closes it when done. Multiple
iterations over the same ``VideoFrames`` are independent.


Resizing in the Filter Graph
-----------------------------

When ``resized()`` has been called, the decode pipeline includes an FFmpeg
scale filter. The resize happens as part of the decode — frames are never
decoded at full resolution and then downscaled in Python. The filter graph
runs in FFmpeg's optimized C code.


repeat_each_frame
-----------------

``repeat_each_frame(n)`` doesn't actually duplicate data. During iteration,
each decoded frame is yielded ``n`` times before decoding the next. This means
the memory cost is one frame regardless of the repeat count.

Slicing and ``repeat_each_frame`` don't compose in the other order — you
cannot slice a repeated video because the repeat multiplier changes the
mapping between output indices and source frames in a way that doesn't map
cleanly onto a ``range``. Apply slicing first:

.. code-block:: python

    # This works: slice first, then repeat
    frames[::2].repeat_each_frame(3)

    # This raises NotImplementedError
    frames.repeat_each_frame(3)[::2]
