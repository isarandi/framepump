Lazy Evaluation and Chaining
============================

:class:`~framepump.VideoFrames` is designed so that creating objects is cheap
and actual I/O only happens when you consume frames. This page explains what's
lazy, what triggers real work, and how operations compose.


What Happens on Construction
----------------------------

When you create a ``VideoFrames``, the constructor only **opens the container
to read stream metadata** (resolution, fps, duration) and closes it again. No
packets are scanned and no frames are decoded.

The **packet index** — the structure behind exact ``len()``, integer indexing
and seeking — is built lazily, the first time an operation needs it. Forward
iteration and prefix-style slicing (``frames[:100]``, ``frames[::2]``) stream
the file in a single pass and never build the index at all. Operations that
need it (``len()``, integer indexing, negative bounds or steps, CFR mode)
build it once, on first use, shared across all views of the same video.

.. code-block:: python

    # Opens the file only to read metadata; no packet scan, no decoding.
    frames = VideoFrames('input.mp4')

    for frame in frames[:100]:   # streams directly — still no index
        ...

    n = len(frames)              # NOW the packet index is built (once)

Building the index costs one packet scan, proportional to the number of
packets (typically a few milliseconds for a short video, up to a few hundred
milliseconds for a multi-hour file).

Note that metadata comes from container and packet headers, not from
decoding: on a damaged file, construction and even ``len()`` can succeed
while iterating later raises :class:`~framepump.VideoDecodeError`.


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

:class:`~framepump.VideoFrames` stores a symbolic slice chain that
represents which frames to decode during iteration. Slicing creates a new
instance with an extended chain; the chain is resolved against the real
frame count only when an operation needs it (and chains that reduce to a
plain forward slice stream without ever resolving):

.. code-block:: python

    frames = VideoFrames('input.mp4')   # all frames
    subset = frames[100:500:2]          # lazy, no I/O
    smaller = subset[:50]               # lazy, still no I/O

No frames are decoded and no index is built. The cost is one slice
composition and one shallow copy of the ``VideoFrames`` metadata.

All views of the same video share one lazily built packet index, so it is
built at most once no matter how many slices exist.


Cloning Operations
------------------

Several methods return a new :class:`~framepump.VideoFrames` clone with
modified parameters:

- ``frames[a:b:c]`` — narrows the frame range
- ``frames.resized((h, w))`` — sets a target resize shape
- ``frames.repeat_each_frame(n)`` — sets a repeat multiplier

These clones share the same packet index and path, but have independent
metadata. None of them trigger I/O.

Derived properties follow the view: e.g. ``frames[::2].fps`` is half the
source fps, and ``repeat_each_frame(2)`` doubles it — ``.fps`` always
describes the sequence the view actually yields.


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

4. **Negative step** (``frames[::-1]``): decode memory-bounded forward
   chunks anchored at keyframes and yield them in reverse (large step
   magnitudes seek per frame, as above).

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
