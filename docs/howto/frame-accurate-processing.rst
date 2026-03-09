Frame-Accurate Video Processing
================================

FramePump provides exact frame counts, correct random access even with
B-frames, and lazy slicing. This guide covers the details.


Exact Frame Counts
------------------

When you create a :class:`~framepump.VideoFrames`, the constructor scans all
packets in the container (without decoding) and builds an index. This gives an
exact frame count:

.. code-block:: python

    from framepump import VideoFrames

    frames = VideoFrames('input.mp4')
    print(len(frames))  # exact

The standalone :func:`~framepump.num_frames` function offers three accuracy
levels:

.. code-block:: python

    from framepump import num_frames

    # Fast estimate (duration × fps) — may be wrong for VFR
    n = num_frames('input.mp4')

    # Exact: builds a packet index (same as len(VideoFrames(...)))
    n = num_frames('input.mp4', exact=True)

    # Decode every frame and count (slowest, handles edge cases)
    n = num_frames('input.mp4', absolutely_exact=True)

For most uses, ``len(frames)`` or ``exact=True`` gives the correct answer.
The estimate (default) can be off for variable-frame-rate videos where
duration × fps doesn't reflect the actual packet count.


Random Access
-------------

Integer indexing decodes the exact frame you ask for:

.. code-block:: python

    frame_100 = frames[100]
    last_frame = frames[-1]

This works correctly even for H.264 and HEVC videos with B-frames. FramePump
knows which keyframe to seek to and how many frames to decode forward to reach
the target.

Out-of-range indices raise ``IndexError``:

.. code-block:: python

    frames[len(frames)]  # IndexError


Slicing and Chaining
--------------------

Slicing is O(1) — it creates a new :class:`~framepump.VideoFrames` that
references a sub-range, without reading any pixel data:

.. code-block:: python

    # Every 3rd frame from index 100 to 200
    subset = frames[100:200:3]
    print(len(subset))  # 34

    # Slices compose — each one narrows the range further
    every_other = frames[::2]      # 0, 2, 4, 6, ...
    first_fifty = every_other[:50] # 0, 2, 4, ..., 98
    print(len(first_fifty))        # 50

Sliced instances support all the same operations: iteration, indexing,
further slicing, and resizing.

.. code-block:: python

    # Chain slicing and resizing
    for frame in frames[::10][:20].resized((128, 128)):
        process(frame)  # 20 frames, every 10th, at 128×128


Frame Repetition
----------------

:meth:`~framepump.VideoFrames.repeat_each_frame` duplicates each frame a given
number of times during iteration:

.. code-block:: python

    repeated = frames.repeat_each_frame(3)
    print(len(repeated))  # len(frames) * 3

    # The effective fps changes accordingly
    print(repeated.fps)   # frames.fps * 3

.. note::

    Slicing must be applied *before* ``repeat_each_frame()``, not after.
    ``frames[::2].repeat_each_frame(3)`` works;
    ``frames.repeat_each_frame(3)[::2]`` raises ``NotImplementedError``.


Variable Frame Rate (VFR) Videos
--------------------------------

Some videos (screen recordings, phone videos) have variable frame rates — the
gap between frames isn't constant. By default, FramePump preserves original
timestamps, so you get every frame as it exists in the file.

To convert a VFR video to constant frame rate, use ``constant_framerate``:

.. code-block:: python

    # Convert to CFR at the video's original fps
    frames = VideoFrames('vfr_video.mp4', constant_framerate=True)

    # Convert to a specific fps (e.g. 60)
    frames = VideoFrames('vfr_video.mp4', constant_framerate=60.0)

In CFR mode, FramePump simulates FFmpeg's frame duplication/dropping algorithm
(vsync=1) to produce evenly-spaced output. Frames that fall between output
positions are duplicated or dropped as needed.


Dimension Conventions
---------------------

FramePump uses two conventions for dimensions, matching the standard practice
in their respective domains:

- :attr:`~framepump.VideoFrames.imshape` returns ``(height, width)`` —
  matching numpy array shapes and image processing convention.

- :func:`~framepump.video_extents` returns ``(width, height)`` as a numpy
  array — matching the video/graphics convention used by FFmpeg, OpenGL, etc.

.. code-block:: python

    from framepump import VideoFrames, video_extents

    frames = VideoFrames('input.mp4')
    h, w = frames.imshape        # numpy convention
    w, h = video_extents('input.mp4')  # video convention

:meth:`~framepump.VideoFrames.resized` takes ``(height, width)`` to match
``imshape``.
