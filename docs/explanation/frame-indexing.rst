Frame Indexing
==============

FramePump builds a packet index when you open a video, giving exact frame
counts and correct random access. This page explains why that's necessary and
how it works.


Why Naive Seeking Is Wrong
--------------------------

Video containers store frames in **decode order**, which can differ from
**display order**. H.264 and HEVC use B-frames (bidirectional predicted
frames) that reference both past and future frames. A typical packet sequence
might look like:

::

    Decode order:  I₀  P₃  B₁  B₂  P₆  B₄  B₅  ...
    Display order: I₀  B₁  B₂  P₃  B₄  B₅  P₆  ...

The packets arrive in decode order (I, P, B, B, P, B, B, ...), but the PTS
(Presentation Time Stamp) values indicate display order.

If you simply seek to a byte position and start decoding, you might land in
the middle of a B-frame sequence without the reference frames it needs. The
result is either a corrupted frame or a decode error.


The Packet Index
----------------

When you create a :class:`~framepump.VideoFrames`, FramePump scans all
packets in the container and collects two things for each packet:

- Its **PTS** (when this frame should be displayed)
- The **running maximum PTS** seen so far in packet order

This scan reads only packet headers — no frames are decoded, so it's fast
(typically milliseconds).

From this data, FramePump computes:

- **frame_pts**: a sorted list of unique PTS values. This is the display
  order. Its length is the exact frame count.

- **safe_seek_pts**: for each frame, the latest packet position where all
  packets needed to decode the target frame have already been seen.

All timestamps are stored as exact fractions (``fractions.Fraction``) to avoid
floating-point rounding errors.


Safe Seek Points
----------------

The key insight is the **running maximum**: at any point in the packet stream,
the running max tells you the latest display timestamp that has been fully
"delivered" — all packets with PTS up to that value have been read.

To find a safe seek point for a target frame:

- Find the latest position in the packet stream where the running maximum is
  less than or equal to the target PTS.
- Seeking to that position guarantees that when you decode forward, all
  reference frames are available.

This handles B-frames correctly without parsing codec-specific bitstream
syntax.

When you access ``frames[42]``, FramePump:

1. Looks up the target PTS and the safe seek point from the index.
2. Seeks to the safe position (at or before a keyframe).
3. Decodes forward, comparing each frame's PTS to the target.
4. Returns the matching frame.


Why Duration × FPS Is Wrong
----------------------------

A common shortcut for estimating frame count is ``duration × fps``. This is
unreliable for several reasons:

- **Variable frame rate (VFR)** videos have inconsistent gaps between frames.
  The average FPS doesn't predict the actual packet count.

- **Container metadata can be wrong.** The duration field in some containers
  is approximate or based on audio duration rather than video.

- **Rounding errors** accumulate. A 29.97 fps video that's 60.06 seconds long
  has 1800 frames, but ``round(60.06 × 29.97)`` gives 1800 or 1801 depending
  on floating-point precision.

FramePump avoids all of this by counting actual packets.


Variable Frame Rate
-------------------

VFR videos have non-uniform frame timing — some frames last longer than
others. FramePump handles VFR correctly by default: each frame has its own
PTS, and iteration yields frames in display order with their original timing.

When you need constant frame rate output (e.g. for tools that expect uniform
timing), use ``constant_framerate``:

.. code-block:: python

    # CFR at the original average fps
    frames = VideoFrames('vfr.mp4', constant_framerate=True)

    # CFR at a specific fps
    frames = VideoFrames('vfr.mp4', constant_framerate=60.0)

In CFR mode, FramePump simulates FFmpeg's frame duplication/dropping algorithm
to produce evenly-spaced output. Frames at timestamps that don't align with
the output grid are either duplicated or dropped, matching the behavior of
``ffmpeg -vsync 1``.


Non-Seekable Streams
--------------------

Some container formats don't support seeking (raw bitstreams, image pipe
formats, certain MPEG-TS files with corrupt indices). FramePump detects
these by probing: it attempts a backward seek and checks whether the decoder
lands at or before the target position.

For non-seekable streams:

- The frame index is still built (by scanning packets from the start).
- All safe seek points are set to zero (the only safe position is the
  beginning).
- Random access works but requires decoding from the start each time, which
  is slow. Sequential iteration is unaffected.
