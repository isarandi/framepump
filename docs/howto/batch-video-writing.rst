Batch Video Writing
===================

:class:`~framepump.VideoWriter` uses a background thread for encoding, so your
main code can continue working while frames are being written. This is
especially useful when writing many short video clips.


Basic Writing
-------------

The simplest usage — a context manager that writes one video:

.. code-block:: python

    import numpy as np
    from framepump import VideoWriter

    with VideoWriter('output.mp4', fps=30) as writer:
        for i in range(100):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.append_data(frame)

The background thread encodes frames as they arrive. When the context manager
exits, it waits for all pending frames to be encoded before closing the file.


Multiple Sequences
------------------

A single :class:`~framepump.VideoWriter` can write multiple video files.
Each call to :meth:`~framepump.VideoWriter.start_sequence` begins a new output
file:

.. code-block:: python

    writer = VideoWriter(fps=30)

    for i, clip in enumerate(clips):
        writer.start_sequence(f'clip_{i:04d}.mp4')
        for frame in clip:
            writer.append_data(frame)

    writer.close()

You don't need to call ``end_sequence()`` between clips —
``start_sequence()`` automatically closes the previous sequence before
starting the new one. The previous file is flushed and finalized in the
background thread.


Blocking vs Non-Blocking
------------------------

:meth:`~framepump.VideoWriter.end_sequence` has a ``block`` parameter that
controls whether it waits for encoding to finish:

.. code-block:: python

    writer = VideoWriter(fps=30)

    # Write first clip
    writer.start_sequence('clip_001.mp4')
    for frame in clip_1:
        writer.append_data(frame)

    # Block until the file is fully written and closed
    writer.end_sequence(block=True)
    # clip_001.mp4 is now complete on disk

    # Or: don't wait, start encoding the next clip immediately
    writer.start_sequence('clip_002.mp4')
    for frame in clip_2:
        writer.append_data(frame)
    writer.end_sequence(block=False)
    # clip_002.mp4 may still be encoding in the background
    # ...do other work here...

    writer.close()  # waits for everything to finish

Use ``block=True`` when you need the file to exist on disk before proceeding
(e.g. to upload it or pass it to another process). Use ``block=False`` when you
want to overlap encoding with other work.


Pipelining: Overlap Compute and Encode
--------------------------------------

The background thread architecture means you can prepare the next batch of
frames while the previous video is still being encoded:

.. code-block:: python

    writer = VideoWriter(fps=30)

    for video_path in input_paths:
        frames = VideoFrames(video_path)
        out_path = video_path.replace('.mp4', '_processed.mp4')

        writer.start_sequence(out_path, fps=frames.fps)
        for frame in frames:
            processed = some_processing(frame)
            writer.append_data(processed)
        # No end_sequence() — next start_sequence() auto-closes

    writer.close()

The queue (default size 32 frames) acts as a buffer. If encoding falls behind,
``append_data()`` blocks until there's room in the queue. If encoding is
faster than frame production, frames are consumed as fast as they arrive.


In-Memory Output
----------------

Write to a file-like object instead of a path. You must specify the container
format explicitly:

.. code-block:: python

    import io
    from framepump import VideoWriter

    buffer = io.BytesIO()

    with VideoWriter() as writer:
        writer.start_sequence(buffer, fps=30, format='mp4')
        for frame in frames:
            writer.append_data(frame)
        writer.end_sequence(block=True)

    video_bytes = buffer.getvalue()

This is useful for serving video over HTTP or storing in a database without
touching the filesystem.


Atomic File Writes
------------------

When writing to a file path, FramePump uses a temporary file in the same
directory as the output. The temp file is renamed to the final path only after
encoding completes successfully. This means:

- No partial files on disk if the process is interrupted
- Other processes won't see a half-written file
- If encoding fails, the temp file is deleted and any existing output file is
  left untouched


GPU Encoding
------------

Pass ``gpu=True`` to use NVIDIA's NVENC hardware encoder instead of libx264:

.. code-block:: python

    with VideoWriter('output.mp4', fps=30, gpu=True) as writer:
        for frame in frames:
            writer.append_data(frame)

Or select a specific GPU by ordinal:

.. code-block:: python

    writer = VideoWriter(fps=30, gpu=1)  # second GPU


Including Audio
---------------

Copy an audio track from an existing file into your output:

.. code-block:: python

    with VideoWriter('output.mp4', fps=30, audio_source_path='input.mp4') as writer:
        for frame in processed_frames:
            writer.append_data(frame)

Audio packets are interleaved with video during encoding — the audio is copied
without re-encoding. You can also set the audio source per-sequence:

.. code-block:: python

    writer = VideoWriter(fps=30)
    writer.start_sequence('out.mp4', audio_source_path='audio_source.mp4')
    ...


Encoder Configuration
---------------------

Fine-tune encoding quality and speed with :class:`~framepump.EncoderConfig`:

.. code-block:: python

    from framepump import EncoderConfig, VideoWriter

    config = EncoderConfig(
        crf=18,       # quality: 0–51, lower = better (default 15)
        gop=120,      # keyframe interval in frames (default 250)
        bframes=2,    # B-frames for compression (default 2)
        codec='h264', # 'h264' or 'hevc'
    )

    with VideoWriter('output.mp4', fps=30, encoder_config=config) as writer:
        ...

Preset names work across encoders — ``'slow'`` (libx264) and ``'p5'`` (NVENC)
are equivalent and auto-translated based on whether GPU encoding is active.

.. list-table:: Preset Translation
   :header-rows: 1
   :widths: 30 30

   * - libx264
     - NVENC
   * - ultrafast / superfast
     - p1
   * - veryfast
     - p2
   * - faster / fast
     - p3
   * - medium
     - p4
   * - slow
     - p5
   * - slower
     - p6
   * - veryslow
     - p7


High Bit Depth
--------------

Pass ``uint16`` frames for 10-bit encoding (``yuv420p10le``):

.. code-block:: python

    frame_16bit = np.zeros((480, 640, 3), dtype=np.uint16)
    with VideoWriter('10bit.mp4', fps=30) as writer:
        writer.append_data(frame_16bit)  # auto-selects 10-bit encoding

Float frames (``float16``/``float32``/``float64``) are auto-converted to
``uint16`` (0.0–1.0 maps to 0–65535) and encoded as 10-bit.


Graceful Shutdown
-----------------

:meth:`~framepump.VideoWriter.close` waits for all pending frames to be
encoded. :meth:`~framepump.VideoWriter.shutdown` stops immediately and
discards any frames still in the queue:

.. code-block:: python

    writer.close()     # wait for everything to finish
    writer.shutdown()  # discard remaining frames (warns if any)

The context manager calls ``close()`` on normal exit and ``shutdown()`` on
``KeyboardInterrupt`` (Ctrl+C).
