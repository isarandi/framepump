Handling Errors
===============

Everything FramePump raises on its own behalf derives from
:class:`~framepump.FramePumpError`, so one handler catches all
library-specific failures:

.. code-block:: python

    from framepump import FramePumpError, VideoFrames

    try:
        for frame in VideoFrames('input.mp4'):
            process(frame)
    except FramePumpError as e:
        print(f'Skipping unreadable video: {e}')

Standard Python exceptions are still used where they are the natural fit:
``FileNotFoundError`` for missing paths, ``IndexError`` for out-of-range
frame indices, ``ValueError``/``TypeError`` for invalid arguments.

The subclasses and when they are raised:

:class:`~framepump.VideoDecodeError`
    Decoding failed: corrupt or truncated data, an FFmpeg-level decode
    error, or a stream that produces no decodable frames.
:class:`~framepump.VideoEncodeError`
    Encoding failed while writing video.
:class:`~framepump.UnsupportedCodecError`
    The FFmpeg build has no decoder for the file's video codec.
:class:`~framepump.NoVideoStreamError`
    The file contains no video stream (e.g. an audio-only file).
:class:`~framepump.NoAudioStreamError`
    No longer raised (kept exported for compatibility): an audio source
    without an audio stream (e.g. via ``audio_source_path`` or
    :func:`~framepump.video_audio_mux`) now simply produces video-only
    output.
:class:`~framepump.IndexBuildError`
    Building the frame index failed (no valid frames found in the file).
:class:`~framepump.FilterConfigError`
    Configuring the format/resize filter graph failed.


Diagnosing Environment Issues
-----------------------------

When a GPU feature fails with a low-level error (CUDA context/out-of-memory
errors, missing NVENC/NVDEC, import errors from the CUDA extras), run
:func:`framepump.diagnose` — it prints the FFmpeg/PyAV versions in use, the
NVIDIA driver and GPUs, the availability of each GPU feature with the
concrete reason when one is missing, and the connected cameras:

.. code-block:: python

    >>> import framepump
    >>> framepump.diagnose()
    framepump 0.3.1
    python 3.10.14 on Linux-6.8.0-x86_64
    PyAV 17.1.0 (libavcodec 62.28.101, ...)
    FFmpeg CUDA decoders (h264_cuvid): present
    NVIDIA driver: ...
    GPU features:
      VideoFramesCuda / frame index: available
      CameraFrames: available
      ...

Attach its output to bug reports.
