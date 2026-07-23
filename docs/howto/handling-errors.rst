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
    An audio source was given (e.g. ``audio_source_path`` or
    :func:`~framepump.video_audio_mux`) but it has no audio stream.
:class:`~framepump.IndexBuildError`
    Building the frame index failed (no valid frames found in the file).
:class:`~framepump.FilterConfigError`
    Configuring the format/resize filter graph failed.
