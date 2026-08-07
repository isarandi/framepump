Running a Model over a Video
============================

The typical inference pipeline — decode, preprocess, batch, predict, and
write results back — in one place. Everything here composes from pieces
described elsewhere; this page just assembles them.


CPU pipeline (numpy in, numpy out)
----------------------------------

``VideoFrames`` handles resizing and dtype conversion during decoding, so
frames arrive model-ready. Float dtypes are scaled to [0, 1].

.. code-block:: python

    import numpy as np
    from framepump import VideoFrames

    frames = VideoFrames('input.mp4', dtype=np.float32).resized((224, 224))

    predictions = []
    for batch in frames.batched(32):
        # batch: (n, 224, 224, 3) float32 in [0, 1]; one sequential decode
        # pass overall, each batch freshly allocated (safe to keep)
        predictions.extend(model(batch))

If you only need specific frames (e.g. one per second), an index list decodes
them in a single gap-aware pass:

.. code-block:: python

    wanted = list(range(0, len(frames), round(frames.fps)))
    batch = frames[wanted]           # (n, 224, 224, 3), eager
    # or lazily, in a streaming fashion:
    for frame in frames.frames_at(wanted):
        ...


GPU pipeline (CUDA tensors, no CPU round-trip)
----------------------------------------------

``VideoFramesCuda`` decodes with NVDEC and hands frames to PyTorch via
DLPack with no copy — preprocessing (resize, dtype, [0, 1] scaling) runs on
the GPU too:

.. code-block:: python

    import numpy as np
    import torch
    from framepump import VideoFramesCuda

    frames = VideoFramesCuda('input.mp4', dtype=np.float32).resized((224, 224))

    predictions = []
    for batch in frames.batched(32):
        tensors = torch.from_dlpack(batch)  # (n, 224, 224, 3) CUDA tensor
        with torch.inference_mode():
            predictions.extend(model(tensors.permute(0, 3, 1, 2)))

Each batch is one independently owned GPU buffer: decoding, preprocessing
and batching all happen on the GPU with a single copy per frame, and no
cloning or stacking is needed — the batches stay valid after iteration
advances. (When iterating frame by frame instead, iteration yields reusable
buffers — process within the step or ``.clone()``; see
:doc:`gpu-acceleration`.)

Decoding works from any thread, so wrapping the loop body in a prefetch
thread (decode batch N+1 while the model runs on batch N) is safe.


Writing results back
--------------------

Render your predictions onto frames and write them out — with the original
audio carried over if you want it:

.. code-block:: python

    from framepump import VideoFrames, VideoWriter

    frames = VideoFrames('input.mp4')
    with VideoWriter('annotated.mp4', fps=frames.fps,
                     audio_source_path='input.mp4') as writer:
        for frame, pred in zip(frames, predictions):
            writer.append_data(draw_overlay(frame.copy(), pred))

The writer encodes on a background thread, so drawing and encoding overlap.
For matching output timing on variable-framerate input, open the reader with
``constant_framerate=True`` and pass the same fps to the writer.
