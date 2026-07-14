# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- CFR mode (`constant_framerate`): windowed reads (`frames[a:b]`) with a positive start
  returned copies of the wrong frame; `len()` could disagree with the number of frames
  iteration yields; streams with a late start PTS (e.g. MPEG-TS) reported phantom leading
  frames. All CFR behavior now derives from one source map that matches ffmpeg's vsync
  output, so count, indexing and iteration always agree.
- `repeat_each_frame()`: integer indexing returned the wrong frame for most indices and
  raised `IndexError` for valid (including negative) ones; now correct for all indices,
  including on sliced and CFR views. Non-integer repeat counts are rejected.
- `VideoWriter`: an encoding error in the background thread could deadlock `append_data`,
  `end_sequence` or `close` forever; errors, `shutdown()` and Ctrl+C promoted a truncated
  but playable file to the final path; the writer was unusable after `shutdown()` or an
  error; repeated `end_sequence(block=False)` grew an internal queue without bound. The
  lifecycle is now an explicit state machine: producer calls fail fast when the worker has
  died, and the writer is reusable afterwards.
- `JpegVideoWriterCUDA`: silently corrupted chroma for heights not divisible by 16
  (including 1080p) in all decode paths; a frame differing from the first frame's
  dimensions or subsampling corrupted GPU memory (now raises `ValueError`); aborting could
  race in-flight GPU work and strand the temp file; the NVENC input ring was smaller than
  the B-frame pipeline requires; audio-source open errors were silently swallowed.
- `NvjpegPhasedDecoder`: the documented pipelined usage pattern raced on internal buffers
  and could silently corrupt frames; the pipeline stages are now double-buffered.
- mkv output from `GLVideoWriter` and `JpegVideoWriterCUDA` failed at container open; mp4
  muxer flags are no longer sent to non-mp4 containers.
- NVENC encoders: `gop` did not set the IDR period, so keyframes appeared every 250 frames
  regardless of the requested GOP and seeking was coarser than requested; end-of-stream
  errors were swallowed (risking a hang in the subsequent bitstream lock); `flush()` was
  not idempotent; error messages now include the driver's own detail string.
- CUDA context hygiene: `NvjpegDecoder`, `NvjpegPhasedDecoder`, `NvencCudaEncoder`,
  `JpegVideoWriterCUDA` and `VideoFramesCuda` no longer leave a different CUDA context
  current than they found (and no longer leak contexts); GPU buffers returned by
  `VideoFramesCuda` are safe to free from any thread, even after the reader is closed.
- High-bit-depth GPU decoding: plane pitches are queried from the decoder and validated
  instead of inferred from pointer arithmetic, so an unexpected layout raises instead of
  producing sheared frames.
- NPP color-conversion kernels were cached per process but are context-specific, breaking
  the second writer within one process.
- File-like video sources: two interleaved iterators over one `BytesIO`-backed
  `VideoFrames` corrupted each other's read position; each reader now gets an independent
  view. For general file-like objects, only one active iterator is supported (documented).
- `EncoderConfig` validates `crf`, `bframes`, `gop`, `codec` and `preset` at construction;
  invalid values previously produced silently clamped or reinterpreted output (notably,
  `codec='h265'` silently encoded H.264).
- `CudaToGLUploader` rejects tensors with wrong dtype, device, shape or memory layout
  instead of silently producing garbled textures.

### Changed

- Errors and forced shutdown during video writing leave no output file at the destination
  path (previously a partial file could appear and look like a successful write).
- GL/CUDA NVENC encoders use preset P4 with high-quality tuning instead of whatever the
  driver enumerates first (P1, the fastest and lowest-quality preset).
- NVENC encoders copy the source through an internal staging ring, so callers may
  re-render into the same texture immediately after submitting it, even with B-frames.
- Audio interleaving uses submit-order timing consistently across all writers.
- Writing avi with B-frames emits a warning (avi stores no PTS; non-FFmpeg players may
  show timing jitter).
- Float output normalization is uniform across dtypes: values are divided by the integer
  type's maximum (65535 for uint16) in float32 before casting; float16 output was
  previously divided by 65504.
- CFR `len()` on late-starting streams no longer counts phantom leading frames (matches
  ffmpeg CLI output).
- Owned CUDA contexts are the device's primary context, interoperating cleanly with
  torch/cupy. Callers that relied on a leaked context staying current must now manage
  their own context.
- A `VideoWriter` that is garbage-collected without `close()` emits a `ResourceWarning`.

### Added

- `IndexBuildError` and `FilterConfigError` are exported from the package root.
- Initial public release
- `VideoFrames` class for lazy, sliceable video frame access
- `VideoWriter` class for threaded video writing
- GPU decoding/encoding support via `gpu=True` or `gpu=<device_ordinal>`
- High bit depth (10-bit) video support
- Audio muxing support in `VideoWriter`
