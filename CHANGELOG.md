# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Random access on streams whose containers misreport keyframes (screen codecs like
  FIC/VMware, open-GOP MPEG-1/2 in TS/raw containers, packed-B MPEG-4) silently
  returned different pixels than iteration — up to near-garbage frames. `VideoFrames`
  now verifies at open time that seeking reproduces sequential decoding for these
  codecs and transparently falls back to decode-from-start access when it does not.
  Consistent files (the vast majority) keep fast seeking.
- On such streams the packet index could also count frames the decoder never produces,
  so `len()` disagreed with iteration and indexing was misaligned; the index is rebuilt
  from actual decoder output when the verification fails. A file whose frames cannot be
  decoded at all now raises `VideoDecodeError` instead of silently yielding nothing.
- `trim_video` auto-detection picked NVENC for videos below the hardware's minimum
  encode size (145x49) — including videos framepump itself writes — and died with a raw
  `[Errno 22] Invalid argument`; auto-detection now falls back to CPU encoding, and an
  explicit `gpu=True` raises a clear `ValueError`.
- Opening a file whose codec has no decoder in the FFmpeg build (JPEG-XL, VVC, EVC, ...)
  crashed with `AttributeError`; now raises the new `UnsupportedCodecError`. Audio-only
  files raise the new `NoVideoStreamError` instead of a plain `ValueError` (both are
  `FramePumpError` subclasses and exported). Streams probing with no valid dimensions
  (e.g. animated WebP without decoder support) raise a clean `VideoDecodeError` at open.
- All FFmpeg-level decode failures (unknown decoder errors, unimplemented features,
  DRM permission errors, ...) are wrapped into `VideoDecodeError` — previously only
  invalid-data errors were, and the rest leaked as raw `av.error.*` internals.
- `VideoFramesCuda` indexed access and offset slices segfaulted outright on
  driver/PyNvVideoCodec combinations where `PyNvDemuxer.Seek()` crashes (observed with
  PyNvVideoCodec 2.0.3 on driver 590.48, for every nonzero seek target). Seek support
  is now probed once per process in a throwaway subprocess; when broken, access falls
  back to decode-from-start with a warning (correct but slower).

- Writing float16 frames produced an all-black video on numpy >= 2 (NEP 50 promotion
  overflowed the scaling to inf inside float16); scaling now happens in float32.
- `VideoFrames(path, seekable=False)` raised on any integer index or positive-start slice
  (the frame index was built before the caller's `seekable` value was applied); an explicit
  `seekable` value now also actually skips the seek probe, as documented.
- Timestampless streams (raw H.264/HEVC elementary streams): positive-start slices
  (`frames[5:10]`) silently yielded zero frames in both VFR and CFR modes; the seek loops
  now fall back to frame counting like integer indexing already did.
- Numeric CFR (`constant_framerate=<fps>`) duplicated/dropped the wrong source frames:
  same frame count as ffmpeg but displaced duplicates. The vsync simulation now mirrors
  ffmpeg's exact rational rescale, midpoint bias and early-frame clipping, and is verified
  frame-exact against the ffmpeg CLI for up- and downsampling, including NTSC rates.
- `trim_video` always dropped the final frame when `end_time` was at or past the last
  frame (e.g. `trim_video(src, out, 0, get_duration(src))`); it also let one pre-start
  audio packet through with negative timestamps. Both `trim_video` and `video_audio_mux`
  now write through a temp file, so errors leave nothing at the destination.
- NVENC bindings: `NV_ENC_PIC_PARAMS` was declared 528 bytes smaller than the SDK 13.0
  struct (undersized codec-params union), so the driver read out of bounds on every
  encoded frame; layouts are now byte-verified against the SDK header by regression tests.
- `JpegVideoWriterCUDA`: an exception inside `with writer.start_sequence(path):` finalized
  the partial file instead of discarding it; a failed first frame (corrupt or unsupported
  JPEG) left the writer in a broken state that crashed retries with `ZeroDivisionError`.
- 4:4:4 chroma downsampling used Lanczos instead of the intended area averaging (the
  bound `NPPI_INTER_SUPER` constant was actually `NPPI_INTER_LANCZOS`); 4:2:0 now uses
  true area averaging and 4:2:2 a linear 2-tap average (NPP rejects SUPER there).
- The first `torch.from_dlpack` export of a GPU frame per process permanently leaked its
  NVDEC decoder session (keepalive key 0 became a NULL pointer); re-exporting a buffer
  that already handed off its memory now raises instead of producing a NULL-data tensor.
- The GL/CUDA writers' muxer silently produced a video without audio when the audio
  source had no audio stream; it now raises `NoAudioStreamError` like `VideoWriter`.
- `VideoWriter`: Ctrl+C during `close()` could leave the worker thread consuming the
  queue behind a restarted worker; worker death by a non-`Exception` no longer lets
  `close()` report success; an encoding-error cleanup that itself failed (e.g. disk full)
  could strand the temp file; `queue_size` is validated (0 meant unbounded).
- nvJPEG decoders: `close()` synchronizes in-flight async work before freeing buffers; a
  failed `parse()` no longer advances the internal slot rotation (a retry could overwrite
  an in-flight slot) and `decode_host()` after a failed parse raises instead of decoding a
  stale frame; constructor failures no longer emit `AttributeError` noise from `__del__`.
- NVENC encoders: a failed construction no longer leaks the encode session (sessions are
  capped on GeForce); a failed submission no longer desyncs the bitstream ring (flush
  could lock a never-filled buffer); oversized source textures raise instead of being
  silently cropped; partial staging-ring allocation failures no longer leak GPU objects;
  an explicit `gpu=` ordinal is honored even when another CUDA context is current.
- The chroma-interleave kernel cache is keyed by context id instead of context address
  (stale kernels after context destroy/re-create at the same address), and its PTX
  declares version 7.0 instead of 13.1-era 9.1, so CUDA 12/13.0 drivers can load it.
- `dtype='float32'`, `dtype=float` and other `DTypeLike` spellings are accepted by
  `VideoFrames` instead of being rejected as unsupported.
- Damaged-but-indexable files: sliced/indexed/CFR access now wraps decoder errors into
  `VideoDecodeError` and tolerates malformed EOF markers, like sequential iteration.
- On CPU-only machines, instantiating `JpegVideoWriterCUDA`/`VideoFramesCuda`/
  `CudaToGLUploader` raises an ImportError naming the missing CUDA dependencies instead
  of `TypeError: 'NoneType' object is not callable`.
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

- `VideoFrames` and `VideoFramesCuda` constructor options (`dtype`, `gpu`,
  `constant_framerate`, `seekable`, ...) are keyword-only: the two classes previously
  disagreed on positional order, so positional calls migrated between them silently
  misassigned arguments.
- `GLVideoWriter` raises `ValueError` for `EncoderConfig` values it cannot honor
  (`codec` other than `'h264'`, `preset` other than `'p4'`/None) instead of silently
  encoding H.264 at P4.
- `AbstractVideoWriter.start_sequence` signature now matches the implementations
  (first parameter `video_output`, optional `fps`, `encoder_config` parameter).
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

- Lazy frame indexing: opening a video no longer scans the whole file. Forward
  iteration, prefix-style slicing (`frames[:100]`, `frames[::2]`) and slice chains that
  reduce to a plain forward slice (via the new `slicecompose` dependency) stream in a
  single pass without ever building the index; `len()`, integer indexing, negative
  bounds, CFR mode and reverse iteration build it on first use, once, shared across all
  views. Note that `list(frames)` calls `len()` as a preallocation hint and therefore
  builds the index; a `for` loop or comprehension streams without it.
- Negative-step slicing: `frames[::-1]`, `frames[100:10:-3]` and friends iterate in
  reverse, decoded internally as memory-bounded forward chunks anchored at keyframes
  (large step magnitudes use per-frame seeking). Composes with slicing, CFR mode,
  `repeat_each_frame()`, resizing and all dtypes.
- `VideoFramesCuda` gained the same lazy indexing and negative-step slicing: forward
  streaming access never scans packets, and reverse iteration buffers chunk frames in
  owned GPU copies (safe to keep without `.clone()`, unlike forward iteration's shared
  buffers).
- File-like objects (anything with `read`/`seek`/`tell`, e.g. `BytesIO`) are accepted as
  video sources by `VideoFrames`. `BytesIO` sources support multiple concurrently active
  iterators; other file-like objects support one.
- `IndexBuildError` and `FilterConfigError` are exported from the package root.

## [0.2.0] - 2026-03-09

### Added

- Full implementation released: lazy sliceable decoding with frame indexing and seeking,
  constant-framerate (CFR) mode, documentation and test suite.
- GPU decoding via `gpu=True` or `gpu=<device_ordinal>` (`VideoFramesCuda`)
- GPU encoders: `GLVideoWriter` (OpenGL → NVENC) and `JpegVideoWriterCUDA`
  (nvJPEG → NVENC), configured via `EncoderConfig`
- High bit depth (10-bit) video support
- Audio muxing support in `VideoWriter`

## [0.1.3] - 2025-05-21

### Fixed

- Packaging fixes (also 0.1.2, same day).

## [0.1.1] - 2025-05-21

### Added

- Initial public release: `VideoFrames` for lazy, sliceable video frame access and
  threaded `VideoWriter`.
