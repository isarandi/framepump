# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `VideoFrames(gpu=True)` silently decoded on the CPU: the CUDA hwaccel flags it
  passed to `av.open()` are ffmpeg CLI options that the libav libraries ignore, so
  since the 0.2.0 PyAV rewrite no GPU decoding ever happened (the 0.1.x ffmpeg-
  subprocess implementation did decode on GPU). Decoding now uses PyAV's HWAccel
  API for real NVDEC decoding, verified by decoder utilization and by construction:
  software fallback is disabled, so `gpu=True` either decodes on the GPU or raises.
  Output is bit-identical to CPU decoding (NVDEC's semi-planar frames are repacked
  losslessly so RGB conversion uses the same swscale path), including 10-bit
  streams, slicing and reverse iteration. The codec allowlist and FLV blocklist
  probe are gone — FFmpeg itself reports NVDEC compatibility at open time, without
  the extra probe open.

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
  PyNvVideoCodec 2.0.3 on driver 590.48, for every nonzero seek target). Demuxing now
  goes through PyAV — reliable seeking, Annex-B conversion via bitstream filters, and
  the same clean `NoVideoStreamError`/`UnsupportedCodecError` behavior as the CPU
  reader — with packets fed directly to the NVDEC decoder; PyNvDemuxer is no longer
  used at all.
- Writing to `.mkv` paths failed with `no container format 'mkv'` in `VideoWriter`,
  `trim_video` and `video_audio_mux` (the path suffix is not a valid libav muxer
  name); the suffix is now mapped to the proper container name, as the GL/CUDA
  writers' muxer already did.

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
- Streams whose decoder emits frames in a different order than the packet index implies
  (broken muxers, non-monotonic PTS) could return wrong frames on indexed/sliced access.
  Frame emission is now checked during decoding (free for well-behaved files: it reuses
  the frames already being decoded) and on any mismatch the access transparently degrades
  to decode-from-start, which is always correct.
- `VideoFramesCuda`: files NVDEC cannot actually decode (e.g. VP8 with resolution
  changes) raise `VideoDecodeError` instead of silently yielding zero frames or raw
  internal exceptions; interlaced JPEG streams, which NVDEC decodes at half height,
  are detected and raise instead of returning distorted frames.
- `trim_video` with an empty time range (`start == end`, or end before start) failed
  with a confusing `FileNotFoundError`, and a start time past the end of the video
  silently produced a one-frame video; both now raise a clear `ValueError`.
- `VideoWriter`: an exception inside the `with` block finalized the partially written
  sequence to the final path; the in-flight sequence is now discarded (matching the
  GL/CUDA writers), and the writer remains usable afterwards.
- `VideoWriter(gpu=True)` with uint16/float frames produced a broken encode attempt
  deep inside NVENC; it now raises a clear `ValueError` up front (h264_nvenc is 8-bit
  only).
- Codec/container mismatches (e.g. H.264 into `.webm`) and other libav-level encoder
  setup failures surfaced as raw `av` errors; they are now wrapped in
  `VideoEncodeError`, and no partial file is left behind.
- `JpegVideoWriterCUDA`: bytes that are not a decodable JPEG raise
  `ValueError('Could not parse JPEG data: ...')` instead of a confusing internal
  `NvencError`/`RuntimeError` (nvJPEG "successfully" parses some garbage as 0x0).
- `VideoWriter.append_data` with a non-array argument (e.g. a list or string) raises
  `TypeError` immediately in the caller's thread instead of killing the worker.
- libx265 printed its build-info banner to stderr on every CPU HEVC encode; it is now
  silenced (encoder errors are still shown).
- Truncated videos with interleaved audio recovered only a fraction of their
  decodable frames: the corrupt tail of the audio stream ended demuxing for the
  video stream too. Non-video streams are now discarded at the demuxer level, so
  recovery matches the ffmpeg CLI (e.g. one broken sample: 105 frames instead of
  30) — and intact multi-stream files skip the demuxer work for streams that are
  never read.
- `DepthVideoWriter` silently ignored a `gpu=` request (FFV1 has no hardware
  encoder); it now raises a clear `ValueError`.
- Float frames with values outside [0, 1] (e.g. raw depth in meters) silently
  clipped to white; writing such frames now emits a warning explaining the
  expected range.
- Palettized (pal8) video from palette-carrying codecs (QuickTime RLE/SMC,
  RSCC screen capture) decoded to a single flat color: seeking a container to 0
  before decoding wipes the decoder's palette state, and every read path did
  exactly that on freshly opened containers. Fresh containers are no longer
  seeked at all (they are already at the start), which also removes a spurious
  trailing decode error on raw VC-1 streams and a stale-position bug where an
  MPEG-PS packet index built after the seekability probe missed half the file.
- Containers that flag every packet as a keyframe (screen/animation codecs like
  Cinepak and CamStudio) silently returned wrong pixels on deep indexed access:
  the shallow seek-verification probe sampled only early frames. Both codecs are
  now content-verified, and files claiming dense keyframes get two deeper
  (budget-capped) probe positions; sparse-GOP files pay nothing extra.
- Streams where packets and frames are not 1:1 (multi-frame packets, flushed
  frames, no-op packets, duplicate timestamps) had a wrong `len()`, and in the
  worst case iterating after `len()` lost frames (an SMV file dropped 10 of 12).
  A full streaming pass now reconciles the index against the observed frame
  count, and duplicate packet timestamps trigger a decoder-verified rebuild -
  both free for well-behaved files. `len()` before any decode remains
  packet-based (exact for 1:1 streams, the overwhelming majority).
- `len()` on raw VVC bitstreams leaked raw internal `av` errors
  (`PatchWelcomeError`); all decoder-side failures during sequential frame
  counting are now tolerated like truncation, ending in typed errors.
- Interlaced 4:2:0 content decoded with progressive chroma upsampling, smearing
  color between the two fields (the scale filter's legacy default). Conversion
  is now field-aware exactly for interlaced-flagged frames - matching FFmpeg 8's
  frame-based swscale API and PyAV's `to_ndarray` - and bit-identical to before
  on all progressive content.
- Documentation: the lazy-evaluation and frame-accurate-processing pages still
  described the old eager packet-index construction (the index is built lazily on
  first use); the README's 10-bit example wrote the same gradient to two channels
  (`gradient.T` of a 1-D array is a no-op) and omitted `DepthVideoWriter` from the
  overview. Also documented that metadata (`len()`, `fps`) can succeed on damaged
  files whose frames later fail to decode; that `num_frames()` defaults to a
  duration-based estimate (use `exact=True` or `len(VideoFrames(...))`); the
  `seekable` parameter; `VideoWriter`'s `gpu`/`encoder_config` arguments;
  `GLVideoWriter`'s expected texture type; that `VideoFramesCuda` output is not
  bit-identical to CPU decoding (unlike `VideoFrames(gpu=True)`, which is); that
  `resized()` stretches without preserving aspect ratio; and that `.fps` reflects
  slicing/repetition. The GPU-acceleration page no longer claims FLV cannot be
  GPU-decoded (it can, since FFmpeg 8).

### Changed

- PyAV >= 17.1 is now required (previously unpinned, developed against 12.x):
  needed for the HWAccel decoding API and `Stream.discard`, and the bundled
  FFmpeg is now the 8.x generation.
  Adjustments for FFmpeg 8: format/resize filter graphs are built from the first
  decoded frame's actual properties instead of stream metadata; illegal 'reserved'
  color metadata emitted by some decoders is sanitized to 'unspecified' before
  conversion (FFmpeg 8's colorspace-managed swscale rejects it, e.g. some ProRes
  files); conversion errors for colorspaces swscale cannot convert (DXV3's YCgCo)
  and truncated-tail demuxer errors during frame counting are wrapped into
  `VideoDecodeError` instead of leaking raw `av` errors. The pin stays below av 18,
  which drops Python 3.10.
- Errors from `VideoWriter`'s background thread are re-raised with their original
  exception type (e.g. `ValueError` for a bad frame, `VideoEncodeError` for encoder
  failures) instead of being wrapped in a `RuntimeError`; the worker traceback is
  preserved. Out-of-range `IndexError` messages on sliced views now blame the view and
  report the source video's frame count.
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
- `DepthVideoWriter`: lossless 16-bit depth video via FFV1-encoded `gray16le` in MKV —
  bit-exact for `(height, width)` uint16 frames at roughly half the size of a PNG
  sequence. Read back with the new `VideoFrames(..., gray=True)`, which decodes any
  video to single-channel `(height, width)` frames (bit-exact for gray16le sources
  with `dtype=np.uint16`, luma conversion for color sources).
- File-like objects (anything with `read`/`seek`/`tell`, e.g. `BytesIO`) are accepted as
  video sources by `VideoFrames`. `BytesIO` sources support multiple concurrently active
  iterators; other file-like objects support one.
- `VideoWriter.append_data` accepts grayscale `(H, W)` and `(H, W, 1)` frames by
  replicating to 3 channels, symmetric with `VideoFrames(gray=True)` on the
  reading side.
- `start_sequence()` on all writers returns a context manager: on clean exit it ends
  the sequence, and if the body raised it aborts it, leaving no file at the output
  path (previously only `JpegVideoWriterCUDA` supported this).
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
