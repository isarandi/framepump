# FramePump

[![CI](https://github.com/isarandi/framepump/actions/workflows/ci.yml/badge.svg)](https://github.com/isarandi/framepump/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/framepump.svg)](https://pypi.org/project/framepump/)
[![Python](https://img.shields.io/pypi/pyversions/framepump.svg)](https://pypi.org/project/framepump/)
[![Documentation](https://readthedocs.org/projects/framepump/badge/?version=latest)](https://framepump.readthedocs.io/)
[![License](https://img.shields.io/pypi/l/framepump.svg)](https://github.com/isarandi/framepump/blob/main/LICENSE)

A Python library for high-performance video processing, built on [PyAV](https://pyav.org) (in-process FFmpeg libraries, no subprocesses). It provides lazy, sliceable, frame-accurate video reading and threaded writing — with NVIDIA GPUs as first-class citizens: video files and even live webcams can be decoded on the GPU with the frames *staying* in GPU memory, handed to PyTorch/CuPy as zero-copy DLPack tensors, and encoded back to video straight from GPU memory (OpenGL textures or JPEG streams) with NVENC.

**Highlights:**

- **Lazy, sliceable reading** — `frames[::2][:100]` chains without decoding anything; exact frame counts and frame-accurate indexing even for variable-framerate files
- **GPU decoding** — `VideoFrames(gpu=True)` decodes on NVDEC with bit-identical output; `VideoFramesCuda` keeps frames in GPU memory, exported zero-copy via DLPack, including whole batches gathered straight from the decoder
- **Live cameras on the GPU** — `CameraFrames` decodes webcam MJPEG via NVDEC and always hands you the latest frame; `list_cameras()` discovers devices and their modes
- **GPU encoding** — `GLVideoWriter` (OpenGL texture → NVENC, zero copy) and `NvJpegVideoWriter` (JPEG bytes → nvJPEG → NVENC, fully GPU-resident)
- **Flexible sources** — local files, HTTP(S)/RTSP URLs, and file-like objects (BytesIO, archive members)
- **Colors done right** — exact YUV↔RGB matrices for all matrix-expressible colorspaces, limited/full range from stream flags, 10-bit support, gamma-correct resizing in linear light
- **Threaded writing** — non-blocking `VideoWriter` with audio carry-over (`like=` copies fps and audio from a reference), 10-bit encoding, lossless 16-bit depth video

## Installation

```bash
pip install framepump
```

If a GPU feature misbehaves, `framepump.diagnose()` prints an environment report (driver, FFmpeg build, per-feature availability with reasons) to attach to bug reports.

## Usage

### Reading Video Frames

The main entry point for reading videos is the `VideoFrames` class. It allows for efficient, slice-based access to video frames.

```python
from framepump import VideoFrames
import numpy as np

frames = VideoFrames('my_video.mp4')  # This is lazy, it only reads some metadata.
# URLs and file-like objects work too: VideoFrames('https://example.com/clip.mp4')

# Get basic information
print(f"Shape: {frames.imshape}")
print(f"FPS: {frames.fps}")
print(f"Number of frames: {len(frames)}")
```

`frames.info` gives the full overview in one readable object:

```pycon
>>> print(frames.info)
my_video.mp4
  video: h264, 1920x1080, 29.97 fps, 12.5 s, ~375 frames
  pixels: yuv420p, 8-bit, colorspace bt709, range tv
  audio: aac, 48000 Hz
```

```python
# Iterate over all frames — this is where decoding begins
for frame in frames:
    # frame is a numpy array of shape (height, width, 3) and dtype uint8
    pass

# Slice the video to get every second frame within the first 100 frames
subset_frames = frames[:100:2]
print(f"Number of frames in subset: {len(subset_frames)}")

# Resize the video on the fly (this creates a new VideoFrames instance, no frames are read yet)
resized_frames = frames.resized((128, 128))
print(f"Resized shape: {resized_frames.imshape}")

# Change the data type (e.g., to float32 for neural network processing).
# Float dtypes yield values scaled to [0, 1].
float_frames = VideoFrames('my_video.mp4', dtype=np.float32)

# Use NVDEC GPU acceleration for decoding (requires an NVIDIA GPU).
# Frames are decoded on GPU and returned as numpy arrays (on CPU),
# bit-identical to CPU decoding. Unsupported codecs raise instead of
# silently falling back to CPU.
frames = VideoFrames('my_video.mp4', gpu=True)

```

For fully GPU-resident frames (no GPU→CPU transfer), use `VideoFramesCuda`:
frames stay in GPU memory and export zero-copy to PyTorch/CuPy via DLPack.

```python
import torch
from framepump import VideoFramesCuda

cuda_frames = VideoFramesCuda('my_video.mp4')
for f in cuda_frames.resized((224, 224)):
    tensor = torch.from_dlpack(f)  # (224, 224, 3) uint8 CUDA tensor, zero-copy
    ...  # use within this step, or .clone() to keep

# Gather specific frames as ONE stacked batch tensor, straight from NVDEC:
batch = torch.from_dlpack(cuda_frames[[10, 50, 300]])  # (3, H, W, 3) on CUDA
```

### Live Cameras on the GPU

`CameraFrames` reads USB webcams (V4L2/MJPEG) and decodes every frame on the GPU via NVDEC's JPEG engine. Iteration always yields the **latest** captured frame — a consumer slower than the camera skips frames instead of processing a growing backlog, keeping latency at one frame interval.

```python
import torch
from framepump import CameraFrames, list_cameras

print(*list_cameras(), sep='\n')  # discover devices and their MJPEG modes

with CameraFrames('/dev/video0', shape=(720, 1280), fps=30) as cam:
    for frame in cam:
        tensor = torch.from_dlpack(frame)  # (720, 1280, 3) uint8 CUDA, zero-copy
        ...  # run your model; cam.last_capture_time tells you the frame's age
```

For models that only reach real-time throughput when batched, `cam.batched(n)` yields adaptive batches of the frames captured since the previous step — never the same frame twice, spread evenly across the missed interval, always ending at the newest.

### Writing Videos

You can write a sequence of frames to a video file using the `VideoWriter` class. It handles the writing process in a separate thread for better performance.

```python
import numpy as np
from framepump import VideoWriter

# Use VideoWriter as a context manager
with VideoWriter('output.mp4', fps=30) as writer:
    for i in range(100):
        # Generate a 100x100 black frame with a moving white square
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[i:i+10, i:i+10] = 255
        writer.append_data(frame)
```

#### Including Audio

You can copy the audio stream from another video file into your output video by providing the `audio_source_path` argument to the `VideoWriter`. The audio will be copied without re-encoding.

```python
import numpy as np
from framepump import VideoWriter

# Create a silent video and then mux it with audio from another file
with VideoWriter('output_with_audio.mp4', fps=30, audio_source_path='input_with_audio.mp4') as writer:
    for i in range(100):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        writer.append_data(frame)
```

For the common read-process-write round trip, `like=` copies the frame rate and the audio from a reference in one go — and it tracks slicing, so a `video[::2]` reference halves the output fps while preserving duration and audio sync:

```python
from framepump import VideoFrames, VideoWriter

video = VideoFrames('input.mp4')
with VideoWriter('annotated.mp4', like=video) as writer:
    for frame in video:
        writer.append_data(process(frame))
```

### Getting Video Information

The recommended way is `VideoFrames`: `frames.info` for the full overview (shown above), `len(frames)` for the exact frame count, `frames.fps` / `frames.imshape` for the effective properties of the (possibly sliced/resized) view. Standalone utility functions also exist for one-off lookups:

```python
from framepump import get_fps, get_duration, num_frames, video_extents

video_path = 'my_video.mp4'

fps = get_fps(video_path)
duration = get_duration(video_path)
n_frames = num_frames(video_path)  # fast estimate (duration x fps)
n_frames = num_frames(video_path, exact=True)  # exact (scans packet headers)
width, height = video_extents(video_path)

print(f"FPS: {fps}, Duration: {duration}s, Frames: {n_frames}, Dimensions: {width}x{height}")
```

### Video Manipulation

#### Trimming

Cut a portion of a video.

```python
from framepump import trim_video

trim_video('input.mp4', 'output_trimmed.mp4', start_time='00:00:10', end_time='00:00:20')
```

#### Muxing Audio and Video

Combine the video stream from one file with the audio stream from another.

```python
from framepump import video_audio_mux

video_audio_mux(
    vidpath_audiosource='video_with_audio.mp4',
    vidpath_imagesource='silent_video.mp4',
    out_video_path='output_muxed.mp4'
)
```

## Core Abstractions

### `VideoFrames`

This is the central class for reading video frames. It's a lazy, sliceable, and chainable frame sequence.

- **Lazy:** Frames are only read from the file when you iterate over them.
- **Sliceable:** You can use standard Python slicing (`[start:stop:step]`) to select a range of frames. This is also lazy and does not read the frames into memory. The resulting object is also sliceable, so you can chain slicing operations, for example `frames[::4][:10]`.
- **Chainable:** Methods like `resized()`, `repeat_each_frame()`, and slicing return a new `VideoFrames` instance, allowing you to chain operations.

### `VideoWriter`

This class handles writing frames to a video file. It uses a separate thread to encode and write the video, which prevents the main thread from blocking on I/O and improves performance. It can be used as a context manager for easy setup and teardown.

### `DepthVideoWriter`

A `VideoWriter` variant for lossless 16-bit grayscale video (e.g. depth maps in millimeters), stored as FFV1-encoded `gray16le` in MKV — bit-exact round-trips at roughly half the size of a PNG sequence. Read the result back with `VideoFrames(path, gray=True, dtype=np.uint16)`.

```python
from framepump import DepthVideoWriter

with DepthVideoWriter('depth.mkv', fps=5) as writer:
    for depth in depth_frames:  # (H, W) uint16 arrays
        writer.append_data(depth)
```

### Resampling Frame Rate

You can resample a video to a constant frame rate using the `constant_framerate` parameter. This is useful for ML pipelines that expect a fixed number of frames per second.

```python
from framepump import VideoFrames

# Resample to 10 fps (drops/duplicates frames as needed)
frames = VideoFrames('my_video.mp4', constant_framerate=10.0)
print(f"FPS: {frames.fps}, Frames: {len(frames)}")

# Ensure constant frame rate at the original fps (useful for VFR videos)
frames = VideoFrames('my_vfr_video.mp4', constant_framerate=True)
```

### Zero-Copy GPU Encoding with NVENC

For real-time rendering applications, `GLVideoWriter` encodes OpenGL textures directly to video using NVIDIA's hardware encoder (NVENC), without any CPU memory transfers.

```python
from framepump import GLVideoWriter

with GLVideoWriter('output.mp4', fps=30) as writer:
    for _ in render_loop:
        render_to_texture(texture)
        ctx.finish()  # Ensure GPU is done rendering
        writer.append_data(texture)  # Encode directly from GPU memory
```

**Key features:**
- **Zero-copy**: Pixel data never leaves the GPU
- **Hardware encoding**: Uses dedicated NVENC hardware, not CUDA cores
- **Headless support**: Works with both GLX (X11) and EGL (headless/containerized) contexts

**Requirements:**
- NVIDIA GPU with NVENC support
- Linux with NVIDIA driver
- For headless: `pip install framepump[nvenc-cuda]`

See the [NVENC documentation](https://framepump.readthedocs.io/en/latest/explanation/nvenc-zero-copy.html) for details.

Its sibling `NvJpegVideoWriter` turns JPEG byte streams (e.g. from MJPEG cameras or archives of per-frame JPEGs) into H.264 video entirely on the GPU: nvJPEG decode → NVENC encode, with no CPU-GPU pixel transfers at all.

```python
from framepump import NvJpegVideoWriter

with NvJpegVideoWriter('output.mp4', fps=30) as writer:
    for jpeg_bytes in jpeg_stream:
        writer.append_data(jpeg_bytes)
```

### High Bit Depth Support

The library supports high bit depth video (e.g., 10-bit) by using the `numpy.uint16` data type for frames.

When reading videos, you can specify `dtype=np.uint16` in the `VideoFrames` constructor. This will decode the video into 16-bit RGB frames.

```python
# Read a video as 16-bit integer frames
uint16_frames = VideoFrames('my_high_bit_depth_video.mp4', dtype=np.uint16)
```

When writing, if you provide the first frame with a `dtype` of `np.uint16`, `framepump` will automatically encode the video using a 10-bit YUV pixel format (`yuv420p10le`), which is suitable for high dynamic range (HDR) content.

```python
import numpy as np
from framepump import VideoWriter

# Use VideoWriter as a context manager
with VideoWriter('output_10bit.mp4', fps=30) as writer:
    for i in range(100):
        # Generate a 16-bit frame (e.g., a gradient)
        frame = np.zeros((100, 100, 3), dtype=np.uint16)
        gradient = np.linspace(0, 65535, 100, dtype=np.uint16)
        frame[:, :, 0] = gradient            # horizontal gradient
        frame[:, :, 1] = gradient[:, None]   # vertical gradient
        writer.append_data(frame)
```

