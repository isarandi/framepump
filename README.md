# FramePump

[![CI](https://github.com/isarandi/framepump/actions/workflows/ci.yml/badge.svg)](https://github.com/isarandi/framepump/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/framepump.svg)](https://pypi.org/project/framepump/)
[![Python](https://img.shields.io/pypi/pyversions/framepump.svg)](https://pypi.org/project/framepump/)
[![Documentation](https://readthedocs.org/projects/framepump/badge/?version=latest)](https://framepump.readthedocs.io/)
[![License](https://img.shields.io/pypi/l/framepump.svg)](https://github.com/isarandi/framepump/blob/main/LICENSE)

A Python library for high-performance video processing, built on [PyAV](https://pyav.org) (Python bindings for FFmpeg's libraries). It provides a simple and efficient way to read, write, and manipulate video files.

## Installation

```bash
pip install framepump
```

## Usage

### Reading Video Frames

The main entry point for reading videos is the `VideoFrames` class. It allows for efficient, slice-based access to video frames.

```python
from framepump import VideoFrames
import numpy as np

frames = VideoFrames('my_video.mp4')  # This is lazy, it only reads some metadata.

# Get basic information
print(f"Shape: {frames.imshape}")
print(f"FPS: {frames.fps}")
print(f"Number of frames: {len(frames)}")

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

# Change the data type (e.g., to float32 for neural network processing)
float_frames = VideoFrames('my_video.mp4', dtype=np.float32)

# Use GPU acceleration for decoding (requires a CUDA-enabled ffmpeg build and a GPU).
# Frames are decoded on GPU and returned as numpy arrays (on CPU).
frames = VideoFrames('my_video.mp4', gpu=True)

# For fully GPU-resident frames (no GPU→CPU transfer), use VideoFramesCuda instead.
# This returns DecodedFrame objects that stay in GPU memory — useful when feeding
# directly into GPU pipelines (e.g. NVENC encoding, CUDA processing).
# from framepump import VideoFramesCuda
# cuda_frames = VideoFramesCuda('my_video.mp4')
```

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

### Getting Video Information

Several utility functions are available to get information about a video file.

```python
from framepump import get_fps, get_duration, num_frames, video_extents

video_path = 'my_video.mp4'

fps = get_fps(video_path)
duration = get_duration(video_path)
n_frames = num_frames(video_path)
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
        frame[:, :, 0] = gradient
        frame[:, :, 1] = gradient.T
        writer.append_data(frame)
```

