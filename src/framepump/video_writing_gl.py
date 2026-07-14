"""GL texture to video writing using NVENC with PyAV muxing.

NVENC encoding is done directly via the nvenc module's ctypes bindings.
PyAV handles only the container muxing (no subprocess), through the shared
H264PassthroughMuxer.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import simplepyutils as spu

from ._h264_mux import H264PassthroughMuxer
from .encoder_config import EncoderConfig
from .video_writing import AbstractVideoWriter, VideoOutput

if TYPE_CHECKING:
    import moderngl

    from .nvenc import NvencCudaEncoder as NvencCudaEncoderType
    from .nvenc import NvencEncoder as NvencEncoderType

try:
    from .nvenc import NvencEncoder
except ImportError:
    NvencEncoder = None

try:
    from .nvenc import NvencCudaEncoder
except ImportError:
    NvencCudaEncoder = None

PathLike = Union[str, Path]


class GLVideoWriter(AbstractVideoWriter['moderngl.Texture'], AbstractContextManager['GLVideoWriter']):
    """Zero-copy GL texture to video writer using NVENC with PyAV muxing.

    Similar API to VideoWriter but runs synchronously (no background thread)
    because NVENC requires the OpenGL context to be current.

    Ending a sequence to which no frame was written is a no-op: no output
    file is created.

    Example:
        >>> with GLVideoWriter() as writer:
        ...     writer.start_sequence('output.mp4', fps=30)
        ...     for frame in render_loop:
        ...         ctx.finish()
        ...         writer.append_data(texture)
        ...     writer.end_sequence()
    """

    def __init__(
        self,
        video_path: PathLike | None = None,
        fps: float | None = None,
        audio_source_path: PathLike | None = None,
        queue_size: int = 32,
        encoder_config: EncoderConfig | None = None,
    ) -> None:
        # queue_size is unused; present for API compatibility with VideoWriter
        del queue_size
        self._writer: GLSequenceWriter | None = None
        self._accepts_new_frames: bool = False
        self._default_fps = fps
        self._default_encoder_config = encoder_config

        if video_path is not None:
            if fps is None:
                raise ValueError('fps must be provided if video_path is provided')
            self.start_sequence(video_path, fps, audio_source_path=audio_source_path,
                                encoder_config=encoder_config)

    @property
    def accepts_new_frames(self) -> bool:
        """Whether new frames are accepted for writing."""
        return self._accepts_new_frames

    def start_sequence(
        self,
        video_output: VideoOutput,
        fps: float | None = None,
        audio_source_path: PathLike | None = None,
        gpu: bool | int = True,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
    ) -> None:
        """Start a new video sequence.

        Args:
            video_output: Output path (str/Path) or file-like object (BinaryIO).
            fps: Frame rate for the video. Falls back to the value passed to the
                constructor if not provided here.
            audio_source_path: Optional path to copy audio from.
            gpu: GPU device ordinal for NVENC encoding. Passed to NvencCudaEncoder
                when using CUDA path (headless). Ignored for GLX path (device is
                determined by the GL context). Always truthy for GL writer.
            encoder_config: Encoder configuration (crf, gop, bframes).
            format: Container format (e.g., 'mp4'). Required for file-like objects.
        """
        if self._writer is not None:
            self._writer.close()

        if fps is None:
            if self._default_fps is None:
                raise ValueError('fps must be provided if not set in constructor')
            fps = self._default_fps

        if encoder_config is None:
            encoder_config = self._default_encoder_config

        if isinstance(video_output, (str, Path)):
            spu.ensure_parent_dir_exists(video_output)
        self._writer = GLSequenceWriter(
            video_output,
            fps=fps,
            audio_source_path=audio_source_path,
            encoder_config=encoder_config,
            format=format,
            gpu=gpu,
        )
        self._accepts_new_frames = True

    def append_data(self, data: moderngl.Texture) -> None:
        """Append a GL texture to the current video sequence.

        Args:
            data: GL texture to encode.
        """
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before appending data')
        assert self._writer is not None
        self._writer.write_frame(data)

    def end_sequence(self) -> None:
        """End the current video sequence."""
        if not self._accepts_new_frames:
            raise ValueError('start_sequence has to be called before ending the sequence')
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._accepts_new_frames = False

    def close(self) -> None:
        """Close the writer and release resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._accepts_new_frames = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            if self._writer is not None:
                self._writer._abort()
                self._writer = None
            self._accepts_new_frames = False


class GLSequenceWriter(AbstractContextManager['GLSequenceWriter']):
    """Writes a single video sequence from GL textures using NVENC with PyAV muxing.

    If no frame is written, closing is a no-op and no output file is created.

    Usage:
        with GLSequenceWriter(path, fps=30) as writer:
            for texture in textures:
                writer.write_frame(texture)

        # Or write to a file-like object:
        buffer = io.BytesIO()
        with GLSequenceWriter(buffer, fps=30, format='mp4') as writer:
            for texture in textures:
                writer.write_frame(texture)
        video_bytes = buffer.getvalue()
    """

    def __init__(
        self,
        video_output: VideoOutput,
        fps: float | Fraction,
        audio_source_path: PathLike | None = None,
        encoder_config: EncoderConfig | None = None,
        format: str | None = None,
        gpu: bool | int = True,
    ) -> None:
        self._fps_frac = (
            fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        )
        self._audio_source_path = audio_source_path
        self._encoder_config = encoder_config if encoder_config is not None else EncoderConfig()
        self._gpu = gpu
        self._video_output = video_output
        self._format = format

        if not isinstance(video_output, (str, Path)) and format is None:
            raise ValueError('format is required when writing to a file-like object')

        # Encoder and muxer are created on the first frame
        self._encoder: NvencEncoderType | NvencCudaEncoderType | None = None
        self._muxer: H264PassthroughMuxer | None = None
        self._closed: bool = False

    def write_frame(self, texture: moderngl.Texture) -> None:
        """Write a GL texture to the video."""
        if self._closed:
            raise RuntimeError('Writer is closed, cannot write more frames.')

        if self._muxer is None:
            self._open(texture)

        for encoded in self._encoder.encode(texture):
            self._muxer.mux(encoded)

    def _open(self, first_texture: moderngl.Texture) -> None:
        """Create the encoder and muxer based on the first texture."""
        if hasattr(first_texture, 'size'):
            width, height = first_texture.size
        elif hasattr(first_texture, 'width') and hasattr(first_texture, 'height'):
            width, height = first_texture.width, first_texture.height
        else:
            raise ValueError(
                'Cannot determine texture size. Pass a moderngl.Texture '
                'or an object with size/width/height attributes.'
            )

        # Create the NVENC encoder first - this is the most likely step to fail
        # (e.g., GL context on wrong GPU). Fail before creating files on disk.
        encoder_kwargs = dict(
            fps=self._fps_frac,
            crf=self._encoder_config.crf,
            gop=self._encoder_config.gop,
            bframes=self._encoder_config.bframes,
        )
        if _is_headless():
            if NvencCudaEncoder is None:
                raise ImportError(
                    'Headless mode requires NvencCudaEncoder. '
                    'Install cuda-python: pip install cuda-python'
                )
            # type() rather than isinstance(): True must not count as ordinal 1
            gpu_device = self._gpu if type(self._gpu) is int else None  # noqa: E721
            self._encoder = NvencCudaEncoder(
                width, height, **encoder_kwargs, gpu=gpu_device)
        else:
            if NvencEncoder is None:
                raise ImportError(
                    'NVENC is not available. Ensure you have an NVIDIA GPU '
                    'with NVENC support and the NVIDIA driver installed.'
                )
            self._encoder = NvencEncoder(width, height, **encoder_kwargs)

        try:
            self._muxer = H264PassthroughMuxer(
                self._video_output,
                fps=self._fps_frac,
                width=width,
                height=height,
                bframes=self._encoder_config.bframes,
                audio_source_path=self._audio_source_path,
                format=self._format,
            )
        except BaseException:
            self._encoder.close()
            self._encoder = None
            raise

    def close(self) -> None:
        """Flush the encoder and finalize the output file.

        On error, the output is discarded (no file at the final path) and the
        error propagates.
        """
        if self._closed:
            return
        self._closed = True

        if self._muxer is None:
            # No frame was ever written; nothing was opened, no file appears.
            return

        try:
            for encoded in self._encoder.flush():
                self._muxer.mux(encoded)
            self._muxer.close()
        except BaseException:
            self._muxer.abort()
            raise
        finally:
            self._encoder.close()
            self._encoder = None
            self._muxer = None

    def _abort(self) -> None:
        """Abort the write: discard output, delete the temp file."""
        if self._closed:
            return
        self._closed = True

        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        if self._muxer is not None:
            self._muxer.abort()
            self._muxer = None

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()


def _is_headless() -> bool:
    """Check if running headless (no X11 display)."""
    return not os.environ.get('DISPLAY')
