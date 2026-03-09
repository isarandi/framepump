"""GL texture to video writing using NVENC with PyAV muxing.

NVENC encoding is done directly via the nvenc module's ctypes bindings.
PyAV handles only the container muxing (no subprocess).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import av
import simplepyutils as spu

from ._temp_file import TempFile
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

        # Determine output mode: file-like or path
        if isinstance(video_output, (str, Path)):
            self._temp_file = TempFile(video_output)
            self._file_output = None
            self._format = format or Path(video_output).suffix.lstrip('.')
        else:
            # File-like object
            self._temp_file = None
            self._file_output = video_output
            if format is None:
                raise ValueError("format is required when writing to a file-like object")
            self._format = format

        # State will be initialized on first frame
        self._output_container: av.container.OutputContainer | None = None
        self._audio_input_container: av.container.InputContainer | None = None
        self._video_stream: av.stream.Stream | None = None
        self._audio_stream: av.stream.Stream | None = None
        self._audio_time_base: Fraction = Fraction(1)
        self._audio_pkts: Iterator[av.Packet] = iter([])
        self._encoder: NvencEncoderType | NvencCudaEncoderType | None = None
        self._h264_probe: av.container.InputContainer | None = None
        self._closed: bool = False
        self._bframes: int = 0
        self._width: int = 0
        self._height: int = 0

    def write_frame(self, texture: moderngl.Texture) -> None:
        """Write a GL texture to the video."""
        if self._closed:
            raise RuntimeError('Writer is closed, cannot write more frames.')

        if self._output_container is None:
            self._open(texture)

        # Encode texture to H.264 - returns list of EncodedPackets
        for encoded in self._encoder.encode(texture):
            self._mux_encoded(encoded)

    def _mux_encoded(self, encoded):
        """Mux an encoded packet into the output container."""
        video_time = encoded.pts / self._fps_frac

        # Interleave: write audio packets up to current video time
        if self._audio_stream is not None:
            for audio_pkt in self._audio_pkts:
                if audio_pkt.dts * self._audio_time_base > video_time:
                    self._audio_pkts = itertools.chain([audio_pkt], self._audio_pkts)
                    break
                audio_pkt.stream = self._audio_stream
                self._output_container.mux(audio_pkt)

        packet = av.Packet(encoded.data)
        packet.stream = self._video_stream
        packet.pts = encoded.pts
        packet.dts = encoded.dts - self._bframes
        packet.time_base = 1 / self._fps_frac
        packet.is_keyframe = encoded.is_keyframe
        self._output_container.mux(packet)

    def _open(self, first_texture: moderngl.Texture) -> None:
        """Open containers and create encoder based on first texture."""
        # Get dimensions
        if hasattr(first_texture, 'size'):
            self._width, self._height = first_texture.size
        elif hasattr(first_texture, 'width') and hasattr(first_texture, 'height'):
            self._width, self._height = first_texture.width, first_texture.height
        else:
            raise ValueError(
                'Cannot determine texture size. Pass a moderngl.Texture '
                'or an object with size/width/height attributes.'
            )

        self._bframes = self._encoder_config.bframes

        # Create NVENC encoder first — this is the most likely step to fail
        # (e.g., GL context on wrong GPU). Fail before creating files on disk.
        encoder_kwargs = dict(
            fps=self._fps_frac,
            crf=self._encoder_config.crf,
            gop=self._encoder_config.gop,
            bframes=self._bframes,
        )
        if _is_headless():
            if NvencCudaEncoder is None:
                raise ImportError(
                    'Headless mode requires NvencCudaEncoder. '
                    'Install cuda-python: pip install cuda-python'
                )
            gpu_device = self._gpu if type(self._gpu) is int else None
            self._encoder = NvencCudaEncoder(
                self._width, self._height, **encoder_kwargs, gpu=gpu_device)
        else:
            if NvencEncoder is None:
                raise ImportError(
                    'NVENC is not available. Ensure you have an NVIDIA GPU '
                    'with NVENC support and the NVIDIA driver installed.'
                )
            self._encoder = NvencEncoder(self._width, self._height, **encoder_kwargs)

        # Open output container with B-frame-aware muxer options
        muxer_options = {}
        if self._bframes > 0 and self._format in ('mp4', 'mov', 'm4v', '3gp'):
            muxer_options['movflags'] = 'negative_cts_offsets'
            muxer_options['use_editlist'] = '0'

        if self._temp_file is not None:
            self._output_container = av.open(
                os.fspath(self._temp_file.temp_path), 'w', format=self._format,
                options=muxer_options)
        else:
            self._output_container = av.open(
                self._file_output, 'w', format=self._format, options=muxer_options)

        # Create video stream for H.264 passthrough muxing.
        # The mov muxer extracts SPS/PPS from the first muxed IDR packet
        # to build the avcC box, so no extradata setup is needed here.
        self._video_stream = self._output_container.add_stream('h264', rate=self._fps_frac)
        self._video_stream.width = self._width
        self._video_stream.height = self._height
        self._video_stream.pix_fmt = 'yuv420p'

        # Set up audio if provided
        if self._audio_source_path is not None:
            self._audio_input_container = av.open(str(self._audio_source_path))
            if self._audio_input_container.streams.audio:
                src_audio = self._audio_input_container.streams.audio[0]
                self._audio_stream = self._output_container.add_stream(template=src_audio)
                self._audio_time_base = src_audio.time_base
                self._audio_pkts = (
                    pkt for pkt in self._audio_input_container.demux(src_audio)
                    if pkt.dts is not None
                )

    def close(self) -> None:
        """Flush encoder and close containers, then rename temp to final."""
        if self._closed:
            return
        self._closed = True

        if self._output_container is None:
            return

        try:
            # Flush encoder and mux remaining video packets
            if self._encoder is not None:
                for encoded in self._encoder.flush():
                    self._mux_encoded(encoded)

            # Flush remaining audio packets
            if self._audio_stream is not None:
                for audio_pkt in self._audio_pkts:
                    audio_pkt.stream = self._audio_stream
                    self._output_container.mux(audio_pkt)
        finally:
            if self._encoder is not None:
                self._encoder.close()
            self._output_container.close()
            if self._audio_input_container is not None:
                self._audio_input_container.close()
            if self._h264_probe is not None:
                self._h264_probe.close()

        if self._temp_file is not None:
            if self._temp_file.temp_path.exists():
                self._temp_file.finalize()
            else:
                # Encoder failed before writing any data
                self._temp_file.cleanup()

    def _abort(self) -> None:
        """Abort write - close resources without flushing, delete temp file."""
        if self._closed:
            return
        self._closed = True

        if self._encoder is not None:
            self._encoder.close()
        if self._output_container is not None:
            self._output_container.close()
        if self._audio_input_container is not None:
            self._audio_input_container.close()
        if self._h264_probe is not None:
            self._h264_probe.close()

        if self._temp_file is not None:
            self._temp_file.cleanup()

    def __exit__(self, exc_type: type[BaseException] | None, *args: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()


def _is_headless() -> bool:
    """Check if running headless (no X11 display)."""
    return not os.environ.get('DISPLAY')


