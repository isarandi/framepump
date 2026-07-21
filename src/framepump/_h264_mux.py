"""Shared H.264 passthrough muxing for the NVENC-based video writers.

The GL and CUDA JPEG writers produce pre-encoded H.264 packets (Annex-B NAL
units with display-order pts and decode-order dts, as emitted by
``NvencEncodeSession``). This module owns everything between those packets
and a finished video file: container setup, the timestamp policy,
per-container muxer options, audio interleaving, and the atomic temp-file
lifecycle.
"""

from __future__ import annotations

import itertools
import os
import warnings
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Union

import av

from ._pyav import NoAudioStreamError
from ._temp_file import TempFile

if TYPE_CHECKING:
    from .nvenc import EncodedPacket

PathLike = Union[str, Path]
VideoOutput = Union[str, Path, BinaryIO]

_MP4_FAMILY = ('mp4', 'mov', 'm4v', '3gp')

# The temp-file scheme replaces the output extension, so the container format
# must be passed to libav explicitly - and some common extensions are not
# valid libav muxer names.
_FORMAT_ALIASES = {'mkv': 'matroska', 'm4v': 'mp4'}


class H264PassthroughMuxer:
    """Muxes pre-encoded H.264 packets into a container via PyAV.

    Timestamp policy: ``pts`` is the display-order frame index, ``dts`` the
    decode-order packet index shifted down by ``bframes`` (so dts <= pts
    always holds), both at a 1/fps time base. For mp4-family containers with
    B-frames, ``negative_cts_offsets`` is used instead of an edit list so
    video is not delayed against muxed audio.

    Audio from ``audio_source_path`` is interleaved by submission order: all
    audio up to ``frames_muxed / fps`` is written before each video packet.

    File lifecycle: path outputs are written to a temp file that ``close()``
    renames to the final path only after a fully successful write. Any error
    (or ``abort()``) deletes the temp file instead - a failed write leaves no
    output. If no video packet is ever muxed, no file is created.
    """

    def __init__(
        self,
        video_output: VideoOutput,
        *,
        fps: Fraction,
        width: int,
        height: int,
        bframes: int,
        pix_fmt: str = 'yuv420p',
        audio_source_path: PathLike | None = None,
        format: str | None = None,
        stream_options: dict[str, str] | None = None,
    ) -> None:
        """Open the output container and set up the video (and audio) streams.

        Args:
            video_output: Output path (str/Path) or seekable file-like object.
            fps: Video frame rate.
            width: Coded frame width (including any macroblock padding).
            height: Coded frame height (including any macroblock padding).
            bframes: Number of B-frames the encoder was configured with.
            pix_fmt: Pixel format to declare on the video stream.
            audio_source_path: Optional file to copy the first audio stream from.
            format: Container format; inferred from the path extension if not
                given. Required for file-like outputs.
            stream_options: Extra codec-context options for the video stream
                (e.g. {'strict': 'experimental'} for 4:2:2/4:4:4 in mp4).
        """
        self._fps_frac = fps
        self._bframes = bframes
        self._frames_muxed = 0
        self._closed = False
        self._audio_input_container: av.container.InputContainer | None = None
        self._audio_stream = None
        self._audio_time_base: Fraction = Fraction(1)
        self._audio_pkts: Iterator[av.Packet] = iter([])

        if isinstance(video_output, (str, Path)):
            self._temp_file = TempFile(video_output)
            target = os.fspath(self._temp_file.temp_path)
            self._format = format or Path(video_output).suffix.lstrip('.')
        else:
            self._temp_file = None
            target = video_output
            if format is None:
                raise ValueError('format is required when writing to a file-like object')
            self._format = format

        muxer_options = {}
        if bframes > 0:
            if self._format in _MP4_FAMILY:
                muxer_options['movflags'] = 'negative_cts_offsets'
                muxer_options['use_editlist'] = '0'
            elif self._format == 'avi':
                warnings.warn(
                    'avi stores no presentation timestamps; B-frame videos may show '
                    'timing jitter in non-FFmpeg players. Consider mp4/mkv, or '
                    'EncoderConfig(bframes=0) for avi output.',
                    RuntimeWarning,
                    stacklevel=2,
                )

        muxer_name = _FORMAT_ALIASES.get(self._format, self._format)
        self._output_container = av.open(target, 'w', format=muxer_name, options=muxer_options)
        try:
            self._video_stream = self._output_container.add_stream('h264', rate=fps)
            self._video_stream.width = width
            self._video_stream.height = height
            self._video_stream.pix_fmt = pix_fmt
            if stream_options:
                self._video_stream.codec_context.options = stream_options
            if audio_source_path is not None:
                self._open_audio(audio_source_path)
        except BaseException:
            self.abort()
            raise

    @property
    def frames_muxed(self) -> int:
        """Number of video packets muxed so far."""
        return self._frames_muxed

    def mux(self, packet: EncodedPacket) -> None:
        """Mux one encoded video packet, interleaving pending audio before it."""
        if self._closed:
            raise RuntimeError('Muxer is closed, cannot mux more packets.')
        if not packet.data:
            return

        # Submit-order clock: monotonic even though packet pts arrive in
        # decode order under B-frame reordering.
        self._mux_audio_until(self._frames_muxed / self._fps_frac)

        av_packet = av.Packet(packet.data)
        av_packet.stream = self._video_stream
        av_packet.pts = packet.pts
        av_packet.dts = packet.dts - self._bframes
        av_packet.time_base = 1 / self._fps_frac
        av_packet.is_keyframe = packet.is_keyframe
        self._output_container.mux(av_packet)
        self._frames_muxed += 1

    def _mux_audio_until(self, video_time: Fraction) -> None:
        if self._audio_stream is None:
            return
        for audio_pkt in self._audio_pkts:
            if audio_pkt.dts * self._audio_time_base > video_time:
                # Put back the packet for the next round
                self._audio_pkts = itertools.chain([audio_pkt], self._audio_pkts)
                break
            audio_pkt.stream = self._audio_stream
            self._output_container.mux(audio_pkt)

    def _open_audio(self, audio_source_path: PathLike) -> None:
        self._audio_input_container = av.open(str(audio_source_path))
        if not self._audio_input_container.streams.audio:
            self._audio_input_container.close()
            self._audio_input_container = None
            raise NoAudioStreamError(audio_source_path)
        src_audio = self._audio_input_container.streams.audio[0]
        self._audio_stream = self._output_container.add_stream(template=src_audio)
        self._audio_time_base = src_audio.time_base
        self._audio_pkts = (
            pkt for pkt in self._audio_input_container.demux(src_audio) if pkt.dts is not None
        )

    def close(self) -> None:
        """Drain remaining audio, close the container, promote the temp file.

        On error, the temp file is deleted before re-raising: a failed write
        leaves no output at the final path.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._frames_muxed > 0 and self._audio_stream is not None:
                for audio_pkt in self._audio_pkts:
                    audio_pkt.stream = self._audio_stream
                    self._output_container.mux(audio_pkt)
        except BaseException:
            self._release(finalize=False)
            raise
        self._release(finalize=self._frames_muxed > 0)

    def abort(self) -> None:
        """Close everything without finalizing; the temp file is deleted."""
        if self._closed:
            return
        self._closed = True
        try:
            self._release(finalize=False)
        except Exception:
            # Best-effort teardown on an error path: the temp file has been
            # removed by _release's finally; don't mask the original error.
            pass

    def _release(self, finalize: bool) -> None:
        try:
            try:
                self._output_container.close()
            finally:
                if self._audio_input_container is not None:
                    self._audio_input_container.close()
                    self._audio_input_container = None
        except BaseException:
            finalize = False
            raise
        finally:
            if self._temp_file is not None:
                if finalize and self._temp_file.temp_path.exists():
                    self._temp_file.finalize()
                else:
                    self._temp_file.cleanup()
                self._temp_file = None
