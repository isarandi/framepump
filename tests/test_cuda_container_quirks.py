"""VideoFramesCuda on container layouts the raw NVDEC parser cannot handle alone.

Two quirks, both invisible to the CPU class because libavcodec handles them:

- MPEG-4 Part 2 (and MPEG-1/2, VC-1) in MP4/MKV keeps its sequence headers
  in the container's ``extradata``, not in the packets; fed bare packets, the
  NVDEC parser emits nothing at all.
- Edit-list trims (``ffmpeg -ss X -i src -c copy out.mp4``) keep the frames
  from the preceding keyframe as discard-flagged, negative-PTS preroll.
  libavcodec drops them; NVDEC decodes and emits them, which shifted every
  positional access by the preroll length.

The oracle is the CPU class. Content is grayscale so the comparison is
independent of the RGB matrix each path picks for streams without color
metadata; frames differ from their neighbors by more than the tolerance,
so any positional shift is caught.

GPU detection happens inside each test (fork-safe): pytest runs with
--forked, and CUDA state created at collection time breaks in the children.
"""

import shutil
import subprocess

import numpy as np
import pytest

from framepump import VideoFrames

FFMPEG = shutil.which('ffmpeg')
pytestmark = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg CLI not available')

W, H, FPS, N_FRAMES = 128, 96, 30, 60
FRAME_STEP = 2  # gray-level shift between consecutive frames
TOLERANCE = 0.8  # mean abs RGB difference; IDCT drift is ~0.1, one frame off is ~2

ENCODERS = {
    'mpeg4': ['-c:v', 'mpeg4', '-g', '15', '-q:v', '3'],
    'mpeg2video': ['-c:v', 'mpeg2video', '-g', '15', '-q:v', '3'],
    'h264': ['-c:v', 'libx264', '-g', '15', '-bf', '2', '-crf', '18'],
}


def _gpu():
    """Import torch + the CUDA reader, skipping when NVDEC is unavailable."""
    import ctypes

    try:
        ctypes.CDLL('libnvcuvid.so.1')
    except OSError:
        pytest.skip('NVDEC (libnvcuvid) not available')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    pytest.importorskip('PyNvVideoCodec')
    from framepump import VideoFramesCuda

    return torch, VideoFramesCuda


def _gray_frames() -> np.ndarray:
    """A smooth gradient, shifted brighter by FRAME_STEP levels per frame."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = 16 + xx * 60 // (W - 1) + yy * 24 // (H - 1)
    return np.stack([np.clip(base + FRAME_STEP * i, 0, 235) for i in range(N_FRAMES)]).astype(
        np.uint8
    )


def _encode(out_path, codec: str, container_ext: str) -> str:
    path = str(out_path / f'src_{codec}{container_ext}')
    cmd = [
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        *ENCODERS[codec], '-pix_fmt', 'yuv420p',
        path,
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=_gray_frames().tobytes(), capture_output=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(
            f'ffmpeg encode failed (encoder not available?): {proc.stderr.decode()[-300:]}'
        )
    return path


def _trim_with_edit_list(src: str, seconds: float) -> str:
    """Stream-copy from ``seconds`` on: the MP4 muxer writes an edit list and
    the demuxer marks the frames between the preceding keyframe and the cut
    as discard-flagged preroll with negative PTS."""
    out = src.replace('src_', 'trim_')
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(seconds), '-i', src,
           '-c', 'copy', out]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return out


def _packet_layout(path: str) -> dict:
    """Facts about the container that the tests rely on as preconditions."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        extradata = bytes(stream.codec_context.extradata or b'')
        packets = [p for p in container.demux(stream) if p.size > 0]
        first = bytes(packets[0])
        return {
            'n_packets': len(packets),
            'n_preroll': sum(1 for p in packets if p.is_discard),
            'n_negative_pts': sum(
                1 for p in packets if (p.pts if p.pts is not None else p.dts) < 0
            ),
            'headers_in_band': bool(extradata) and first.startswith(extradata),
            'has_extradata': bool(extradata),
        }


def _collect(torch, frames) -> list:
    """Download GPU frames as they arrive (yielded frames are only valid
    until the next iteration step)."""
    return [torch.from_dlpack(f).cpu().numpy() for f in frames]


def _assert_frames_match(cpu_frames, gpu_frames, what: str) -> None:
    assert len(cpu_frames) == len(
        gpu_frames
    ), f'{what}: CPU yielded {len(cpu_frames)} frames, CUDA {len(gpu_frames)}'
    for i, (a, b) in enumerate(zip(cpu_frames, gpu_frames)):
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
        assert diff < TOLERANCE, f'{what}: frame {i} differs from CPU by {diff:.2f} (mean abs)'


class TestOutOfBandSequenceHeaders:
    """MPEG-4/MPEG-2 streams whose headers live only in extradata (MP4/MKV),
    plus AVI as the in-band control (the filter must not disturb it)."""

    @pytest.mark.parametrize(
        'codec, ext',
        [('mpeg4', '.mp4'), ('mpeg4', '.mkv'), ('mpeg4', '.avi'), ('mpeg2video', '.mkv')],
    )
    def test_full_decode_matches_cpu(self, tmp_path, codec, ext):
        torch, VideoFramesCuda = _gpu()
        path = _encode(tmp_path, codec, ext)
        layout = _packet_layout(path)
        assert layout['has_extradata'], 'test premise: the container carries extradata'
        if ext == '.avi':
            assert layout['headers_in_band'], 'test premise: AVI keeps headers in the packets'

        cpu = VideoFrames(path)
        gpu = VideoFramesCuda(path)
        # Streamed pass first: forward iteration before any length query must
        # not need the index and must still see decodable packets
        streamed = _collect(torch, gpu)
        cpu_frames = [f.copy() for f in cpu]
        _assert_frames_match(cpu_frames, streamed, f'{codec}{ext} streamed')
        assert len(streamed) == N_FRAMES == len(cpu) == len(gpu)
        # Index-based paths: sequential, seek, per-frame, reverse
        _assert_frames_match(cpu_frames, _collect(torch, gpu), f'{codec}{ext} sequential')
        for i in (0, 7, 31, N_FRAMES - 1):
            _assert_frames_match([cpu[i]], _collect(torch, [gpu[i]]), f'{codec}{ext} index {i}')
        _assert_frames_match(
            list(cpu[20:50:3]), _collect(torch, gpu[20:50:3]), f'{codec}{ext} slice'
        )
        _assert_frames_match(
            list(cpu[40:10:-4]), _collect(torch, gpu[40:10:-4]), f'{codec}{ext} reverse'
        )


class TestEditListPreroll:
    """Cut MP4s: CPU and CUDA readers must agree on length, order and content."""

    @pytest.mark.parametrize('codec', ['mpeg4', 'h264'])
    def test_trimmed_file_matches_cpu(self, tmp_path, codec):
        torch, VideoFramesCuda = _gpu()
        # Cut at 0.7 s: keyframes every 15 frames, so 6 preroll frames (15..20)
        path = _trim_with_edit_list(_encode(tmp_path, codec, '.mp4'), 0.7)
        layout = _packet_layout(path)
        if not layout['n_preroll']:
            pytest.skip('this ffmpeg did not produce an edit-list preroll with -ss -c copy')
        assert (
            layout['n_negative_pts'] == layout['n_preroll']
        ), 'test premise: preroll packets are exactly the negative-PTS ones'
        n_kept = layout['n_packets'] - layout['n_preroll']

        cpu = VideoFrames(path)
        gpu = VideoFramesCuda(path)
        cpu_frames = [f.copy() for f in cpu]
        assert len(cpu_frames) == n_kept, 'oracle check: CPU class drops the preroll'

        # Streamed (index-free) forward pass, then the index-based paths
        _assert_frames_match(cpu_frames, _collect(torch, gpu), f'{codec} trimmed streamed')
        assert len(gpu) == len(cpu) == n_kept
        _assert_frames_match(cpu_frames, _collect(torch, gpu), f'{codec} trimmed sequential')
        for i in (0, 1, 5, 6, n_kept - 1):
            _assert_frames_match([cpu[i]], _collect(torch, [gpu[i]]), f'{codec} trimmed index {i}')
        _assert_frames_match(
            list(cpu[3:30:2]), _collect(torch, gpu[3:30:2]), f'{codec} trimmed slice'
        )
        _assert_frames_match(
            list(cpu[25:2:-3]), _collect(torch, gpu[25:2:-3]), f'{codec} trimmed reverse'
        )
        _assert_frames_match(
            list(cpu[::40]), _collect(torch, gpu[::40]), f'{codec} trimmed per-frame seek'
        )

    def test_trimmed_batched_and_cfr_paths(self, tmp_path):
        """batched() and the CFR source map both go through PTS-keyed paths."""
        torch, VideoFramesCuda = _gpu()
        path = _trim_with_edit_list(_encode(tmp_path, 'h264', '.mp4'), 0.7)
        if not _packet_layout(path)['n_preroll']:
            pytest.skip('this ffmpeg did not produce an edit-list preroll with -ss -c copy')

        cpu_frames = [f.copy() for f in VideoFrames(path)]
        gpu = VideoFramesCuda(path)
        stacked = np.concatenate(_collect(torch, gpu.batched(8)))
        _assert_frames_match(cpu_frames, list(stacked), 'batched')

        cfr_cpu = [f.copy() for f in VideoFrames(path, constant_framerate=True)]
        cfr_gpu = _collect(torch, VideoFramesCuda(path, constant_framerate=True))
        _assert_frames_match(cfr_cpu, cfr_gpu, 'cfr')
