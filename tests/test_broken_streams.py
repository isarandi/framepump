"""Behavior on damaged, exotic and misindexed real-world streams.

Fixtures:
- ``unreliable_seek.ts``: MPEG-2 in MPEG-TS where the packet index counts
  frames the decoder never produces and keyframe-based access returns wrong
  pixels; FramePump must detect this at open time and fall back to
  sequential-only access with a decoder-accurate index.
- ``no_decodable_frames.mov``: SVQ3 file whose single indexed frame cannot be
  decoded; iteration must raise instead of silently yielding nothing.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from framepump import (
    FramePumpError,
    NoVideoStreamError,
    UnsupportedCodecError,
    VideoDecodeError,
    VideoFrames,
)

DATA_DIR = Path(__file__).parent / 'data'


def _ffmpeg_or_skip():
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg CLI not available')


def _make(tmp_path, name, *args):
    _ffmpeg_or_skip()
    out = tmp_path / name
    subprocess.run(['ffmpeg', '-y', '-v', 'error', *args, str(out)], check=True)
    return str(out)


class TestUnreliableSeekFallback:
    def test_len_matches_iteration(self):
        vf = VideoFrames(str(DATA_DIR / 'unreliable_seek.ts'))
        assert len(vf) == sum(1 for _ in vf)

    def test_indexing_matches_iteration(self):
        vf = VideoFrames(str(DATA_DIR / 'unreliable_seek.ts'))
        seq = list(vf)
        for i in (0, 1, 2, len(seq) // 2, len(seq) - 1):
            assert np.array_equal(vf[i], seq[i]), f'frame {i}'
        assert np.array_equal(vf[-1], seq[-1])

    def test_slicing_matches_iteration(self):
        vf = VideoFrames(str(DATA_DIR / 'unreliable_seek.ts'))
        seq = list(vf)
        for a, b in zip(vf[1:4], seq[1:4]):
            assert np.array_equal(a, b)
        for a, b in zip(vf[::2], seq[::2]):
            assert np.array_equal(a, b)


class TestSeekVerificationKeepsGoodFiles:
    """Suspect codecs that pass the open-time probe must keep fast seeking."""

    @pytest.mark.parametrize(
        'name, codec_args',
        [
            ('good_mpeg2.mp4', ['-c:v', 'mpeg2video', '-g', '12']),
            ('good_mpeg4.mp4', ['-c:v', 'mpeg4', '-g', '12']),
        ],
    )
    def test_seek_kept_and_correct(self, tmp_path, name, codec_args):
        path = _make(
            tmp_path,
            name,
            '-f',
            'lavfi',
            '-i',
            'testsrc=duration=2:size=128x96:rate=25',
            *codec_args,
        )
        vf = VideoFrames(path)
        len(vf)  # build the index, which runs the seek-reliability probe
        assert not vf._lazy.seek_disabled, 'probe must not disable seeking for a consistent file'
        seq = list(vf)
        for i in (1, 10, len(seq) - 1):
            assert np.array_equal(vf[i], seq[i]), f'frame {i}'


class TestNoDecodableFrames:
    def test_iteration_raises_instead_of_silent_empty(self):
        vf = VideoFrames(str(DATA_DIR / 'no_decodable_frames.mov'))
        assert len(vf) > 0
        with pytest.raises(VideoDecodeError, match='no frames'):
            list(vf)


class TestStreamErrors:
    def test_audio_only_raises_catchable_error(self, tmp_path):
        path = _make(
            tmp_path, 'audio_only.mp4', '-f', 'lavfi', '-i', 'sine=duration=1', '-c:a', 'aac'
        )
        with pytest.raises(NoVideoStreamError):
            VideoFrames(path)
        assert issubclass(NoVideoStreamError, FramePumpError)
        assert issubclass(UnsupportedCodecError, FramePumpError)


class TestRuntimeDisorderTrigger:
    """With the suspect list disabled, runtime detection alone must deliver
    correct frames: the seek path self-detects PTS disorder while decoding
    (at no cost to well-behaved files) and degrades to decode-from-start."""

    @pytest.fixture(autouse=True)
    def _no_suspect_list(self, monkeypatch):
        from framepump import _core

        monkeypatch.setattr(_core, '_SEEK_UNRELIABLE_CODECS', frozenset())

    def test_indexed_first_access_is_correct(self):
        path = str(DATA_DIR / 'unreliable_seek.ts')
        vf = VideoFrames(path)
        frame2 = vf[2].copy()  # indexed access before any sequential iteration
        seq = [f.copy() for f in VideoFrames(path)]
        assert np.array_equal(frame2, seq[2])
        assert vf._lazy.seek_disabled, 'runtime trigger must have degraded this file'

    def test_slice_after_trigger_is_correct(self):
        path = str(DATA_DIR / 'unreliable_seek.ts')
        vf = VideoFrames(path)
        seq = [f.copy() for f in VideoFrames(path)]
        got = [f.copy() for f in vf[3:6]]
        assert len(got) == 3
        for a, b in zip(got, seq[3:6]):
            assert np.array_equal(a, b)

    def test_out_of_range_after_trigger_raises_clean_index_error(self):
        path = str(DATA_DIR / 'unreliable_seek.ts')
        vf = VideoFrames(path)
        stale_len = len(vf)  # packet-index length, resolved before the trigger
        vf[2]  # triggers degradation; the rebuilt index has fewer frames
        with pytest.raises(IndexError, match='out of range'):
            vf[stale_len - 1]


class TestTruncatedInterleavedFile:
    """A truncated mp4 with interleaved audio: the corrupt tail of the audio
    stream must not end video demuxing early (non-video streams are discarded
    at the demuxer level, matching what the ffmpeg CLI recovers)."""

    @pytest.fixture
    def truncated(self, tmp_path):

        # faststart puts the index at the front so the truncated file stays openable
        path = _make(
            tmp_path,
            'full.mp4',
            '-f',
            'lavfi',
            '-i',
            'testsrc2=duration=2:size=128x96:rate=25',
            '-f',
            'lavfi',
            '-i',
            'sine=frequency=440:duration=2',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'aac',
            '-movflags',
            '+faststart',
            '-shortest',
        )
        data = Path(path).read_bytes()
        cut = tmp_path / 'cut.mp4'
        cut.write_bytes(data[: int(len(data) * 0.7)])
        return str(cut)

    def test_recovers_majority_of_frames(self, truncated):
        vf = VideoFrames(truncated)
        n = 0
        try:
            for _ in vf:
                n += 1
        except VideoDecodeError:
            pass  # a loud error after delivering the decodable frames is fine
        # Without demuxer-level discard of the audio stream, the corrupt audio
        # tail ended demuxing after the first GOP (a small fraction of frames)
        assert n >= 25, f'only {n} of 50 frames recovered'

    def test_indexed_access_works(self, truncated):
        vf = VideoFrames(truncated)
        n = len(vf)
        assert n >= 25
        frame = vf[20]
        assert frame.shape == (96, 128, 3)
