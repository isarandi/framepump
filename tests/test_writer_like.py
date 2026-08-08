"""VideoWriter(like=...): copying fps/audio settings from a reference video."""

import os

import numpy as np
import pytest

import framepump

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AUDIO_SRC = os.path.join(DATA_DIR, 'with_audio.mp4')


def _write_all(out, frames, **kwargs):
    with framepump.VideoWriter(out, **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)


class TestLike:
    def test_copies_fps_and_audio_from_instance(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        out = str(tmp_path / 'out.mp4')
        _write_all(out, v, like=v)
        info = framepump.VideoFrames(out).info
        assert info.fps == pytest.approx(v.fps)
        assert info.has_audio

    def test_copies_from_path(self, tmp_path):
        out = str(tmp_path / 'out.mp4')
        _write_all(out, framepump.VideoFrames(AUDIO_SRC), like=AUDIO_SRC)
        info = framepump.VideoFrames(out).info
        assert info.fps == pytest.approx(framepump.get_fps(AUDIO_SRC))
        assert info.has_audio

    def test_sliced_reference_preserves_duration(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        out = str(tmp_path / 'half.mp4')
        _write_all(out, v[::2], like=v[::2])
        info = framepump.VideoFrames(out).info
        assert info.fps == pytest.approx(v.fps / 2)
        assert info.duration == pytest.approx(v.info.duration, abs=0.1)

    def test_audio_opt_out(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        out = str(tmp_path / 'silent.mp4')
        _write_all(out, v, like=v, audio_source_path=False)
        assert not framepump.VideoFrames(out).info.has_audio

    def test_explicit_fps_wins(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        out = str(tmp_path / 'retimed.mp4')
        _write_all(out, v, like=v, fps=10)
        assert framepump.VideoFrames(out).info.fps == pytest.approx(10)

    def test_no_audio_reference(self, tmp_path):
        src = os.path.join(DATA_DIR, 'exact_30fps.mp4')
        v = framepump.VideoFrames(src)
        out = str(tmp_path / 'out.mp4')
        _write_all(out, v[:5], like=v)
        info = framepump.VideoFrames(out).info
        assert info.fps == pytest.approx(30)
        assert not info.has_audio

    def test_start_sequence_accepts_like(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        writer = framepump.VideoWriter()
        out = str(tmp_path / 'seq.mp4')
        with writer:
            with writer.start_sequence(out, like=v):
                for frame in v[:5]:
                    writer.append_data(frame)
        info = framepump.VideoFrames(out).info
        assert info.fps == pytest.approx(v.fps)
        assert info.has_audio

    def test_frames_written_correctly(self, tmp_path):
        v = framepump.VideoFrames(AUDIO_SRC)
        out = str(tmp_path / 'out.mp4')
        _write_all(out, v, like=v)
        first_in = v[0].astype(np.float32)
        first_out = framepump.VideoFrames(out)[0].astype(np.float32)
        assert np.abs(first_in - first_out).mean() < 5  # lossy encode, same image
