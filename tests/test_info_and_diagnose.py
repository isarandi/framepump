"""video.info, list_cameras and diagnose()."""

import os

import framepump

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestVideoInfo:
    def test_basic_fields(self):
        info = framepump.VideoFrames(os.path.join(DATA_DIR, 'exact_30fps.mp4')).info
        assert info.codec == 'h264'
        assert info.imshape == (720, 1280)
        assert info.fps == 30.0
        assert info.pix_fmt == 'yuv420p'
        assert info.bit_depth == 8
        assert info.colorspace == 'bt709'
        assert info.color_range == 'tv'
        assert not info.has_audio
        assert info.audio_codec is None

    def test_high_bit_depth(self):
        info = framepump.VideoFrames(os.path.join(DATA_DIR, '10bit.mp4')).info
        assert info.pix_fmt == 'yuv420p10le'
        assert info.bit_depth == 10

    def test_audio_fields(self):
        info = framepump.VideoFrames(os.path.join(DATA_DIR, 'with_audio.mp4')).info
        assert info.has_audio
        assert info.audio_codec == 'aac'
        assert info.audio_sample_rate > 0

    def test_reports_source_not_view(self):
        """Slicing/resizing must not change info; the view properties do change."""
        v = framepump.VideoFrames(os.path.join(DATA_DIR, 'exact_30fps.mp4'))
        view = v[::2].resized((360, 640))
        assert view.info.fps == v.info.fps
        assert view.info.imshape == (720, 1280)
        assert view.fps == v.fps / 2
        assert view.imshape == (360, 640)

    def test_str_is_readable(self):
        text = str(framepump.VideoFrames(os.path.join(DATA_DIR, 'with_audio.mp4')).info)
        assert 'h264' in text and 'aac' in text and 'bt709' in text


def test_diagnose_smoke(capsys):
    report = framepump.diagnose()
    printed = capsys.readouterr().out
    assert report in printed
    assert 'framepump' in report
    assert 'PyAV' in report
    assert 'GPU features:' in report
