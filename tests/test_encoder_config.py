"""Tests for EncoderConfig validation."""

import pytest

from framepump import EncoderConfig


class TestValidConfigs:
    def test_defaults(self):
        config = EncoderConfig()
        assert config.crf == 15
        assert config.bframes == 2
        assert config.gop == 250
        assert config.codec == 'h264'

    @pytest.mark.parametrize('crf', [0, 15, 51])
    def test_crf_range(self, crf):
        assert EncoderConfig(crf=crf).crf == crf

    @pytest.mark.parametrize('bframes', [0, 2, 4])
    def test_bframes_range(self, bframes):
        assert EncoderConfig(bframes=bframes).bframes == bframes

    @pytest.mark.parametrize('gop', [1, 30, 250, 100000])
    def test_gop_range(self, gop):
        assert EncoderConfig(gop=gop).gop == gop

    @pytest.mark.parametrize('codec', ['h264', 'hevc'])
    def test_codecs(self, codec):
        assert EncoderConfig(codec=codec).codec == codec

    @pytest.mark.parametrize(
        'preset',
        [
            None,
            'p1',
            'p2',
            'p3',
            'p4',
            'p5',
            'p6',
            'p7',
            'ultrafast',
            'superfast',
            'veryfast',
            'faster',
            'fast',
            'medium',
            'slow',
            'slower',
            'veryslow',
        ],
    )
    def test_presets(self, preset):
        assert EncoderConfig(preset=preset).preset == preset

    def test_with_overrides_valid(self):
        config = EncoderConfig().with_overrides(crf=20, bframes=0)
        assert config.crf == 20
        assert config.bframes == 0


class TestInvalidConfigs:
    @pytest.mark.parametrize('crf', [-1, 52, 100])
    def test_crf_out_of_range(self, crf):
        with pytest.raises(ValueError, match='crf'):
            EncoderConfig(crf=crf)

    @pytest.mark.parametrize('crf', [15.5, '15', None, True])
    def test_crf_wrong_type(self, crf):
        with pytest.raises(TypeError, match='crf'):
            EncoderConfig(crf=crf)

    @pytest.mark.parametrize('bframes', [-1, 5, 100])
    def test_bframes_out_of_range(self, bframes):
        with pytest.raises(ValueError, match='bframes'):
            EncoderConfig(bframes=bframes)

    @pytest.mark.parametrize('bframes', [2.0, '2', False])
    def test_bframes_wrong_type(self, bframes):
        with pytest.raises(TypeError, match='bframes'):
            EncoderConfig(bframes=bframes)

    @pytest.mark.parametrize('gop', [0, -1, -250])
    def test_gop_out_of_range(self, gop):
        with pytest.raises(ValueError, match='gop'):
            EncoderConfig(gop=gop)

    @pytest.mark.parametrize('gop', [30.0, '250', True])
    def test_gop_wrong_type(self, gop):
        with pytest.raises(TypeError, match='gop'):
            EncoderConfig(gop=gop)

    @pytest.mark.parametrize('codec', ['h265', 'av1', 'H264', 'x264', ''])
    def test_unknown_codec(self, codec):
        """'h265' must raise, not silently encode as h264."""
        with pytest.raises(ValueError, match='codec'):
            EncoderConfig(codec=codec)

    @pytest.mark.parametrize('preset', ['blazing', 'p0', 'p8', 'MEDIUM', ''])
    def test_unknown_preset(self, preset):
        with pytest.raises(ValueError, match='preset'):
            EncoderConfig(preset=preset)

    def test_with_overrides_revalidates(self):
        with pytest.raises(ValueError, match='crf'):
            EncoderConfig().with_overrides(crf=100)
