"""Stress tests to break framepump - find weak spots in the library."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from framepump import VideoFrames, VideoWriter, VideoDecodeError, get_fps, get_duration, num_frames, video_extents

# Path to various test video directories
FATE_DIR = Path(__file__).parent.parent / 'fate'
DATA_DIR = Path(__file__).parent.parent / 'data'

_has_data = DATA_DIR.exists() and any(DATA_DIR.glob('*.*'))
_has_fate = FATE_DIR.exists() and any(FATE_DIR.iterdir())

# Files in fate/ that have no video stream (verified with ffprobe)
# These should raise ValueError("No video stream found")
AUDIO_ONLY_FATE_FILES = {
    'aac/CT_DecoderCheck/File2.mp4',
    'aac/CT_DecoderCheck/File3.mp4',
    'aac/CT_DecoderCheck/File4.mp4',
    'aac/CT_DecoderCheck/File5.mp4',
    'aac/CT_DecoderCheck/sbr_bc-ps_bc.mp4',
    'aac/CT_DecoderCheck/sbr_bc-ps_i.mp4',
    'aac/CT_DecoderCheck/sbr_i-ps_bic.mp4',
    'aac/CT_DecoderCheck/sbr_i-ps_i.mp4',
    'aac/Fd_2_c1_Ms_0x01.mp4',
    'aac/Fd_2_c1_Ms_0x04.mp4',
    'aac/al04_44.mp4',
    'aac/al04sf_48.mp4',
    'aac/al05_44.mp4',
    'aac/al06_44.mp4',
    'aac/al07_96.mp4',
    'aac/al15_44.mp4',
    'aac/al17_44.mp4',
    'aac/al18_44.mp4',
    'aac/al_sbr_cm_48_2.mp4',
    'aac/al_sbr_cm_48_5.1.mp4',
    'aac/al_sbr_ps_04_new.mp4',
    'aac/al_sbr_ps_06_new.mp4',
    'aac/al_sbr_sr_48_2_fsaac48.mp4',
    'aac/am00_88.mp4',
    'aac/am05_44.mp4',
    'aac/ap05_48.mp4',
    'aac/er_ad6000np_44_ep0.mp4',
    'aac/er_eld1001np_44_ep0.mp4',
    'aac/er_eld2000np_48_ep0.mp4',
    'aac/er_eld2100np_48_ep0.mp4',
    'audiomatch/tones_dolby_44100_mono_aac_he.mp4',
    'audiomatch/tones_dolby_44100_mono_aac_lc.mp4',
    'audiomatch/tones_dolby_44100_stereo_aac_he.mp4',
    'audiomatch/tones_dolby_44100_stereo_aac_he2.mp4',
    'audiomatch/tones_dolby_44100_stereo_aac_lc.mp4',
    'audiomatch/tones_quicktime7_44100_stereo_aac_lc.mp4',
    'duck/salsa-audio-only.avi',
    'duck/sop-audio-only.avi',
    'gsm/sample-gsm-8000.mov',
    'imc/imc.avi',
    'lossless-audio/als_00_2ch48k16b.mp4',
    'lossless-audio/als_01_2ch48k16b.mp4',
    'lossless-audio/als_02_2ch48k16b.mp4',
    'lossless-audio/als_03_2ch48k16b.mp4',
    'lossless-audio/als_04_2ch48k16b.mp4',
    'lossless-audio/als_05_2ch48k16b.mp4',
    'lossless-audio/als_07_2ch192k32bF.mp4',
    'lossless-audio/als_09_512ch2k16b.mp4',
    'mkv/codec_delay_opus.mkv',
    'mov/aac-2048-priming.mov',
    'mov/faststart-4gb-overflow.mov',
    'mov/mov_neg_first_pts_discard_vorbis.mp4',
    'mpegaudio/packed_maindata.mp3.mp4',
    'mpegh3da/mpegh_config_change_cicp_2_14_6_lc_baseline_compatible_32kbps.mp4',
    'nellymoser/nellymoser.flv',
    'qt-surge-suite/surge-1-16-B-alaw.mov',
    'qt-surge-suite/surge-1-16-B-ima4.mov',
    'qt-surge-suite/surge-1-16-B-ulaw.mov',
    'qt-surge-suite/surge-1-8-MAC3.mov',
    'qt-surge-suite/surge-1-8-MAC6.mov',
    'qt-surge-suite/surge-1-8-raw.mov',
    'qt-surge-suite/surge-2-16-B-QDM2.mov',
    'qt-surge-suite/surge-2-16-B-alaw.mov',
    'qt-surge-suite/surge-2-16-B-ima4.mov',
    'qt-surge-suite/surge-2-16-B-twos.mov',
    'qt-surge-suite/surge-2-16-B-ulaw.mov',
    'qt-surge-suite/surge-2-16-L-ms02.mov',
    'qt-surge-suite/surge-2-16-L-ms11.mov',
    'qt-surge-suite/surge-2-16-L-sowt.mov',
    'qt-surge-suite/surge-2-8-MAC3.mov',
    'qt-surge-suite/surge-2-8-MAC6.mov',
    'qt-surge-suite/surge-2-8-raw.mov',
    'sub/MovText_capability_tester.mp4',
    'vp5/potter512-400-partial.avi',
    'vp8/dash_audio1.webm',
    'vp8/dash_audio2.webm',
    'vp8/dash_audio3.webm',
}

# Partial/truncated files - verified with ffmpeg to produce errors during decode
# Both success (partial decode) and error are valid outcomes for these
PARTIAL_FATE_FILES = {
    '8bps/full9iron-partial.mov',
    'VMnc/VS2k5DebugDemo-01-partial.avi',
    'cljr/testcljr-partial.avi',
    'cvid/catfight-cvid-pal8-partial.mov',
    'cvid/laracroft-cinepak-partial.avi',
    'duck/sonic3dblast_intro-partial.avi',
    'duck/vf2end-partial.avi',
    'fic/fic-partial-2MB.avi',
    'fraps/Griffin_Ragdoll01-partial.avi',
    'fraps/WoW_2006-11-03_14-58-17-19-nosound-partial.avi',
    'fraps/fraps-v5-bouncing-balls-partial.avi',
    'fraps/psclient-partial.avi',
    'fraps/test3-nosound-partial.avi',
    'iv41/indeo41-partial.avi',
    'png1/corepng-partial.avi',
    'tscc/2004-12-17-uebung9-partial.avi',
    'v210/v210_720p-partial.avi',
    'vble/flowers-partial-2MB.avi',
    'vp5/potter512-400-partial.avi',
    'wc4-xan/wc4trailer-partial.avi',
    'zmbv/wc2_001-partial.avi',
}


def find_videos_in_fate(extensions=None, max_count=50):
    """Find video files in fate/ directory for testing exotic formats."""
    if extensions is None:
        extensions = {'.avi', '.mov', '.mp4', '.mkv', '.webm', '.flv'}

    videos = []
    for ext in extensions:
        for p in FATE_DIR.rglob(f'*{ext}'):
            videos.append(p)
            if len(videos) >= max_count:
                return videos
    return videos


@pytest.mark.skipif(not _has_fate, reason='fate/ test data not available')
class TestExoticFormats:
    """Test various exotic video formats from fate/ directory."""

    @pytest.fixture(scope='class')
    def fate_videos(self):
        """Get a sample of videos from fate/."""
        return find_videos_in_fate(max_count=30)

    @pytest.mark.parametrize('video_idx', range(30))
    def test_can_open_fate_video(self, fate_videos, video_idx):
        """Test that we can at least open various formats."""
        if video_idx >= len(fate_videos):
            pytest.skip('Not enough videos')

        video_path = fate_videos[video_idx]
        rel_path = str(video_path.relative_to(FATE_DIR))
        is_audio_only = rel_path in AUDIO_ONLY_FATE_FILES
        is_partial = rel_path in PARTIAL_FATE_FILES

        try:
            frames = VideoFrames(str(video_path))
            # Try to get basic info without iterating
            _ = len(frames)
            _ = frames.fps
            _ = frames.imshape
            if is_audio_only:
                pytest.fail(f'{video_path.name} should have raised "No video stream" error')
        except ValueError as e:
            if 'No video stream' in str(e):
                if is_audio_only:
                    return  # Expected error for audio-only file
                pytest.fail(f'{video_path.name} has video but got "No video stream" error')
            pytest.fail(f'Failed to open {video_path.name}: {e}')
        except VideoDecodeError as e:
            if is_partial:
                return  # Expected error for partial/corrupt file
            pytest.fail(f'Unexpected decode error for {video_path.name}: {e}')
        except Exception as e:
            pytest.fail(f'Failed to open {video_path.name}: {e}')

    @pytest.mark.parametrize('video_idx', range(15))
    def test_can_iterate_fate_video(self, fate_videos, video_idx):
        """Test that we can iterate through exotic formats."""
        if video_idx >= len(fate_videos):
            pytest.skip('Not enough videos')

        video_path = fate_videos[video_idx]
        rel_path = str(video_path.relative_to(FATE_DIR))
        is_audio_only = rel_path in AUDIO_ONLY_FATE_FILES
        is_partial = rel_path in PARTIAL_FATE_FILES

        try:
            frames = VideoFrames(str(video_path))
            count = 0
            for frame in frames:
                assert frame.ndim == 3
                assert frame.shape[2] == 3
                count += 1
                if count >= 5:
                    break
            assert count > 0, f'No frames yielded from {video_path.name}'
            if is_audio_only:
                pytest.fail(f'{video_path.name} should have raised "No video stream" error')
        except ValueError as e:
            if 'No video stream' in str(e):
                if is_audio_only:
                    return  # Expected error for audio-only file
                pytest.fail(f'{video_path.name} has video but got "No video stream" error')
            pytest.fail(f'Failed to iterate {video_path.name}: {e}')
        except VideoDecodeError as e:
            if is_partial:
                return  # Expected error for partial/corrupt file
            pytest.fail(f'Unexpected decode error for {video_path.name}: {e}')
        except Exception as e:
            pytest.fail(f'Failed to iterate {video_path.name}: {e}')


class TestSlicingEdgeCases:
    """Test slicing edge cases that might break the library."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a test video."""
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(50):
                frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_empty_slice(self, sample_video):
        """Empty slice should yield no frames."""
        frames = VideoFrames(sample_video)
        sliced = frames[10:10]  # Empty range
        assert len(sliced) == 0
        count = sum(1 for _ in sliced)
        assert count == 0

    def test_slice_beyond_end(self, sample_video):
        """Slice going beyond video length."""
        frames = VideoFrames(sample_video)
        sliced = frames[40:1000]  # Beyond the 50 frames
        assert len(sliced) == 10  # Should clamp to actual length
        count = sum(1 for _ in sliced)
        assert count == 10

    def test_negative_slice_start(self, sample_video):
        """Negative slice indices."""
        frames = VideoFrames(sample_video)
        sliced = frames[-10:]  # Last 10 frames
        assert len(sliced) == 10

    def test_negative_slice_end(self, sample_video):
        """Negative end index."""
        frames = VideoFrames(sample_video)
        sliced = frames[:-10]  # All but last 10
        assert len(sliced) == 40

    def test_very_large_step(self, sample_video):
        """Step larger than video length."""
        frames = VideoFrames(sample_video)
        sliced = frames[::100]  # Step > length
        assert len(sliced) == 1  # Should get just first frame
        count = sum(1 for _ in sliced)
        assert count == 1

    def test_double_slice_consistency(self, sample_video):
        """Double slicing should be consistent."""
        frames = VideoFrames(sample_video)

        # Method 1: Single slice
        single = frames[10:30:2]

        # Method 2: Chain slices
        chained = frames[10:30][::2]

        assert len(single) == len(chained)

        # Compare actual frames
        for f1, f2 in zip(single, chained):
            np.testing.assert_array_equal(f1, f2)

    def test_slice_then_repeat_not_supported(self, sample_video):
        """Slicing a repeated video should raise."""
        frames = VideoFrames(sample_video)
        repeated = frames.repeat_each_frame(2)

        with pytest.raises(NotImplementedError):
            _ = repeated[5:10]

    def test_integer_index_negative(self, sample_video):
        """Negative integer indexing."""
        frames = VideoFrames(sample_video)
        frame_last = frames[-1]
        assert isinstance(frame_last, np.ndarray)

    def test_integer_index_out_of_bounds(self, sample_video):
        """Out of bounds integer index."""
        frames = VideoFrames(sample_video)
        with pytest.raises(IndexError):
            _ = frames[1000]

    def test_integer_index_negative_out_of_bounds(self, sample_video):
        """Negative index out of bounds."""
        frames = VideoFrames(sample_video)
        with pytest.raises(IndexError):
            _ = frames[-1000]


class TestDtypeEdgeCases:
    """Test dtype conversion edge cases."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(10):
                frame = np.full((64, 64, 3), 255, dtype=np.uint8)  # Max value
                writer.append_data(frame)
        return str(video_path)

    def test_float16_no_overflow(self, sample_video):
        """float16 should not overflow on max uint16 values."""
        frames = VideoFrames(sample_video, dtype=np.float16)
        for frame in frames:
            assert frame.dtype == np.float16
            assert np.isfinite(frame).all(), 'Got inf/nan in float16 conversion'
            assert frame.max() <= 1.0
            break

    def test_float64_precision(self, sample_video):
        """float64 should preserve precision."""
        frames = VideoFrames(sample_video, dtype=np.float64)
        for frame in frames:
            assert frame.dtype == np.float64
            break

    def test_uint16_range(self, sample_video):
        """uint16 frames should use full range (values > 255)."""
        frames = VideoFrames(sample_video, dtype=np.uint16)
        for frame in frames:
            assert frame.dtype == np.uint16
            # Source is all-255 uint8 → uint16 should scale above 255
            assert frame.max() > 255, (
                f'uint16 frame max is {frame.max()}, expected > 255 '
                f'(full-range scaling from uint8 source)'
            )
            break


@pytest.mark.skipif(not _has_data, reason='data/ test files not available')
class TestSeekingEdgeCases:
    """Test frame seeking edge cases."""

    @pytest.fixture
    def vfr_webm(self):
        """Get a webm that might be VFR."""
        webms = list(DATA_DIR.glob('*.webm'))
        if webms:
            return str(webms[0])
        pytest.skip('No webm files available')

    def test_seek_first_frame(self, vfr_webm):
        """Seeking to first frame."""
        frames = VideoFrames(vfr_webm)
        frame = frames[0]
        assert frame.ndim == 3

    def test_seek_last_frame(self, vfr_webm):
        """Seeking to last frame."""
        frames = VideoFrames(vfr_webm)
        frame = frames[-1]
        assert frame.ndim == 3

    def test_seek_middle_frame(self, vfr_webm):
        """Seeking to middle frame."""
        frames = VideoFrames(vfr_webm)
        mid = len(frames) // 2
        frame = frames[mid]
        assert frame.ndim == 3

    def test_random_access_order_independence(self, vfr_webm):
        """Random access should return same frame regardless of order."""
        frames = VideoFrames(vfr_webm)
        n = min(len(frames), 20)

        # Get frame 10 first
        frame10_first = frames[min(10, n-1)].copy()

        # Get other frames, then frame 10 again
        _ = frames[min(5, n-1)]
        _ = frames[min(15, n-1)]
        frame10_second = frames[min(10, n-1)]

        np.testing.assert_array_equal(frame10_first, frame10_second)


@pytest.mark.skipif(not _has_data, reason='data/ test files not available')
class TestFrameCountAccuracy:
    """Test frame count is accurate for various video types."""

    @pytest.fixture
    def mp4_videos(self):
        return list(DATA_DIR.glob('*.mp4'))[:5]

    @pytest.mark.parametrize('video_idx', range(5))
    def test_len_matches_iteration(self, mp4_videos, video_idx):
        """len(VideoFrames) should match actual iteration count."""
        if video_idx >= len(mp4_videos):
            pytest.skip('Not enough videos')

        video_path = str(mp4_videos[video_idx])
        frames = VideoFrames(video_path)

        expected = len(frames)
        actual = sum(1 for _ in frames)

        # Allow small tolerance for VFR
        assert abs(expected - actual) <= 2, f'len={expected}, iterated={actual}'


class TestVideoWriterEdgeCases:
    """Test VideoWriter edge cases."""

    def test_write_to_nonexistent_dir(self, tmp_path):
        """Writing to nonexistent directory should work (creates parent)."""
        video_path = tmp_path / 'subdir' / 'nested' / 'video.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)

        assert video_path.exists()

    def test_write_zero_frames(self, tmp_path):
        """Writing zero frames should not crash."""
        video_path = tmp_path / 'empty.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            pass  # Write nothing

        # File might or might not exist, but no crash

    def test_write_very_small_frame(self, tmp_path):
        """Very small frames (might fail with NVENC)."""
        video_path = tmp_path / 'tiny.mp4'

        with VideoWriter(str(video_path), fps=10) as writer:
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            writer.append_data(frame)

        assert video_path.exists()

    def test_write_odd_dimensions(self, tmp_path):
        """Odd dimension frames should raise informative error."""
        video_path = tmp_path / 'odd.mp4'

        with pytest.raises((ValueError, RuntimeError), match='dimensions must be even'):
            with VideoWriter(str(video_path), fps=10) as writer:
                frame = np.zeros((65, 65, 3), dtype=np.uint8)
                writer.append_data(frame)

    def test_multiple_sequences(self, tmp_path):
        """Writing multiple video sequences."""
        writer = VideoWriter()

        for i in range(3):
            video_path = tmp_path / f'seq_{i}.mp4'
            writer.start_sequence(str(video_path), fps=10)
            frame = np.full((64, 64, 3), i * 80, dtype=np.uint8)
            writer.append_data(frame)
            writer.end_sequence()
            assert video_path.exists()

        writer.close()

    def test_end_sequence_without_start(self, tmp_path):
        """end_sequence without start should raise."""
        writer = VideoWriter()

        with pytest.raises(ValueError):
            writer.end_sequence()

        writer.close()

    def test_append_without_start(self, tmp_path):
        """append_data without start should raise."""
        writer = VideoWriter()

        with pytest.raises(ValueError):
            writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))

        writer.close()


class TestResizeEdgeCases:
    """Test resize functionality edge cases."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(10):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_resize_to_very_small(self, sample_video):
        """Resize to very small dimensions."""
        frames = VideoFrames(sample_video).resized((4, 4))

        for frame in frames:
            assert frame.shape == (4, 4, 3)
            break

    def test_resize_to_very_large(self, sample_video):
        """Resize to large dimensions."""
        frames = VideoFrames(sample_video).resized((1000, 1000))

        for frame in frames:
            assert frame.shape == (1000, 1000, 3)
            break

    def test_resize_non_square(self, sample_video):
        """Resize to non-square aspect ratio."""
        frames = VideoFrames(sample_video).resized((50, 200))

        for frame in frames:
            assert frame.shape == (50, 200, 3)
            break

    def test_resize_combined_with_slice(self, sample_video):
        """Resize combined with slicing."""
        frames = VideoFrames(sample_video)[::2].resized((32, 32))

        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (32, 32, 3)
            break


class TestCorruptedInputs:
    """Test handling of corrupted or unusual inputs."""

    def test_nonexistent_file(self):
        """Opening nonexistent file should raise."""
        with pytest.raises(Exception):
            VideoFrames('/path/to/nonexistent/video.mp4')

    def test_non_video_file(self, tmp_path):
        """Opening a non-video file should raise."""
        text_file = tmp_path / 'not_a_video.txt'
        text_file.write_text('This is not a video')

        with pytest.raises(Exception):
            VideoFrames(str(text_file))

    def test_truncated_video(self, tmp_path):
        """Test handling of truncated video files."""
        # Create a valid video first
        video_path = tmp_path / 'truncated.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(30):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.append_data(frame)

        # Now truncate it
        truncated_path = tmp_path / 'truncated2.mp4'
        with open(video_path, 'rb') as f:
            data = f.read()
        with open(truncated_path, 'wb') as f:
            f.write(data[:len(data)//2])  # Write only half

        # Should either decode some frames or raise VideoDecodeError
        try:
            frames = VideoFrames(str(truncated_path))
            count = sum(1 for _ in frames)
            # Truncated at half → should get fewer frames than original
            assert 0 < count < 30, (
                f'Expected partial decode (1-29 frames), got {count}'
            )
        except (VideoDecodeError, RuntimeError):
            pass  # Clean error on corrupt data is acceptable


class TestCFRvsVFR:
    """Test constant vs variable framerate handling."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=30) as writer:
            for i in range(60):
                frame = np.full((64, 64, 3), i * 4, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_cfr_mode_default(self, sample_video):
        """Default mode should work."""
        frames = VideoFrames(sample_video, constant_framerate=True)
        count = sum(1 for _ in frames)
        assert count > 0

    def test_vfr_mode(self, sample_video):
        """VFR mode should work."""
        frames = VideoFrames(sample_video, constant_framerate=False)
        count = sum(1 for _ in frames)
        assert count > 0

    def test_custom_target_fps(self, sample_video):
        """Custom target FPS."""
        # Double the FPS
        frames = VideoFrames(sample_video, constant_framerate=60.0)
        # Should get roughly double the frames
        expected = len(VideoFrames(sample_video)) * 2
        actual = len(frames)
        assert abs(actual - expected) <= 5

    def test_half_target_fps(self, sample_video):
        """Half the target FPS."""
        frames = VideoFrames(sample_video, constant_framerate=15.0)
        expected = len(VideoFrames(sample_video)) // 2
        actual = len(frames)
        assert abs(actual - expected) <= 3


class TestRepeatEachFrame:
    """Test repeat_each_frame functionality."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        video_path = tmp_path / 'test.mp4'
        with VideoWriter(str(video_path), fps=10) as writer:
            for i in range(10):
                frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
                writer.append_data(frame)
        return str(video_path)

    def test_repeat_once(self, sample_video):
        """repeat_each_frame(1) should be no-op."""
        frames = VideoFrames(sample_video).repeat_each_frame(1)
        assert len(frames) == 10

    def test_repeat_three_times(self, sample_video):
        """repeat_each_frame(3) should triple frame count."""
        frames = VideoFrames(sample_video).repeat_each_frame(3)
        assert len(frames) == 30

    def test_repeat_negative_raises(self, sample_video):
        """Negative repeat count should raise."""
        frames = VideoFrames(sample_video)
        with pytest.raises(ValueError):
            frames.repeat_each_frame(-1)

    def test_repeat_zero_raises(self, sample_video):
        """Zero repeat count should raise."""
        frames = VideoFrames(sample_video)
        with pytest.raises(ValueError):
            frames.repeat_each_frame(0)

    def test_repeat_chained(self, sample_video):
        """Chained repeats should multiply."""
        frames = VideoFrames(sample_video).repeat_each_frame(2).repeat_each_frame(3)
        assert len(frames) == 10 * 2 * 3


@pytest.mark.skipif(not _has_data, reason='data/ test files not available')
class TestBFrameVideos:
    """Test videos with B-frames (reordering)."""

    @pytest.fixture
    def h264_videos(self):
        """Find H264 videos that likely have B-frames."""
        videos = []
        for pattern in ['*.mp4', '*.mkv']:
            videos.extend(DATA_DIR.glob(pattern))
        return videos[:5]

    @pytest.mark.parametrize('video_idx', range(5))
    def test_seek_in_bframe_video(self, h264_videos, video_idx):
        """Seeking in B-frame videos should work correctly."""
        if video_idx >= len(h264_videos):
            pytest.skip('Not enough videos')

        video_path = str(h264_videos[video_idx])
        frames = VideoFrames(video_path)

        if len(frames) < 10:
            pytest.skip('Video too short')

        # Seek to various points
        frame0 = frames[0]
        frame5 = frames[5]
        frame9 = frames[9]

        # Frames should be different
        assert not np.array_equal(frame0, frame5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
