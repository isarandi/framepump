"""Tests for VideoFrames class."""

import numpy as np
import pytest

from framepump import VideoFrames, VideoWriter


@pytest.fixture
def sample_video(tmp_path):
    """Create a small test video for testing."""
    video_path = tmp_path / 'test_video.mp4'
    fps = 10
    n_frames = 30
    height, width = 64, 64

    with VideoWriter(str(video_path), fps=fps) as writer:
        for i in range(n_frames):
            # Create a frame with a unique pattern for each frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add frame number as intensity
            frame[:, :, 0] = i * 8  # Red channel varies with frame
            frame[:, :, 1] = 128  # Green constant
            frame[:, :, 2] = 64  # Blue constant
            writer.append_data(frame)

    return str(video_path), fps, n_frames, (height, width)


class TestVideoFramesBasic:
    """Basic VideoFrames functionality tests."""

    def test_create_videoframes(self, sample_video):
        """Test creating a VideoFrames instance."""
        video_path, fps, n_frames, imshape = sample_video
        frames = VideoFrames(video_path)

        assert frames.path == video_path
        assert frames.original_fps == fps
        assert tuple(frames.imshape) == imshape

    def test_len(self, sample_video):
        """Test __len__ returns correct frame count."""
        video_path, fps, n_frames, _ = sample_video
        frames = VideoFrames(video_path)
        assert len(frames) == n_frames

    def test_iterate_frames(self, sample_video):
        """Test iterating over frames yields numpy arrays."""
        video_path, fps, n_frames, imshape = sample_video
        frames = VideoFrames(video_path)

        count = 0
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (*imshape, 3)
            assert frame.dtype == np.uint8
            count += 1

        assert count == n_frames

    def test_fps_property(self, sample_video):
        """Test fps property returns correct value."""
        video_path, fps, _, _ = sample_video
        frames = VideoFrames(video_path)
        assert frames.fps == fps

    def test_imshape_property(self, sample_video):
        """Test imshape property returns correct value."""
        video_path, _, _, imshape = sample_video
        frames = VideoFrames(video_path)
        assert tuple(frames.imshape) == imshape


class TestVideoFramesSlicing:
    """Test slicing operations on VideoFrames."""

    def test_slice_start_stop(self, sample_video):
        """Test slicing with start:stop."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        sliced = frames[5:15]
        assert len(sliced) == 10

    def test_slice_with_step(self, sample_video):
        """Test slicing with step."""
        video_path, fps, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        # Every second frame
        sliced = frames[::2]
        assert len(sliced) == (n_frames + 1) // 2

        # FPS should be halved when step is 2
        assert sliced.fps == fps / 2

    def test_slice_chaining(self, sample_video):
        """Test chaining multiple slices."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        # First get every 2nd frame, then take first 10
        sliced = frames[::2][:10]
        assert len(sliced) == 10

    def test_slice_returns_videoframes(self, sample_video):
        """Test that slicing returns a new VideoFrames instance."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        sliced = frames[5:15]
        assert isinstance(sliced, VideoFrames)
        assert sliced is not frames

    def test_slice_is_lazy(self, sample_video):
        """Test that slicing doesn't read frames immediately."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        # This should be instant (no frame reading)
        sliced = frames[::10][:5]
        assert len(sliced) == min(5, (30 + 9) // 10)


class TestVideoFramesResize:
    """Test resize functionality."""

    def test_resized_returns_new_instance(self, sample_video):
        """Test that resized() returns a new VideoFrames instance."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        resized = frames.resized((32, 32))
        assert isinstance(resized, VideoFrames)
        assert resized is not frames

    def test_resized_imshape(self, sample_video):
        """Test that resized frames have correct shape."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        new_shape = (32, 48)
        resized = frames.resized(new_shape)
        assert resized.imshape == new_shape

    def test_resized_frames_actual_shape(self, sample_video):
        """Test that iterated frames have the resized shape."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        new_shape = (32, 48)
        resized = frames.resized(new_shape)

        for frame in resized:
            assert frame.shape == (new_shape[0], new_shape[1], 3)
            break  # Only check first frame


class TestVideoFramesRepeat:
    """Test frame repetition functionality."""

    def test_repeat_each_frame(self, sample_video):
        """Test repeat_each_frame multiplies frame count."""
        video_path, _, n_frames, _ = sample_video
        frames = VideoFrames(video_path)

        repeated = frames.repeat_each_frame(3)
        assert len(repeated) == n_frames * 3

    def test_repeat_fps(self, sample_video):
        """Test that fps is multiplied when repeating frames."""
        video_path, fps, _, _ = sample_video
        frames = VideoFrames(video_path)

        repeated = frames.repeat_each_frame(2)
        assert repeated.fps == fps * 2

    def test_repeat_invalid_count(self, sample_video):
        """Test that repeat_each_frame rejects invalid counts."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        with pytest.raises(ValueError):
            frames.repeat_each_frame(0)


class TestVideoFramesDtype:
    """Test dtype conversion functionality."""

    def test_default_dtype_uint8(self, sample_video):
        """Test default dtype is uint8."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path)

        for frame in frames:
            assert frame.dtype == np.uint8
            break

    def test_dtype_uint16(self, sample_video):
        """Test uint16 dtype."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path, dtype=np.uint16)

        for frame in frames:
            assert frame.dtype == np.uint16
            break

    def test_dtype_float32(self, sample_video):
        """Test float32 dtype normalization."""
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path, dtype=np.float32)

        for frame in frames:
            assert frame.dtype == np.float32
            assert frame.min() >= 0.0
            assert frame.max() <= 1.0
            break

    def test_invalid_dtype(self, sample_video):
        """Test that invalid dtypes are rejected."""
        video_path, _, _, _ = sample_video

        with pytest.raises(ValueError):
            VideoFrames(video_path, dtype=np.int32)


class TestSeekableParameter:
    def test_seekable_false_indexing_matches_default(self, sample_video):
        video_path, _, n_frames, _ = sample_video
        ref = VideoFrames(video_path)
        vf = VideoFrames(video_path, seekable=False)
        assert len(vf) == n_frames
        assert np.array_equal(vf[10], ref[10])
        assert np.array_equal(vf[-1], ref[-1])

    def test_seekable_false_positive_start_slice(self, sample_video):
        video_path, _, _, _ = sample_video
        ref = list(VideoFrames(video_path))
        sliced = list(VideoFrames(video_path, seekable=False)[5:8])
        assert len(sliced) == 3
        for a, b in zip(sliced, ref[5:8]):
            assert np.array_equal(a, b)


class TestTimestamplessStreams:
    """Raw H.264 elementary streams have no timestamps; indexed and windowed
    access must fall back to frame counting instead of yielding nothing."""

    @pytest.fixture
    def raw_h264(self):
        import pathlib

        return str(pathlib.Path(__file__).parent / 'data' / 'raw25.h264')

    def test_positive_start_slice_yields_frames(self, raw_h264):
        vf = VideoFrames(raw_h264)
        full = list(vf)
        assert len(full) == len(vf)
        sliced = list(vf[5:10])
        assert len(sliced) == 5
        for a, b in zip(sliced, full[5:10]):
            assert np.array_equal(a, b)

    def test_positive_start_slice_cfr(self, raw_h264):
        vf = VideoFrames(raw_h264, constant_framerate=True)
        full = list(vf)
        sliced = list(vf[5:10])
        assert len(sliced) == 5
        for a, b in zip(sliced, full[5:10]):
            assert np.array_equal(a, b)

    def test_integer_indexing(self, raw_h264):
        vf = VideoFrames(raw_h264)
        full = list(vf)
        assert np.array_equal(vf[7], full[7])


class TestDtypeSpellings:
    """Any DTypeLike spelling of a supported dtype must be accepted."""

    @pytest.mark.parametrize('spelling', ['float32', float, np.dtype(np.float32)])
    def test_dtype_like_spellings(self, sample_video, spelling):
        video_path, _, _, _ = sample_video
        frames = VideoFrames(video_path, dtype=spelling)
        assert frames[0].dtype in (np.float32, np.float64)

    def test_unsupported_dtype_message_is_clean(self, sample_video):
        video_path, _, _, _ = sample_video
        with pytest.raises(ValueError, match='Unsupported dtype'):
            VideoFrames(video_path, dtype='int32')


class TestIndexErrorWording:
    def test_empty_view_blames_the_view(self, sample_video):
        video_path, _, n_frames, _ = sample_video
        vf = VideoFrames(video_path)
        with pytest.raises(
            IndexError, match=rf'view with 0 frames \(source video has {n_frames}\)'
        ):
            vf[10:10][0]

    def test_whole_video_message_unchanged(self, sample_video):
        video_path, _, n_frames, _ = sample_video
        vf = VideoFrames(video_path)
        with pytest.raises(IndexError, match=f'video with {n_frames} frames'):
            vf[n_frames]


class TestInterlacedChroma:
    """Interlaced 4:2:0 chroma is per-field; conversion must be field-aware
    (matching FFmpeg's frame-based sws_scale_frame API), not progressive."""

    @pytest.fixture
    def interlaced_video(self, tmp_path):
        import shutil
        import subprocess

        if shutil.which('ffmpeg') is None:
            pytest.skip('ffmpeg CLI not available')
        path = tmp_path / 'interlaced.mpg'
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-v',
                'error',
                '-f',
                'lavfi',
                '-i',
                'testsrc2=duration=1:size=192x144:rate=25',
                '-c:v',
                'mpeg2video',
                '-flags',
                '+ildct+ilme',
                '-top',
                '1',
                str(path),
            ],
            check=True,
        )
        return str(path)

    def test_matches_field_aware_reference(self, interlaced_video):
        import av

        with av.open(interlaced_video) as container:
            frame = next(container.decode(video=0))
            assert frame.interlaced_frame, 'fixture must be interlaced-flagged'
            # to_ndarray uses sws_scale_frame, which is interlace-aware
            reference = frame.to_ndarray(format='rgb24')

        got = next(iter(VideoFrames(interlaced_video)))
        assert np.array_equal(got, reference)


class TestArrayProtocol:
    """np.asarray materializes a view in one sequential decode pass."""

    def test_asarray_matches_iteration(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        arr = np.asarray(v[3:8])
        assert arr.shape[0] == 5
        assert np.array_equal(arr, np.stack(list(v[3:8])))

    def test_empty_selection(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        h, w = v.imshape
        assert np.asarray(v[5:5]).shape == (0, h, w, 3)

    def test_dtype_argument(self, sample_video):
        path, *_ = sample_video
        arr = np.asarray(VideoFrames(path)[:2], dtype=np.float32)
        assert arr.dtype == np.float32


class TestFancyIndexing:
    """Integer-list indexing decodes the listed frames into a stacked array."""

    def test_list_matches_singles(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        picked = v[[0, 2, 5, 2, -1]]
        expected = [0, 2, 5, 2, len(v) - 1]
        assert picked.shape == (5, *v.imshape, 3)
        for i, j in enumerate(expected):
            assert np.array_equal(picked[i], v[j])

    def test_numpy_index_array_and_scalar(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        assert v[np.array([1, 3])].shape == (2, *v.imshape, 3)
        assert v[np.int64(3)].shape == (*v.imshape, 3)  # scalar stays a single frame

    def test_invalid_indices_rejected(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        with pytest.raises(TypeError):
            v[[True, False]]
        with pytest.raises(TypeError):
            v[[0.5, 1.5]]
        with pytest.raises(IndexError):
            v[[0, 10_000]]


class TestFramesAt:
    """Lazy gap-aware access: decode-through small gaps, seek over large ones."""

    def test_given_order_preserved(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        wanted = [2, 5, 16, 3, 3, -1]
        resolved = [2, 5, 16, 3, 3, len(v) - 1]
        for got, j in zip(v.frames_at(wanted), resolved):
            assert np.array_equal(got, v[j])

    def test_is_lazy(self, sample_video):
        path, *_ = sample_video
        gen = VideoFrames(path).frames_at([0, 10_000])  # bad index not reached
        assert np.array_equal(next(gen), VideoFrames(path)[0])
        with pytest.raises(IndexError):
            next(gen)

    def test_repeat_view_rejected(self, sample_video):
        path, *_ = sample_video
        with pytest.raises(NotImplementedError):
            next(VideoFrames(path).repeat_each_frame(2).frames_at([0]))

    def test_on_sliced_view(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)[::2]
        for got, j in zip(v.frames_at([1, 4]), [1, 4]):
            assert np.array_equal(got, v[j])


class TestBatched:
    """batched() yields freshly allocated stacked batches from one pass."""

    def test_batches_match_asarray(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path, dtype=np.float32)
        batches = list(v.batched(8))
        assert [b.shape[0] for b in batches] == [8, 8, 8, 6]
        assert all(b.dtype == np.float32 for b in batches)
        assert np.array_equal(np.concatenate(batches), np.asarray(v))

    def test_validation(self, sample_video):
        path, *_ = sample_video
        v = VideoFrames(path)
        with pytest.raises(ValueError):
            next(v.batched(0))
        with pytest.raises(TypeError):
            next(v.batched('8'))
