"""VideoFramesCuda feature parity: resized(), repeat_each_frame(), float dtypes.

The GPU resize (NPP area/Lanczos) does not bit-match swscale's bicubic, so
value comparisons against the CPU class use smooth content and tolerances;
structural invariants (shapes, lengths, fps, index mapping) are exact.
"""

from pathlib import Path

import numpy as np
import pytest

from framepump import VideoFrames, VideoWriter


def _require_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available for torch')
    pytest.importorskip('PyNvVideoCodec')
    return torch


def _cuda_frames(path, **kw):
    from framepump import VideoFramesCuda

    return VideoFramesCuda(path, **kw)


def _np(obj):
    import torch

    return torch.from_dlpack(obj).cpu().numpy()


@pytest.fixture(scope='module')
def smooth_video(tmp_path_factory):
    """Smooth gradients with a per-frame brightness shift (no sharp edges)."""
    path = tmp_path_factory.mktemp('cuda_feat') / 'smooth.mp4'
    yy, xx = np.mgrid[0:240, 0:320]
    base = np.stack(
        [xx * 255 // 319, yy * 255 // 239, (xx + yy) * 255 // 558], axis=-1
    ).astype(np.int16)
    with VideoWriter(str(path), fps=30) as w:
        for i in range(12):
            w.append_data(np.clip(base + i * 3, 0, 255).astype(np.uint8))
    return str(path)


class TestResized:
    def test_shape_len_fps(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video).resized((120, 160))
        assert v.imshape == (120, 160)
        assert v.original_imshape == (240, 320)
        assert len(v) == 12
        frame = _np(v[0])
        assert frame.shape == (120, 160, 3)

    def test_bad_shape_rejected(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video)
        with pytest.raises(TypeError):
            v.resized([120, 160])
        with pytest.raises(TypeError):
            v.resized((120.0, 160))

    def test_downscale_close_to_cpu(self, smooth_video):
        _require_cuda()
        cpu = VideoFrames(smooth_video).resized((120, 160))[5].astype(np.int32)
        gpu = _np(_cuda_frames(smooth_video).resized((120, 160))[5]).astype(np.int32)
        diff = np.abs(cpu - gpu)
        assert diff.mean() < 3 and np.percentile(diff, 99) < 10, (
            f'mean {diff.mean():.2f}, p99 {np.percentile(diff, 99):.1f}'
        )

    def test_upscale_close_to_cpu(self, smooth_video):
        _require_cuda()
        cpu = VideoFrames(smooth_video).resized((480, 640))[2].astype(np.int32)
        gpu = _np(_cuda_frames(smooth_video).resized((480, 640))[2]).astype(np.int32)
        diff = np.abs(cpu - gpu)
        assert diff.mean() < 3 and np.percentile(diff, 99) < 10

    def test_iteration_matches_indexing(self, smooth_video):
        _require_cuda()
        import torch

        v = _cuda_frames(smooth_video).resized((120, 160))
        iterated = [torch.from_dlpack(f).clone() for f in v[:4]]
        for i, it_frame in enumerate(iterated):
            assert torch.equal(it_frame, torch.from_dlpack(v[i]))

    def test_reverse_iteration_with_resize(self, smooth_video):
        _require_cuda()
        import torch

        v = _cuda_frames(smooth_video).resized((120, 160))
        fwd = [torch.from_dlpack(f).clone() for f in v[:6]]
        rev = [torch.from_dlpack(f).clone() for f in v[:6][::-1]]
        for a, b in zip(fwd, reversed(rev)):
            assert torch.equal(a, b)

    def test_uint16_resized(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video, dtype=np.uint16).resized((120, 160))
        frame = _np(v[0])
        assert frame.shape == (120, 160, 3) and frame.dtype == np.uint16
        assert frame.max() > 255  # actually 16-bit scaled


class TestRepeatEachFrame:
    def test_len_fps_mapping(self, smooth_video):
        _require_cuda()
        import torch

        v = _cuda_frames(smooth_video)[:4]
        rep = v.repeat_each_frame(3)
        assert len(rep) == 12
        assert rep.fps == pytest.approx(v.fps * 3)
        assert torch.equal(torch.from_dlpack(rep[7]), torch.from_dlpack(v[2]))

    def test_iteration_repeats_values(self, smooth_video):
        _require_cuda()
        import torch

        for kw in ({}, {'dtype': np.uint16}):
            v = _cuda_frames(smooth_video, **kw)[:3]
            frames = [torch.from_dlpack(f).clone() for f in v.repeat_each_frame(2)]
            singles = [torch.from_dlpack(f).clone() for f in v]
            assert len(frames) == 6
            for i in range(3):
                assert torch.equal(frames[2 * i], singles[i])
                assert torch.equal(frames[2 * i + 1], singles[i])

    def test_slice_after_repeat_raises(self, smooth_video):
        _require_cuda()
        rep = _cuda_frames(smooth_video).repeat_each_frame(2)
        with pytest.raises(NotImplementedError):
            rep[1:3]

    def test_validation(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video)
        with pytest.raises(ValueError):
            v.repeat_each_frame(0)
        with pytest.raises(TypeError):
            v.repeat_each_frame(1.5)


class TestGammaCorrectResize:
    @pytest.fixture(scope='class')
    def checker_video(self, tmp_path_factory):
        """1-pixel gray checkerboard, encoded losslessly (luma-only pattern).

        Area-averaging a 0/255 checker in gamma space gives ~128; averaging
        in linear light and re-encoding gives ~186 (0.5^(1/2.2) * 255) — the
        strongest possible discriminator between the two resize modes.
        """
        from framepump import EncoderConfig

        path = tmp_path_factory.mktemp('gamma') / 'checker.mp4'
        yy, xx = np.mgrid[0:240, 0:320]
        checker = ((yy + xx) % 2 * 255).astype(np.uint8)
        frame = np.stack([checker] * 3, axis=-1)
        with VideoWriter(str(path), fps=30, encoder_config=EncoderConfig(crf=0)) as w:
            for _ in range(4):
                w.append_data(frame)
        return str(path)

    def test_downscale_brightness(self, checker_video):
        _require_cuda()
        naive = _np(_cuda_frames(checker_video).resized((120, 160))[1]).mean()
        correct = _np(
            _cuda_frames(checker_video).resized((120, 160), gamma_correct=True)[1]
        ).mean()
        assert 112 < naive < 145, f'gamma-space mean {naive:.1f}'
        assert 170 < correct < 200, f'linear-light mean {correct:.1f}'

    def test_smooth_content_nearly_unchanged(self, smooth_video):
        _require_cuda()
        naive = _np(_cuda_frames(smooth_video).resized((120, 160))[2]).astype(np.int32)
        correct = _np(
            _cuda_frames(smooth_video).resized((120, 160), gamma_correct=True)[2]
        ).astype(np.int32)
        assert np.abs(naive - correct).mean() < 4

    def test_dtypes_and_ranges(self, smooth_video):
        _require_cuda()
        u16 = _np(_cuda_frames(smooth_video, dtype=np.uint16).resized((120, 160), gamma_correct=True)[0])
        assert u16.dtype == np.uint16 and u16.shape == (120, 160, 3) and u16.max() > 255
        f32 = _np(_cuda_frames(smooth_video, dtype=np.float32).resized((120, 160), gamma_correct=True)[0])
        assert f32.dtype == np.float32 and 0.0 <= f32.min() and f32.max() <= 1.0

    def test_hdr_transfer_rejected(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video)
        v._trc_id = 16  # pretend PQ
        with pytest.raises(NotImplementedError):
            v.resized((120, 160), gamma_correct=True)

    def test_curve_matches_reference_formula(self):
        """The GPU kernel must implement the exact IEC 61966-2-1 piecewise
        sRGB transfer (linear toe + power segment), not a plain power law."""
        torch = _require_cuda()
        from framepump._cuda.kernels import srgb_curve_inplace

        x = torch.linspace(0, 1, 65536, device='cuda', dtype=torch.float32)
        y = x.clone()
        srgb_curve_inplace(y.data_ptr(), y.numel(), decode=True)
        torch.cuda.synchronize()
        xn = x.cpu().numpy().astype(np.float64)
        ref = np.where(xn <= 0.04045, xn / 12.92, ((xn + 0.055) / 1.055) ** 2.4)
        np.testing.assert_allclose(y.cpu().numpy(), ref, atol=2e-6)

        z = torch.as_tensor(ref.astype(np.float32), device='cuda')
        srgb_curve_inplace(z.data_ptr(), z.numel(), decode=False)
        torch.cuda.synchronize()
        np.testing.assert_allclose(z.cpu().numpy(), xn, atol=3e-5)

    def test_dark_toe_uses_piecewise_curve(self, tmp_path_factory):
        """A 0/10 checker lives in the sRGB linear-toe region: piecewise
        averaging gives ~5, a pure power-2.2 law would give ~7.3."""
        _require_cuda()
        from framepump import EncoderConfig

        path = tmp_path_factory.mktemp('toe') / 'dark_checker.mp4'
        yy, xx = np.mgrid[0:240, 0:320]
        checker = ((yy + xx) % 2 * 10).astype(np.uint8)
        frame = np.stack([checker] * 3, axis=-1)
        with VideoWriter(str(path), fps=30, encoder_config=EncoderConfig(crf=0)) as w:
            for _ in range(4):
                w.append_data(frame)
        correct = _np(_cuda_frames(str(path)).resized((120, 160), gamma_correct=True)[1]).mean()
        assert 3.5 < correct < 6.3, f'toe-region mean {correct:.2f} (power law would give ~7.3)'


class TestConstantFramerate:
    VFR = str(Path(__file__).parent / 'data' / 'variable_fps.mp4')

    def test_matches_cpu_class_map(self):
        """CFR must select exactly the source frames the CPU class selects."""
        torch = _require_cuda()
        cpu = VideoFrames(self.VFR, constant_framerate=True)
        gpu_cfr = _cuda_frames(self.VFR, constant_framerate=True)
        plain = _cuda_frames(self.VFR)
        assert len(gpu_cfr) == len(cpu)
        assert gpu_cfr.fps == pytest.approx(cpu.fps)
        smap = cpu._cfr_source_map
        iterated = [torch.from_dlpack(f).clone() for f in gpu_cfr[:10]]
        for i, frame in enumerate(iterated):
            assert torch.equal(frame, torch.from_dlpack(plain[smap[i]])), f'frame {i}'
            assert torch.equal(frame, torch.from_dlpack(gpu_cfr[i])), f'getitem {i}'

    def test_numeric_target_fps(self):
        _require_cuda()
        cpu = VideoFrames(self.VFR, constant_framerate=12)
        gpu = _cuda_frames(self.VFR, constant_framerate=12)
        assert len(gpu) == len(cpu)
        assert gpu.fps == pytest.approx(12.0)

    def test_slicing_and_reverse(self):
        torch = _require_cuda()
        gpu = _cuda_frames(self.VFR, constant_framerate=True)
        full = [torch.from_dlpack(f).clone() for f in gpu[:12]]
        sliced = [torch.from_dlpack(f).clone() for f in gpu[2:12:3]]
        assert all(torch.equal(s, full[2 + 3 * i]) for i, s in enumerate(sliced))
        rev = [torch.from_dlpack(f).clone() for f in gpu[:6][::-1]]
        assert all(torch.equal(rev[i], full[5 - i]) for i in range(6))

    def test_composes_with_resize_and_repeat(self):
        torch = _require_cuda()
        v = _cuda_frames(self.VFR, constant_framerate=True).resized((120, 160))
        frame = torch.from_dlpack(v[0])
        assert tuple(frame.shape) == (120, 160, 3)
        rep = _cuda_frames(self.VFR, constant_framerate=True)[:3].repeat_each_frame(2)
        assert len(rep) == 6
        frames = [torch.from_dlpack(f).clone() for f in rep]
        assert all(torch.equal(frames[2 * i], frames[2 * i + 1]) for i in range(3))


class TestFileLikeSources:
    SHORT = str(Path(__file__).parent / 'data' / 'short.mp4')

    def test_bytesio_matches_path_based(self):
        import io

        torch = _require_cuda()
        v = _cuda_frames(io.BytesIO(Path(self.SHORT).read_bytes()))
        ref = _cuda_frames(self.SHORT)
        assert len(v) == len(ref)
        assert '<file-like>' in repr(v)
        assert torch.equal(torch.from_dlpack(v[5]), torch.from_dlpack(ref[5]))
        for a, b in zip(v[:4], ref[:4]):
            assert torch.equal(torch.from_dlpack(a).clone(), torch.from_dlpack(b).clone())

    def test_bytesio_concurrent_iterators(self):
        import io

        torch = _require_cuda()
        v = _cuda_frames(io.BytesIO(Path(self.SHORT).read_bytes()))
        for a, b in zip(v[:4], v[:4]):
            assert torch.equal(torch.from_dlpack(a).clone(), torch.from_dlpack(b).clone())

    def test_generic_file_object_sequential(self):
        _require_cuda()
        with open(self.SHORT, 'rb') as fobj:
            v = _cuda_frames(fobj)
            assert sum(1 for _ in v) == 24
            # A second full pass rewinds and works (sessions share the object)
            assert sum(1 for _ in v) == 24

    def test_bytesio_with_features(self):
        import io

        _require_cuda()
        v = _cuda_frames(io.BytesIO(Path(self.SHORT).read_bytes()), dtype=np.float32)
        frame = _np(v.resized((180, 320))[2])
        assert frame.shape == (180, 320, 3) and frame.dtype == np.float32


class TestIndexParityWithCpuClass:
    @pytest.mark.parametrize('name', ['short.mp4', 'exact_30fps.mp4', 'variable_fps.mp4'])
    def test_len_matches(self, name):
        _require_cuda()
        path = str(Path(__file__).parent / 'data' / name)
        assert len(_cuda_frames(path)) == len(VideoFrames(path))


class TestFloatDtypes:
    def test_float32_matches_uint16_scaled(self, smooth_video):
        _require_cuda()
        u16 = _np(_cuda_frames(smooth_video, dtype=np.uint16)[3]).astype(np.float64)
        f32 = _np(_cuda_frames(smooth_video, dtype=np.float32)[3])
        assert f32.dtype == np.float32
        assert 0.0 <= f32.min() and f32.max() <= 1.0
        np.testing.assert_allclose(f32, u16 / 65535.0, atol=2e-5)

    def test_float16(self, smooth_video):
        _require_cuda()
        f16 = _np(_cuda_frames(smooth_video, dtype=np.float16)[3])
        f32 = _np(_cuda_frames(smooth_video, dtype=np.float32)[3])
        assert f16.dtype == np.float16
        np.testing.assert_allclose(f16.astype(np.float32), f32, atol=1e-3)

    def test_float64_rejected(self, smooth_video):
        _require_cuda()
        with pytest.raises(NotImplementedError):
            _cuda_frames(smooth_video, dtype=np.float64)

    def test_float_iteration_matches_indexing(self, smooth_video):
        _require_cuda()
        import torch

        v = _cuda_frames(smooth_video, dtype=np.float32)[:3]
        iterated = [torch.from_dlpack(f).clone() for f in v]
        for i, frame in enumerate(iterated):
            assert torch.equal(frame, torch.from_dlpack(v[i]))

    def test_float_resized(self, smooth_video):
        _require_cuda()
        frame = _np(_cuda_frames(smooth_video, dtype=np.float32).resized((60, 80))[0])
        assert frame.shape == (60, 80, 3) and frame.dtype == np.float32
        assert 0.0 <= frame.min() and frame.max() <= 1.0


class TestSuspectCodecVerification:
    """Codecs whose containers lie about keyframes go through the CPU class's
    content-verified index, so count and frame identity match ground truth."""

    TS = str(Path(__file__).parent / 'data' / 'unreliable_seek.ts')

    def test_count_and_indexing_match_sequential_truth(self):
        torch = _require_cuda()
        cpu = VideoFrames(self.TS)
        v = _cuda_frames(self.TS)
        assert len(v) == len(cpu)
        seq = [torch.from_dlpack(f).clone() for f in v]
        assert len(seq) == len(v)
        for k in (0, 5, 9, len(v) - 1):
            assert torch.equal(torch.from_dlpack(v[k]), seq[k]), f'frame {k}'


class TestBatchGatherAndFramesAt:
    """GPU fancy indexing yields one stacked batch buffer; frames_at is lazy."""

    def test_batch_matches_singles(self, smooth_video):
        torch = _require_cuda()
        v = _cuda_frames(smooth_video)
        wanted = [2, 5, 9, 5, -1]
        resolved = [2, 5, 9, 5, 11]
        batch = torch.from_dlpack(v[wanted])
        assert tuple(batch.shape) == (5, 240, 320, 3) and batch.is_cuda
        for i, j in enumerate(resolved):
            assert torch.equal(batch[i], torch.from_dlpack(v[j])), f'slot {i}'

    def test_batch_with_float_and_resize(self, smooth_video):
        torch = _require_cuda()
        v = _cuda_frames(smooth_video, dtype=np.float32).resized((120, 160))
        batch = torch.from_dlpack(v[[0, 3]])
        assert tuple(batch.shape) == (2, 120, 160, 3) and batch.dtype == torch.float32
        assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0

    def test_invalid_indices_rejected(self, smooth_video):
        _require_cuda()
        v = _cuda_frames(smooth_video)
        with pytest.raises(TypeError):
            v[[True, False]]
        with pytest.raises(IndexError):
            v[[0, 10_000]]

    def test_frames_at_order_and_laziness(self, smooth_video):
        torch = _require_cuda()
        v = _cuda_frames(smooth_video)
        wanted = [2, 5, 11, 3, 3]
        got = [torch.from_dlpack(f).clone() for f in v.frames_at(wanted)]
        for g, j in zip(got, wanted):
            assert torch.equal(g, torch.from_dlpack(v[j]))
        gen = v.frames_at([0, 10_000])
        assert torch.from_dlpack(next(gen)) is not None
        with pytest.raises(IndexError):
            next(gen)

    def test_frames_at_repeat_rejected(self, smooth_video):
        _require_cuda()
        with pytest.raises(NotImplementedError):
            next(_cuda_frames(smooth_video).repeat_each_frame(2).frames_at([0]))
