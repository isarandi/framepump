"""B-frame correctness tests for the NVENC encoders and GLVideoWriter.

These tests content-verify encoded output (every frame carries its index in
the red channel), so frame duplication, reordering errors, or stale-input
corruption fail loudly. They cover the GL (GLX) and CUDA encoder paths, IDR
placement, flush idempotence, and the ring-size/config invariant.
"""

import os
import subprocess
import sys

import numpy as np
import pytest


def _nvenc_available():
    """Library-presence probe only: no CUDA/GL initialization at collection."""
    try:
        import ctypes

        ctypes.CDLL('libnvidia-encode.so.1')
        import glfw  # noqa: F401
        import moderngl  # noqa: F401

        return True
    except Exception:
        return False


NVENC_AVAILABLE = _nvenc_available()
pytestmark = pytest.mark.skipif(
    not NVENC_AVAILABLE, reason='Requires NVIDIA GPU with NVENC support'
)

_glfw_initialized = False


@pytest.fixture
def gl_context():
    """Headless GLX context on the NVIDIA GPU (PRIME offload set in conftest)."""
    global _glfw_initialized
    import glfw
    import moderngl

    if not _glfw_initialized:
        if not glfw.init():
            pytest.skip('Failed to initialize GLFW')
        _glfw_initialized = True

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, 'test', None, None)
    if not window:
        pytest.skip('Failed to create GLFW window')
    glfw.make_context_current(window)

    import ctypes

    gl = ctypes.cdll.LoadLibrary('libGL.so.1')
    gl.glGetString.restype = ctypes.c_char_p
    renderer = (gl.glGetString(0x1F01) or b'').decode(errors='replace')
    if 'nvidia' not in renderer.lower():
        glfw.destroy_window(window)
        pytest.skip(f'GL context is on non-NVIDIA GPU: {renderer}')

    ctx = moderngl.create_context()
    yield ctx
    glfw.destroy_window(window)


def expected_red(i):
    return (i * 37) % 200 + 20


def make_frame(i, width, height):
    arr = np.zeros((height, width, 4), np.uint8)
    arr[..., 0] = expected_red(i)
    arr[..., 1] = 128
    arr[..., 2] = (i * 11) % 256
    arr[..., 3] = 255
    x = (i * 3) % (width - 10)
    arr[:, x : x + 10, :3] = 255
    return arr


def decode_annexb(data):
    """Decode a raw Annex-B H.264 bitstream to RGB frames in display order."""
    import io

    import av

    frames = []
    with av.open(io.BytesIO(data), format='h264') as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
    return frames


def check_frames(frames, n_expected, tol=15):
    assert len(frames) == n_expected
    for i, frame in enumerate(frames):
        med = float(np.median(frame[..., 0]))
        exp = expected_red(i)
        assert abs(med - exp) <= tol, f'frame {i}: median red {med:.0f}, expected {exp}'


def encode_moving_frames(encoder, gl_context, n_frames, width, height):
    """Encode n frames through the same repeatedly-rewritten texture."""
    texture = gl_context.texture((width, height), 4)
    packets = []
    for i in range(n_frames):
        texture.write(make_frame(i, width, height).tobytes())
        gl_context.finish()
        packets.extend(encoder.encode(texture))
    packets.extend(encoder.flush())
    return packets


def make_encoder(kind, width, height, **kwargs):
    if kind == 'gl':
        from framepump.nvenc import NvencEncoder

        return NvencEncoder(width, height, **kwargs)
    from framepump.nvenc import NvencCudaEncoder

    return NvencCudaEncoder(width, height, **kwargs)


@pytest.mark.parametrize('kind', ['gl', 'cuda'])
@pytest.mark.parametrize('bframes', [0, 2])
def test_encoder_content_roundtrip(gl_context, kind, bframes):
    """Same-texture-rewrite pattern must round-trip content at bframes 0 and 2."""
    n = 60
    w, h = 320, 240
    with make_encoder(kind, w, h, fps=30, bframes=bframes) as encoder:
        packets = encode_moving_frames(encoder, gl_context, n, w, h)

    assert sorted(p.pts for p in packets) == list(range(n))
    dts = [p.dts for p in packets]
    assert dts == sorted(dts)
    assert packets[0].is_keyframe

    frames = decode_annexb(b''.join(p.data for p in packets))
    check_frames(frames, n)


@pytest.mark.parametrize('kind', ['gl', 'cuda'])
def test_idr_period_respected(gl_context, kind):
    """gop=N must place IDR frames every N frames, not at the preset default."""
    n, gop = 60, 25
    w, h = 320, 240
    with make_encoder(kind, w, h, fps=30, gop=gop, bframes=2) as encoder:
        packets = encode_moving_frames(encoder, gl_context, n, w, h)

    keyframe_pts = sorted(p.pts for p in packets if p.is_keyframe)
    assert keyframe_pts == [0, 25, 50], f'IDR placement wrong: {keyframe_pts}'

    frames = decode_annexb(b''.join(p.data for p in packets))
    check_frames(frames, n)


@pytest.mark.parametrize('kind', ['gl', 'cuda'])
def test_flush_idempotent(gl_context, kind):
    """flush() must be safely re-callable and not lose or duplicate packets."""
    n = 10
    w, h = 320, 240
    with make_encoder(kind, w, h, fps=30, bframes=2) as encoder:
        packets = encode_moving_frames(encoder, gl_context, n, w, h)
        assert encoder.flush() == []
        assert encoder.flush() == []
    assert len(packets) == n


@pytest.mark.parametrize('kind', ['gl', 'cuda'])
def test_ring_derived_from_config(gl_context, kind):
    """Buffer rings must be sized from the finalized encoder config."""
    with make_encoder(kind, 320, 240, fps=30, bframes=2) as encoder:
        session = encoder._session
        config = session._config
        lookahead_enabled = bool(config.rcParams.rcFlags & (1 << 5))
        assert not lookahead_enabled, 'lookahead must be explicitly disabled'
        expected = config.frameIntervalP + config.rcParams.lookaheadDepth + 1
        assert session._ring_size == expected
        assert len(session._bitstream_buffers) == expected
        assert config.encodeCodecConfig.h264Config.idrPeriod == config.gopLength


@pytest.mark.parametrize('bframes', [0, 2])
def test_glvideowriter_moving_content(gl_context, tmp_path, bframes):
    """End-to-end: GLVideoWriter output must contain every frame's content."""
    from framepump import EncoderConfig, GLVideoWriter, VideoFrames

    n = 60
    w, h = 320, 240
    out = tmp_path / f'moving_b{bframes}.mp4'
    texture = gl_context.texture((w, h), 4)

    with GLVideoWriter(encoder_config=EncoderConfig(bframes=bframes)) as writer:
        writer.start_sequence(str(out), fps=30)
        for i in range(n):
            texture.write(make_frame(i, w, h).tobytes())
            gl_context.finish()
            writer.append_data(texture)
        writer.end_sequence()

    frames = list(VideoFrames(str(out)))
    check_frames(frames, n)


_HEADLESS_SCRIPT = r"""
import sys
import numpy as np
import moderngl
from framepump import EncoderConfig, GLVideoWriter, VideoFrames

def expected_red(i):
    return (i * 37) % 200 + 20

n, w, h = 40, 320, 240
ctx = moderngl.create_context(standalone=True, backend='egl')
texture = ctx.texture((w, h), 4)
out = sys.argv[1]

with GLVideoWriter(encoder_config=EncoderConfig(bframes=2)) as writer:
    writer.start_sequence(out, fps=30)
    for i in range(n):
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., 0] = expected_red(i)
        arr[..., 1] = 128
        arr[..., 3] = 255
        x = (i * 3) % (w - 10)
        arr[:, x:x + 10, :3] = 255
        texture.write(arr.tobytes())
        ctx.finish()
        writer.append_data(texture)
    writer.end_sequence()

frames = list(VideoFrames(out))
assert len(frames) == n, f'{len(frames)} != {n}'
for i, f in enumerate(frames):
    med = float(np.median(f[..., 0]))
    exp = expected_red(i)
    assert abs(med - exp) <= 15, f'frame {i}: {med} vs {exp}'
print('HEADLESS_OK')
"""


def test_headless_cuda_path(tmp_path):
    """The headless (EGL + CUDA encoder) path must round-trip moving content."""
    script = tmp_path / 'headless_encode.py'
    script.write_text(_HEADLESS_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k != 'DISPLAY'}
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / 'headless.mp4')],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if 'Failed to initialize' in result.stderr or 'egl' in result.stderr.lower():
        if result.returncode != 0 and 'HEADLESS_OK' not in result.stdout:
            pytest.skip(f'EGL headless context unavailable: {result.stderr[-500:]}')
    assert result.returncode == 0, f'stderr: {result.stderr[-2000:]}'
    assert 'HEADLESS_OK' in result.stdout
