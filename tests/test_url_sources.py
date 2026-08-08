"""Reading videos from URLs (served over HTTP by a local test server)."""

import functools
import http.server
import os
import re
import threading

import numpy as np
import pytest

import framepump
from framepump._pyav import is_url

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class _RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with HTTP Range support.

    FFmpeg needs range requests to seek (mp4 files keep their index in the
    moov atom, often at the end); real servers (nginx, S3, ...) support them,
    Python's stock handler does not.
    """

    def send_head(self):
        range_header = self.headers.get('Range')
        if range_header is None:
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', range_header.strip())
        path = self.translate_path(self.path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
        end = min(end, size - 1)
        if start >= size:
            f.close()
            self.send_error(416)
            return None
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        f.seek(start)
        return _FileSlice(f, end - start + 1)

    def log_message(self, *args):
        pass


class _FileSlice:
    """File wrapper that stops reading after `remaining` bytes."""

    def __init__(self, f, remaining):
        self.f = f
        self.remaining = remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b''
        n = self.remaining if n < 0 else min(n, self.remaining)
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


@pytest.fixture(scope='module')
def http_url_base():
    handler = functools.partial(_RangeHTTPRequestHandler, directory=DATA_DIR)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()


def test_is_url():
    assert is_url('http://example.com/a.mp4')
    assert is_url('rtsp://camera.local/stream')
    assert not is_url('video.mp4')
    assert not is_url('/abs/path/video.mp4')
    assert not is_url('C://not/a/url')  # Windows drive path
    assert not is_url(b'http://bytes')


def test_http_matches_file(http_url_base):
    v_url = framepump.VideoFrames(f'{http_url_base}/exact_30fps.mp4')
    v_file = framepump.VideoFrames(os.path.join(DATA_DIR, 'exact_30fps.mp4'))
    assert len(v_url) == len(v_file)
    for a, b in zip(v_url, v_file):
        assert np.array_equal(a, b)


def test_http_indexing_and_slicing(http_url_base):
    v_url = framepump.VideoFrames(f'{http_url_base}/exact_30fps.mp4')
    v_file = framepump.VideoFrames(os.path.join(DATA_DIR, 'exact_30fps.mp4'))
    assert np.array_equal(v_url[17], v_file[17])
    assert np.array_equal(v_url[-1], v_file[-1])
    subset_url = [f.mean() for f in v_url[5:20:3]]
    subset_file = [f.mean() for f in v_file[5:20:3]]
    assert subset_url == subset_file


def test_http_info(http_url_base):
    url = f'{http_url_base}/exact_30fps.mp4'
    info = framepump.VideoFrames(url).info
    assert info.source == url
    assert info.codec == 'h264'
    assert info.imshape == (720, 1280)


def test_missing_file_still_raises():
    with pytest.raises(FileNotFoundError):
        framepump.VideoFrames('definitely_not_here.mp4')


def test_http_gpu(http_url_base):
    """VideoFramesCuda over HTTP (same demux path as CPU, NVDEC decode)."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    pytest.importorskip('PyNvVideoCodec')
    v_url = framepump.VideoFramesCuda(f'{http_url_base}/exact_30fps.mp4')
    v_file = framepump.VideoFrames(os.path.join(DATA_DIR, 'exact_30fps.mp4'))
    assert len(v_url) == len(v_file)
    t = torch.from_dlpack(v_url[10])
    assert t.is_cuda and tuple(t.shape) == (720, 1280, 3)
