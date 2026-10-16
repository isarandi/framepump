"""Freeing a frame-threaded MPEG-4 decoder after a mid-stream decode error
must not deadlock (avcodec_free_context -> ff_frame_thread_free ->
pthread_cond_wait). The hazard is a GIL inversion that needs PyAV's Python
log callback to be active (av.logging.set_level(...), which e.g. torchvision
calls at import time): the codec context is freed from a Cython dealloc with
the GIL held, waiting for the frame-thread workers, while a worker waits for
the GIL inside the log callback because the damaged stream makes it log
decoder errors. _pyav._flush_decoder() (GIL released) drains the workers
before every container close / reopen / __del__.

Synthetically damaged files (stray start codes, random bytes, zeros, bit
flips, stream-copied clips) do not trigger the deadlock: it needs a real
recording's in-frame bitstream damage (enough worker-thread log traffic at
teardown). So this test uses a real damaged file from a local path (env
FRAMEPUMP_CORRUPT_MPEG4, or a known sample path) and is skipped where that is
unavailable. It is sensitive there: with _flush_decoder disabled the scenario
hangs. The scenario runs in a subprocess with a timeout so a regression fails
instead of hanging the suite."""
import os
import subprocess
import sys
import textwrap

import pytest

DEFAULT_SAMPLE = '/data/freeman/videos/20221012_ebbdbef707_subj19/vframes/c08.mp4'

SCENARIO = textwrap.dedent('''
    import gc, sys
    import av
    av.logging.set_level(av.logging.ERROR)  # the trigger (torchvision does this at import)
    import framepump
    path = sys.argv[1]
    vf = framepump.VideoFrames(path)
    n = len(vf)
    print('len', n, flush=True)
    # an indexed read whose frame-thread read-ahead crosses the damage: the
    # decoder errors, framepump reopens, and the OLD context is freed there
    for i in (int(n * 0.25), int(n * 0.5), int(n * 0.75)):
        try:
            vf[i]
        except Exception:
            pass
    print('reads done', flush=True)
    del vf
    gc.collect()
    print('freed OK', flush=True)
''')


@pytest.mark.timeout(240)
def test_indexed_reads_on_damaged_mpeg4_do_not_hang():
    path = os.environ.get('FRAMEPUMP_CORRUPT_MPEG4', DEFAULT_SAMPLE)
    if not os.path.exists(path):
        pytest.skip(f'no damaged MPEG-4 sample at {path} (set FRAMEPUMP_CORRUPT_MPEG4)')
    try:
        r = subprocess.run([sys.executable, '-c', SCENARIO, path], capture_output=True, text=True, timeout=150)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b'').decode()
        pytest.fail(f'HANG: frame-threaded decoder teardown deadlock regression. stdout so far: {out}')
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'freed OK' in r.stdout, r.stdout
