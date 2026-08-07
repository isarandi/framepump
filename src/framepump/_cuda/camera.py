"""Live camera reading: GPU-decoded frames, always the freshest ones.

A capture thread demuxes the camera (V4L2 via PyAV) and decodes every
arriving frame on the GPU (NVDEC's JPEG engine for the MJPEG wire format
that UVC cameras use at real resolutions). Decoded frames are published to
a small ring buffer from which consumers only ever receive frames they have
not seen before; frames the consumer was too slow for are dropped rather
than queued (measured: naive queueing reaches seconds of staleness within
moments — see explorations/2026-08-07-webcam-poc).
"""

from __future__ import annotations

import bisect
import ctypes
import operator
import threading

import av
import PyNvVideoCodec as nvc

from .._pyav import VideoDecodeError
from .compat import cuda_ctx_pushed, retain_primary_context
from .dlpack import _GpuRgbBuffer


class CameraFrames:
    """Iterator over live camera frames as GPU-resident RGB buffers.

    Live semantics, deliberately different from the file readers: there is
    no ``len()``, no seeking and no slicing — only iteration, and each step
    yields the *latest* captured frame (frames the consumer was too slow for
    are dropped, keeping latency at one frame interval instead of a growing
    queue). Yielded frames support ``__dlpack__`` for zero-copy import into
    PyTorch/CuPy and stay valid until the next iteration step; ``.clone()``
    the tensor to keep it longer.

    For models that only reach real-time throughput when batching,
    :meth:`batched` yields adaptive batches of all frames that arrived
    since the previous step instead of just the latest one.

    Linux/V4L2 only for now; requires an MJPEG-capable camera (the format
    UVC cameras use at real resolutions — essentially all of them).

    Example:
        >>> with CameraFrames('/dev/video0', shape=(720, 1280), fps=30) as cam:
        ...     for frame in cam:
        ...         tensor = torch.from_dlpack(frame)  # (H, W, 3) uint8 CUDA
        ...         if done:
        ...             break

    Args:
        device: V4L2 device path, e.g. ``'/dev/video0'``.
        shape: Requested capture size as (height, width), or None for the
            camera's default. The camera may negotiate a different size;
            check ``imshape``.
        fps: Requested frame rate, or None for the camera's default. The
            camera may clamp it; check ``fps``.
        gpu: GPU device ordinal for decoding.
    """

    def __init__(
        self,
        device: str = '/dev/video0',
        *,
        shape: tuple[int, int] | None = None,
        fps: float | None = None,
        gpu: int = 0,
    ) -> None:
        self.device = device
        self._gpu = gpu
        options = {'input_format': 'mjpeg'}
        if shape is not None:
            options['video_size'] = f'{shape[1]}x{shape[0]}'
        if fps is not None:
            options['framerate'] = str(fps)
        self._container = av.open(device, format='v4l2', options=options)
        self._stream = self._container.streams.video[0]
        codec_name = self._stream.codec_context.codec.name
        if codec_name != 'mjpeg':
            self._container.close()
            raise VideoDecodeError(
                device, 0, RuntimeError(f'Camera negotiated {codec_name!r}, expected mjpeg')
            )
        self.imshape: tuple[int, int] = (self._stream.height, self._stream.width)
        self.fps: float = float(self._stream.average_rate or 0.0)
        self._time_base = float(self._stream.time_base)

        h, w = self.imshape
        self._row_bytes = w * 3
        self._bufs: list[int] = []
        # Published, not-yet-delivered frames as (buffer index, capture time),
        # oldest first; the writer trims it to _max_batch entries.
        self._queue: list[tuple[int, float]] = []
        # Buffer delivered zero-copy in the last step (still readable by the
        # consumer's tensor) — the writer must not touch it.
        self._held: list[int] = []
        self._max_batch = 1
        self._queue_depth = 1
        self._last_delivered_ts: float | None = None
        self._batch_bufs: list[int] = []
        self._batch_capacity = 0
        self._i_batch = 0
        self._n_delivered = 0
        self.last_capture_time: float | None = None
        self.last_capture_times: list[float] | None = None
        self._cond = threading.Condition()
        self._error: BaseException | None = None
        self._closed = False

        self._device_handle, self._ctx = retain_primary_context(gpu)
        try:
            self._dec = nvc.CreateDecoder(
                gpuid=gpu,
                codec=nvc.cudaVideoCodec.JPEG,
                cudacontext=int(self._ctx),
                cudastream=0,
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGB,
                latency=nvc.DisplayDecodeLatencyType.LOW,
            )
        except BaseException:
            self._release()
            raise

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ── Capture thread ───────────────────────────────────────────────

    def _capture_loop(self) -> None:
        try:
            with self._cond:
                # One buffer per retained-history slot, one being written, one
                # spare (in single-frame mode: the zero-copy-held one).
                self._ensure_bufs_locked(self._queue_depth + 2)
            for packet in self._container.demux(self._stream):
                if self._closed:
                    return
                data = bytes(packet)
                if not data:
                    continue
                buf = ctypes.create_string_buffer(data, len(data))
                pd = nvc.PacketData()
                pd.bsl_data = ctypes.addressof(buf)
                pd.bsl = len(data)
                pd.pts = packet.pts if packet.pts is not None else 0
                frames = self._dec.Decode(pd)
                for frame in frames:
                    self._store_latest(frame, pd.pts * self._time_base)
        except BaseException as e:  # noqa: BLE001 — surfaced on next()
            with self._cond:
                self._error = e
                self._cond.notify_all()

    def _ensure_bufs_locked(self, count: int) -> None:
        """Grow the frame ring to `count` buffers. Call with the lock held."""
        from cuda.bindings import driver

        if len(self._bufs) >= count:
            return
        frame_bytes = self._row_bytes * self.imshape[0]
        with cuda_ctx_pushed(self._ctx):
            while len(self._bufs) < count:
                err, ptr = driver.cuMemAlloc(frame_bytes)
                if err != driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f'Failed to allocate camera buffer: {err}')
                self._bufs.append(int(ptr))

    def _store_latest(self, frame, capture_ts: float) -> None:
        """Copy the decoded surface into a free buffer and publish it."""
        from cuda.bindings import driver

        with self._cond:
            # Any buffer that is neither queued, nor held by the consumer, is
            # free; the ring is sized so at least one always exists, and the
            # consumer never touches free buffers, so frames cannot tear.
            queued = {i for i, _ in self._queue}
            back = next(
                i for i in range(len(self._bufs)) if i not in queued and i not in self._held
            )
        with cuda_ctx_pushed(self._ctx):
            views = frame.cuda()
            view = views[0] if isinstance(views, (list, tuple)) else views
            cai = view.__cuda_array_interface__
            strides = cai.get('strides')
            copy = driver.CUDA_MEMCPY2D()
            copy.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
            copy.srcDevice = cai['data'][0]
            copy.srcPitch = strides[0] if strides else self._row_bytes
            copy.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
            copy.dstDevice = self._bufs[back]
            copy.dstPitch = self._row_bytes
            copy.WidthInBytes = self._row_bytes
            copy.Height = self.imshape[0]
            (err,) = driver.cuMemcpy2D(copy)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Failed to copy camera frame: {err}')
        with self._cond:
            self._queue.append((back, capture_ts))
            if len(self._queue) > self._queue_depth:
                del self._queue[0]  # history cap reached: drop the oldest
            self._cond.notify_all()

    # ── Consumer side ────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> _GpuRgbBuffer:
        """The latest captured frame (waits for a fresh one if needed)."""
        with self._cond:
            self._wait_for_frames_locked()
            idx, ts = self._queue[-1]
            self._queue.clear()
            self._held = [idx]
            self.last_capture_time = ts
            self.last_capture_times = [ts]
            self._last_delivered_ts = ts
            self._n_delivered += 1
        h, w = self.imshape
        return _GpuRgbBuffer(
            self._bufs[idx], h, w, self._row_bytes, self._gpu,
            owns_memory=False, bits=8, code=1,
        )  # fmt: skip

    def batched(self, max_batch_size: int, history: int | None = None):
        """Iterate over adaptive batches that evenly cover the unseen interval.

        Each step waits until at least one new frame exists, then yields up
        to ``max_batch_size`` frames stacked into one GPU buffer of shape
        ``(k, height, width, 3)``, chronological and always ending with the
        newest frame. The frames are chosen by dividing the time since the
        previously delivered frame into ``max_batch_size`` equal steps and
        taking the retained frame nearest to each step — so a slow consumer
        receives frames spread evenly across the whole interval it missed,
        not a burst of near-identical latest frames. Selections that land
        on the same frame collapse, so a consumer that keeps up with the
        camera simply gets ``k = 1`` each step. No frame is ever delivered
        twice; undelivered frames between the selected ones are dropped.
        The capture timestamps of the delivered frames are in
        ``last_capture_times``.

        The camera retains at most ``history`` undelivered frames (default:
        two seconds' worth), bounding both memory (height x width x 3 bytes
        per retained frame) and the interval that can be covered; a
        consumer away longer than that gets frames spread evenly over the
        retained history.

        Like single-frame iteration, the yielded buffer is reused and only
        valid until the next step; import via DLPack and use (or
        ``.clone()``) it before continuing.

        Note: runtimes that specialize per input shape (TorchScript JIT,
        cudnn benchmark mode) recompile for each new batch size — warm up
        every size from 1 to ``max_batch_size`` to avoid mid-stream stalls.
        """
        max_batch_size = operator.index(max_batch_size)
        if max_batch_size < 1:
            raise ValueError('max_batch_size must be at least 1')
        if history is None:
            history = round(2 * self.fps) if self.fps > 0 else 60
        history = max(operator.index(history), max_batch_size)
        from cuda.bindings import driver

        frame_bytes = self._row_bytes * self.imshape[0]
        with self._cond:
            if self._error is not None:
                raise VideoDecodeError(self.device, self._n_delivered, self._error)
            self._ensure_bufs_locked(history + 2)
            if self._batch_capacity < max_batch_size:
                # Ping-pong pair of stacked buffers (reused every other step)
                with cuda_ctx_pushed(self._ctx):
                    for ptr in self._batch_bufs:
                        driver.cuMemFree(ptr)
                    self._batch_bufs = []
                    for _ in range(2):
                        err, ptr = driver.cuMemAlloc(frame_bytes * max_batch_size)
                        if err != driver.CUresult.CUDA_SUCCESS:
                            raise RuntimeError(f'Failed to allocate batch buffer: {err}')
                        self._batch_bufs.append(int(ptr))
                self._batch_capacity = max_batch_size
            self._max_batch = max_batch_size
            self._queue_depth = history

        def gen():
            while True:
                try:
                    yield self._next_batch()
                except StopIteration:
                    return

        return gen()

    def _select_evenly_locked(self) -> list[tuple[int, float]]:
        """Pick up to _max_batch queued frames evenly covering the unseen
        time interval, newest included. Call with the lock held."""
        entries = self._queue
        ts_list = [ts for _, ts in entries]
        t_now = ts_list[-1]
        # The unseen interval starts at the previously delivered frame, or —
        # if the consumer was away longer than the retained history (or on
        # the first call) — just before the oldest retained frame, so the
        # subdivision then spreads over what was retained.
        if len(ts_list) > 1:
            interval = (ts_list[-1] - ts_list[0]) / (len(ts_list) - 1)
        else:
            interval = 1 / self.fps if self.fps > 0 else 0.033
        anchor = ts_list[0] - interval
        if self._last_delivered_ts is not None:
            anchor = max(anchor, self._last_delivered_ts)
        gap = t_now - anchor
        k = self._max_batch
        selected_idxs: list[int] = []
        for i in range(1, k + 1):
            target = anchor + i * gap / k
            j = bisect.bisect_left(ts_list, target)
            if j == 0:
                best = 0
            elif j >= len(ts_list):
                best = len(ts_list) - 1
            else:
                best = j if ts_list[j] - target < target - ts_list[j - 1] else j - 1
            if not selected_idxs or best != selected_idxs[-1]:
                selected_idxs.append(best)
        return [entries[i] for i in selected_idxs]

    def _next_batch(self) -> _GpuRgbBuffer:
        from cuda.bindings import driver

        h, w = self.imshape
        frame_bytes = self._row_bytes * h
        with self._cond:
            self._wait_for_frames_locked()
            selected = self._select_evenly_locked()
            self._queue.clear()
            self._held = []
            self._i_batch ^= 1
            dst_base = self._batch_bufs[self._i_batch]
            # Gather under the lock: the writer cannot recycle the source
            # buffers while we hold it, and k short device copies block it
            # for far less than a frame interval.
            with cuda_ctx_pushed(self._ctx):
                for i, (idx, _) in enumerate(selected):
                    (err,) = driver.cuMemcpyDtoD(
                        dst_base + i * frame_bytes, self._bufs[idx], frame_bytes
                    )
                    if err != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f'Failed to gather camera batch: {err}')
            self.last_capture_times = [ts for _, ts in selected]
            self.last_capture_time = self.last_capture_times[-1]
            self._last_delivered_ts = self.last_capture_time
            self._n_delivered += len(selected)
        return _GpuRgbBuffer(
            dst_base, h, w, self._row_bytes, self._gpu,
            owns_memory=False, bits=8, code=1, batch=len(selected),
        )  # fmt: skip

    def _wait_for_frames_locked(self) -> None:
        """Wait until the queue is non-empty. Call with the lock held."""
        while not self._queue and self._error is None:
            if self._closed:
                raise StopIteration
            self._cond.wait(timeout=5.0)
            if not self._queue and self._error is None and not self._closed:
                raise VideoDecodeError(
                    self.device, self._n_delivered,
                    RuntimeError('No frame arrived from the camera within 5 s'),
                )  # fmt: skip
        if self._error is not None:
            raise VideoDecodeError(self.device, self._n_delivered, self._error)

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop capturing and release the camera and GPU resources."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        # The capture loop checks the flag on every packet (the camera
        # delivers within a frame interval), so it exits promptly; closing
        # the container from here only as a fallback if it is stuck.
        self._thread.join(timeout=2.0)
        self._container.close()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release()

    def _release(self) -> None:
        from cuda.bindings import driver

        if getattr(self, '_dec', None) is not None:
            with cuda_ctx_pushed(self._ctx):
                self._dec = None
        if self._bufs or self._batch_bufs:
            with cuda_ctx_pushed(self._ctx):
                for ptr in self._bufs + self._batch_bufs:
                    driver.cuMemFree(ptr)
            self._bufs = []
            self._batch_bufs = []
        if self._device_handle is not None:
            driver.cuDevicePrimaryCtxRelease(self._device_handle)
            self._device_handle = None
            self._ctx = None

    def __enter__(self) -> CameraFrames:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not getattr(self, '_closed', True):
                self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        h, w = self.imshape
        return f"CameraFrames('{self.device}', {w}x{h}, {self.fps:.4g} fps)"
