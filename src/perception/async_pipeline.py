"""Asynchronous perception pipeline for handling multiple simulators."""

from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from .detector import YoloDetector
from .pipeline import PerceptionPipeline
from .segmentation import SegmentationAdapter

__all__ = ["AsyncPerceptionPipeline"]


class AsyncPerceptionPipeline:
    """Wrap segmentation and detection with a background worker thread.

    The asynchronous variant maintains a small queue of RGB frames. New frames are
    added during :meth:`build_observation_inputs`, and a worker thread consumes
    the queue to perform detector inference without blocking the main control
    loop. When the queue is full, older frames are dropped, effectively
    introducing frame skipping under load, which allows the perception subsystem
    to keep up with multiple concurrent simulators.
    """

    def __init__(
        self,
        *,
        segmentation: SegmentationAdapter,
        detector: YoloDetector,
        enable_segmentation: bool = True,
        max_queue_size: int = 2,
        poll_interval: float = 0.01,
    ) -> None:
        self._segmentation = segmentation
        self._detector = detector
        self._enable_segmentation = enable_segmentation
        self._queue: Queue[np.ndarray] = Queue(maxsize=max_queue_size)
        self._latest_detections: list[dict[str, Any]] = []
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_interval = poll_interval

        self._worker = threading.Thread(target=self._run, name="perception-worker", daemon=True)
        self._worker.start()

    def build_observation_inputs(self, raw_rgb: Any) -> dict[str, Any]:
        mask = None
        if self._enable_segmentation and self._segmentation is not None:
            try:
                mask = self._segmentation.capture_segmentation()
            except Exception:
                mask = None
        frame = PerceptionPipeline._to_numpy_image(raw_rgb)
        if frame is not None:
            self._submit_frame(frame)
        detections = self._get_latest_detections()
        return {
            "segmentation_mask": mask,
            "detections": detections,
        }

    def close(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        with self._latest_lock:
            self._latest_detections = []

    def _submit_frame(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except Full:
            self._drain_queue()
            self._queue.put_nowait(frame)

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except Empty:
            return

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=self._poll_interval)
            except Empty:
                continue
            try:
                detections = self._detector.run_detection(frame)
            except Exception:
                detections = []
            with self._latest_lock:
                self._latest_detections = detections
            self._queue.task_done()

    def _get_latest_detections(self) -> list[dict[str, Any]]:
        with self._latest_lock:
            return list(self._latest_detections)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
