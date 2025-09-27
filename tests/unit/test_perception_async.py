from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pytest

from perception.async_pipeline import AsyncPerceptionPipeline


class DummySegmentation:
    def __init__(self) -> None:
        self.calls = 0

    def capture_segmentation(self) -> np.ndarray:
        self.calls += 1
        return np.ones((2, 2), dtype=np.int32) * self.calls


class DummyDetector:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def run_detection(self, image: Any) -> list[dict[str, Any]]:
        frame_id = int(np.mean(image)) if isinstance(image, np.ndarray) else -1
        self.frames.append(frame_id)
        return [{"class_id": 0, "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": float(frame_id)}]


def test_async_pipeline_processes_latest_frame() -> None:
    segmentation = DummySegmentation()
    detector = DummyDetector()
    pipeline = AsyncPerceptionPipeline(
        segmentation=cast(Any, segmentation),
        detector=cast(Any, detector),
        max_queue_size=1,
        poll_interval=0.001,
    )

    try:
        frame0 = np.zeros((4, 4, 3), dtype=np.uint8)
        pipeline.build_observation_inputs(frame0)
        time.sleep(0.02)

        frame1 = np.ones_like(frame0) * 10
        pipeline.build_observation_inputs(frame1)
        frame2 = np.ones_like(frame0) * 20
        pipeline.build_observation_inputs(frame2)

        time.sleep(0.05)
        result = pipeline.build_observation_inputs(None)

        assert segmentation.calls >= 4
        assert detector.frames[-1] == 20
        assert result["detections"][0]["confidence"] == pytest.approx(20.0)
    finally:
        pipeline.close()
