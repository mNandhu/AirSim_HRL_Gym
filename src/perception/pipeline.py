"""Compose segmentation and detection adapters into a single perception pipeline."""

from __future__ import annotations

from typing import Any

from .detector import YoloDetector
from .segmentation import SegmentationAdapter

__all__ = ["PerceptionPipeline"]


class PerceptionPipeline:
    def __init__(self, *, segmentation: SegmentationAdapter, detector: YoloDetector) -> None:
        self._segmentation = segmentation
        self._detector = detector

    def build_observation_inputs(self, raw_rgb: Any) -> dict[str, Any]:
        mask = self._segmentation.capture_segmentation()
        detections = self._detector.run_detection(raw_rgb)
        return {
            "segmentation_mask": mask,
            "detections": detections,
        }
