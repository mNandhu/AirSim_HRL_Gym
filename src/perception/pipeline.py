"""Compose segmentation and detection adapters into a single perception pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .detector import YoloDetector
from .segmentation import SegmentationAdapter

__all__ = ["PerceptionPipeline"]


class PerceptionPipeline:
    def __init__(
        self,
        *,
        segmentation: SegmentationAdapter,
        detector: YoloDetector,
        enable_segmentation: bool = True,
    ) -> None:
        self._segmentation = segmentation
        self._detector = detector
        self._enable_segmentation = enable_segmentation

    def build_observation_inputs(self, raw_rgb: Any) -> dict[str, Any]:
        mask = None
        if self._enable_segmentation and self._segmentation is not None:
            try:
                mask = self._segmentation.capture_segmentation()
            except Exception:
                # If segmentation fails, proceed without it to avoid blocking the control loop
                mask = None
        rgb_array = self._to_numpy_image(raw_rgb)
        detections = self._detector.run_detection(rgb_array) if rgb_array is not None else []
        return {
            "segmentation_mask": mask,
            "detections": detections,
        }

    @staticmethod
    def _to_numpy_image(image: Any) -> np.ndarray | None:
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            if image.ndim == 2:
                return np.stack([image] * 3, axis=-1)
            return None

        raw = getattr(image, "image_data_uint8", None)
        height = getattr(image, "height", None)
        width = getattr(image, "width", None)
        if raw is None or height is None or width is None:
            return None
        array = np.frombuffer(raw, dtype=np.uint8)
        expected = height * width * 3
        if array.size == expected:
            return array.reshape(height, width, 3)
        if array.size == height * width:
            gray = array.reshape(height, width)
            return np.stack([gray] * 3, axis=-1)
        return None
