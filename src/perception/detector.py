"""YOLO detection wrapper using the ``ultralytics`` package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency when torch unavailable
    import torch  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency when ultralytics unavailable
    from ultralytics import YOLO  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

__all__ = ["ModelLoadError", "YoloDetector"]


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model fails to load."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    bbox: list[float]
    confidence: float


class YoloDetector:
    """Wrapper around a YOLO model loaded via ``ultralytics``."""

    def __init__(
        self,
        model_name: str = "yolov12n",
        *,
        model: Any | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        else:
            if YOLO is None:
                raise ModelLoadError("ultralytics is required to load YOLO models")
            try:
                self._model = YOLO(model_name)
            except Exception as exc:  # pragma: no cover - network errors
                raise ModelLoadError(f"Failed to load YOLO model {model_name!r}: {exc}") from exc
        self._model_name = model_name

    def run_detection(self, image: Any) -> list[dict[str, Any]]:
        if image is None:
            return []
        try:
            predictions = self._model(image, verbose=False)
        except TypeError as exc:
            message = str(exc).lower()
            if "unexpected keyword" in message and "verbose" in message:
                predictions = self._model(image)
            elif "got an unexpected keyword argument" in message and "verbose" in message:
                predictions = self._model(image)
            else:
                raise
        if not predictions:
            return []
        first = predictions[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            return []
        xywh = self._to_numpy(getattr(boxes, "xywhn", []))
        classes = self._to_numpy(getattr(boxes, "cls", []))
        confidences = self._to_numpy(getattr(boxes, "conf", []))
        detections: list[dict[str, Any]] = []
        for bbox, cls, conf in zip(xywh, classes, confidences, strict=True):
            detections.append(
                {
                    "class_id": int(cls),
                    "bbox": [float(coord) for coord in bbox.tolist()],
                    "confidence": float(conf),
                }
            )
        return detections

    def build_observation_inputs(
        self, raw_rgb: Any, *, segmentation_adapter: Any
    ) -> dict[str, Any]:
        mask = segmentation_adapter.capture_segmentation()
        detections = self.run_detection(raw_rgb)
        return {"segmentation_mask": mask, "detections": detections}

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch is not None and hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if torch is not None and hasattr(value, "cpu"):
            return value.cpu().numpy()
        return np.asarray(value)
