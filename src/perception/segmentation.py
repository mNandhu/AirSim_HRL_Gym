"""Segmentation adapter for retrieving AirSim ground-truth masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import airsim
except ImportError:  # pragma: no cover - fallback for local testing
    airsim = None  # type: ignore

__all__ = ["SegmentationAdapter", "SegmentationCaptureError"]


class SegmentationCaptureError(RuntimeError):
    """Raised when a segmentation mask cannot be retrieved."""


@dataclass(frozen=True)
class _ImageRequest:
    camera_name: str
    image_type: int
    pixels_as_float: bool = False
    compress: bool = False


class SegmentationAdapter:
    """Retrieve segmentation masks from an AirSim client."""

    def __init__(self, client: Any, camera_name: str, image_type: int | None = None) -> None:
        self._client = client
        self._camera_name = camera_name
        if image_type is not None:
            self._image_type = image_type
        elif airsim is not None:
            if hasattr(airsim.ImageType, "Segmentation"):
                self._image_type = airsim.ImageType.Segmentation
            else:  # pragma: no cover - legacy AirSim fallback
                self._image_type = 5
        else:
            self._image_type = 5  # AirSim default enum value for segmentation

    def capture_segmentation(self) -> np.ndarray:
        request = self._create_request()
        response = self._client.simGetImages([request])
        if not response:
            raise SegmentationCaptureError("AirSim returned no segmentation images")
        image = response[0]
        height = getattr(image, "height", None)
        width = getattr(image, "width", None)
        raw = getattr(image, "image_data_uint8", None)
        if height is None or width is None or raw is None:
            raise SegmentationCaptureError("Segmentation response missing required fields")
        array = np.frombuffer(raw, dtype=np.uint8)
        expected = height * width
        if array.size == expected:
            return array.reshape(height, width)

        rgb_expected = expected * 3
        if array.size == rgb_expected:
            rgb = array.reshape(height, width, 3).astype(np.uint32)
            mask = rgb[:, :, 0] + (rgb[:, :, 1] << 8) + (rgb[:, :, 2] << 16)
            return mask.astype(np.int32)

        raise SegmentationCaptureError(
            f"Unexpected segmentation buffer size {array.size}, expected {expected} or {rgb_expected}"
        )

    def _create_request(self) -> Any:
        if airsim is not None:
            return airsim.ImageRequest(
                self._camera_name,
                self._image_type,
                pixels_as_float=False,
                compress=False,
            )
        return _ImageRequest(
            camera_name=self._camera_name,
            image_type=self._image_type,
            pixels_as_float=False,
            compress=False,
        )
