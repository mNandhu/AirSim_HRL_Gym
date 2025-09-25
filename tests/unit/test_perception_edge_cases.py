import numpy as np
import pytest

from perception.detector import ModelLoadError, YoloDetector
from perception.segmentation import SegmentationAdapter, SegmentationCaptureError


class NoImageClient:
    def simGetImages(self, requests):  # noqa: N802 - mimic AirSim API
        return []


def test_segmentation_adapter_raises_when_no_images():
    adapter = SegmentationAdapter(NoImageClient(), camera_name="0", image_type=5)
    with pytest.raises(SegmentationCaptureError):
        adapter.capture_segmentation()


def test_detector_returns_empty_when_model_has_no_predictions():
    class DummyModel:
        def __call__(self, image):
            return []

    detector = YoloDetector(model=DummyModel())
    assert detector.run_detection(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detector_raises_when_torch_missing(monkeypatch):
    monkeypatch.setattr("perception.detector.torch", None)
    with pytest.raises(ModelLoadError):
        YoloDetector()
