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


def test_segmentation_adapter_decodes_rgb_buffer():
    class DummyImage:
        height = 2
        width = 2
        image_data_uint8 = bytes(
            [
                1,
                0,
                0,
                2,
                0,
                0,
                3,
                0,
                0,
                4,
                0,
                0,
            ]
        )

    class RGBClient:
        def simGetImages(self, requests):  # noqa: N802
            return [DummyImage()]

    adapter = SegmentationAdapter(RGBClient(), camera_name="0", image_type=5)
    mask = adapter.capture_segmentation()
    assert mask.shape == (2, 2)
    assert mask.dtype == np.int32
    assert mask[0, 0] == 1
    assert mask[1, 1] == 4


def test_detector_raises_when_ultralytics_missing(monkeypatch):
    monkeypatch.setattr("perception.detector.YOLO", None)
    with pytest.raises(ModelLoadError):
        YoloDetector()
