from types import SimpleNamespace

import numpy as np
import pytest

from perception.detector import ModelLoadError, YoloDetector
from perception.segmentation import SegmentationAdapter


class FakeAirSimClient:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def simGetImages(self, requests):  # noqa: N802 - mimic AirSim API casing
        if not requests:
            raise ValueError("At least one image request required")
        height, width = self._mask.shape
        flattened = (self._mask.astype(np.uint8)).tobytes()
        return [
            SimpleNamespace(
                image_data_uint8=flattened,
                width=width,
                height=height,
            )
        ]


def test_capture_segmentation_returns_integer_mask():
    mask = np.arange(9, dtype=np.uint8).reshape(3, 3)
    adapter = SegmentationAdapter(FakeAirSimClient(mask), camera_name="0")

    result = adapter.capture_segmentation()

    assert result.shape == mask.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, mask)


def test_yolo_detector_lazy_load(monkeypatch):
    detections = []

    class DummyBoxes:
        def __init__(self) -> None:
            self.xywhn = np.array([[0.5, 0.5, 0.2, 0.1]])
            self.cls = np.array([1.0])
            self.conf = np.array([0.9])

    class DummyResult:
        boxes = DummyBoxes()

    def fake_loader(repo, model):  # noqa: ARG001 - signature compatibility
        detections.append((repo, model))
        return lambda image: [DummyResult()]

    monkeypatch.setattr(
        "perception.detector.torch", SimpleNamespace(hub=SimpleNamespace(load=fake_loader))
    )

    detector = YoloDetector(model_name="yolov12n")
    output = detector.run_detection(np.zeros((480, 640, 3), dtype=np.uint8))

    assert detections == [("ultralytics/yolov12", "yolov12n")]
    assert output == [{"class_id": 1, "bbox": [0.5, 0.5, 0.2, 0.1], "confidence": 0.9}]


def test_yolo_detector_fails_without_torch():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setitem(globals(), "torch", None)
    with pytest.raises(ModelLoadError):
        YoloDetector(model_name="yolov12n")
    monkeypatch.undo()


def test_build_observation_inputs_merges_modalities(monkeypatch):
    adapter = SegmentationAdapter(
        FakeAirSimClient(np.zeros((2, 2), dtype=np.uint8)), camera_name="0"
    )
    detector = YoloDetector(model=lambda image: [])

    mask = np.ones((2, 2), dtype=np.uint8)
    detections = [{"class_id": 0, "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.5}]

    monkeypatch.setattr(adapter, "capture_segmentation", lambda: mask)
    monkeypatch.setattr(detector, "run_detection", lambda image: detections)

    inputs = detector.build_observation_inputs(
        np.zeros((2, 2, 3), dtype=np.uint8), segmentation_adapter=adapter
    )

    assert inputs["segmentation_mask"] is mask
    assert inputs["detections"] is detections
