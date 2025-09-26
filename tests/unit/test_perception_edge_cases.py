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


def test_segmentation_adapter_handles_grayscale_buffer():
    class DummyImage:
        height = 2
        width = 2
        image_data_uint8 = bytes([1, 2, 3, 4])

    class GrayClient:
        def simGetImages(self, requests):  # noqa: N802
            return [DummyImage()]

    adapter = SegmentationAdapter(GrayClient(), camera_name="0", image_type=5)
    mask = adapter.capture_segmentation()
    assert mask.shape == (2, 2)
    assert np.array_equal(mask, np.array([[1, 2], [3, 4]], dtype=np.uint8))


def test_segmentation_adapter_raises_on_missing_fields():
    class BadImage:
        height = 2
        width = None
        image_data_uint8 = bytes([1, 2, 3, 4])

    class BadClient:
        def simGetImages(self, requests):  # noqa: N802
            return [BadImage()]

    adapter = SegmentationAdapter(BadClient(), camera_name="0", image_type=5)
    with pytest.raises(SegmentationCaptureError):
        adapter.capture_segmentation()


def test_segmentation_adapter_raises_on_unexpected_size():
    class BadImage:
        height = 2
        width = 2
        image_data_uint8 = bytes(range(5))

    class BadClient:
        def simGetImages(self, requests):  # noqa: N802
            return [BadImage()]

    adapter = SegmentationAdapter(BadClient(), camera_name="0", image_type=5)
    with pytest.raises(SegmentationCaptureError):
        adapter.capture_segmentation()


def test_detector_raises_when_ultralytics_missing(monkeypatch):
    monkeypatch.setattr("perception.detector.YOLO", None)
    with pytest.raises(ModelLoadError):
        YoloDetector()


def test_yolo_detector_to_numpy_prefers_detach(monkeypatch):
    monkeypatch.setattr("perception.detector.torch", object())

    class DummyTensor:
        def __init__(self, payload: np.ndarray) -> None:
            self._payload = payload

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._payload

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = YoloDetector._to_numpy(DummyTensor(array))
    assert np.array_equal(result, array)


def test_yolo_detector_to_numpy_uses_cpu_fallback(monkeypatch):
    monkeypatch.setattr("perception.detector.torch", object())

    class CPUOnly:
        def __init__(self, payload: np.ndarray) -> None:
            self._payload = payload

        def cpu(self):
            return self

        def numpy(self):
            return self._payload

    array = np.arange(4, dtype=np.float32).reshape(2, 2)
    result = YoloDetector._to_numpy(CPUOnly(array))
    assert np.array_equal(result, array)


def test_detector_returns_empty_when_image_none():
    class FailingModel:
        def __call__(self, _image):
            raise AssertionError("Model should not be invoked when image is None")

    detector = YoloDetector(model=FailingModel())
    assert detector.run_detection(None) == []


def test_detector_returns_empty_when_boxes_missing():
    class DummyResult:
        boxes = None

    class DummyModel:
        def __call__(self, image):  # noqa: ARG002 - mimic API
            return [DummyResult()]

    detector = YoloDetector(model=DummyModel())
    output = detector.run_detection(np.zeros((1, 1, 3), dtype=np.uint8))
    assert output == []


def test_detector_reraises_unexpected_type_error():
    class BrokenModel:
        def __call__(self, image, verbose=False):  # noqa: ARG002
            raise TypeError("boom")

    detector = YoloDetector(model=BrokenModel())
    with pytest.raises(TypeError):
        detector.run_detection(np.zeros((1, 1, 3), dtype=np.uint8))


def test_yolo_detector_to_numpy_handles_generic_iterable():
    result = YoloDetector._to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
