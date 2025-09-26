import numpy as np

from perception.pipeline import PerceptionPipeline


class StubSegmentation:
    def __init__(self) -> None:
        self.called = False

    def capture_segmentation(self):
        self.called = True
        return [[0, 1], [1, 0]]


class StubDetector:
    def __init__(self) -> None:
        self.called = False
        self.last_image = None

    def run_detection(self, image):
        self.called = True
        self.last_image = image
        return [{"label": "car", "confidence": 0.42}]


def test_pipeline_combines_segmentation_and_detection():
    segmentation = StubSegmentation()
    detector = StubDetector()
    pipeline = PerceptionPipeline(segmentation=segmentation, detector=detector)  # type: ignore[arg-type]

    fake_image = type(
        "Img",
        (),
        {
            "image_data_uint8": bytes([255, 0, 0]) * 4,
            "height": 1,
            "width": 4,
        },
    )()

    result = pipeline.build_observation_inputs(raw_rgb=fake_image)

    assert segmentation.called is True
    assert detector.called is True
    assert result["segmentation_mask"] == [[0, 1], [1, 0]]
    assert result["detections"][0]["label"] == "car"
    assert isinstance(detector.last_image, np.ndarray)
    assert detector.last_image.shape == (1, 4, 3)


def test_to_numpy_image_handles_numpy_variants():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    assert PerceptionPipeline._to_numpy_image(rgb) is rgb

    grayscale = np.arange(4, dtype=np.uint8).reshape(2, 2)
    converted = PerceptionPipeline._to_numpy_image(grayscale)
    assert converted is not None
    assert converted.shape == (2, 2, 3)
    assert np.array_equal(converted[..., 0], grayscale)


def test_to_numpy_image_handles_buffers_and_invalid_cases():
    color_bytes = bytes(range(12))
    color_image = type(
        "ColorImg",
        (),
        {"image_data_uint8": color_bytes, "height": 2, "width": 2},
    )()
    color = PerceptionPipeline._to_numpy_image(color_image)
    assert color is not None
    assert color.shape == (2, 2, 3)

    gray_bytes = bytes(range(4))
    gray_image = type(
        "GrayImg",
        (),
        {"image_data_uint8": gray_bytes, "height": 2, "width": 2},
    )()
    gray = PerceptionPipeline._to_numpy_image(gray_image)
    assert gray is not None
    assert gray.shape == (2, 2, 3)
    assert np.array_equal(gray[..., 0], np.array([[0, 1], [2, 3]], dtype=np.uint8))

    missing_meta = type("BadImg", (), {"image_data_uint8": color_bytes})()
    assert PerceptionPipeline._to_numpy_image(missing_meta) is None

    mismatch = type(
        "MismatchImg",
        (),
        {"image_data_uint8": bytes(range(5)), "height": 2, "width": 2},
    )()
    assert PerceptionPipeline._to_numpy_image(mismatch) is None

    invalid = np.zeros((5,), dtype=np.uint8)
    assert PerceptionPipeline._to_numpy_image(invalid) is None


def test_to_numpy_image_none_returns_none():
    assert PerceptionPipeline._to_numpy_image(None) is None


def test_pipeline_skips_detection_when_image_cannot_convert():
    segmentation = StubSegmentation()
    detector = StubDetector()
    pipeline = PerceptionPipeline(segmentation=segmentation, detector=detector)  # type: ignore[arg-type]

    class InvalidImage:
        image_data_uint8 = None
        height = None
        width = None

    result = pipeline.build_observation_inputs(raw_rgb=InvalidImage())

    assert detector.called is False
    assert result["detections"] == []


def test_pipeline_handles_none_image_input():
    segmentation = StubSegmentation()
    detector = StubDetector()
    pipeline = PerceptionPipeline(segmentation=segmentation, detector=detector)  # type: ignore[arg-type]

    result = pipeline.build_observation_inputs(raw_rgb=None)

    assert detector.called is False
    assert result["detections"] == []
