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

    def run_detection(self, image):
        self.called = True
        return [{"label": "car", "confidence": 0.42}]


def test_pipeline_combines_segmentation_and_detection():
    segmentation = StubSegmentation()
    detector = StubDetector()
    pipeline = PerceptionPipeline(segmentation=segmentation, detector=detector)  # type: ignore[arg-type]

    result = pipeline.build_observation_inputs(raw_rgb=None)

    assert segmentation.called is True
    assert detector.called is True
    assert result["segmentation_mask"] == [[0, 1], [1, 0]]
    assert result["detections"][0]["label"] == "car"
