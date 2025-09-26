from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scripts.visualize_perception as viz


def test_colorize_segmentation_produces_rgb_image():
    mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    colored = viz._colorize_segmentation(mask)
    assert colored is not None
    assert colored.shape == mask.shape + (3,)
    assert colored.dtype == np.uint8
    assert colored.max() > 0


def test_overlay_segmentation_blends_values():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.full((2, 2, 3), 255, dtype=np.uint8)
    blended = viz._overlay_segmentation(rgb, mask, alpha=0.5)
    assert np.all(blended == 127) or np.all(blended == 128)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0.5, 0.5, 0.5, 0.5], (25, 25, 75, 75)),
        ([0.0, 0.0, 0.2, 0.2], (0, 0, 10, 10)),
        ([1.0, 1.0, 0.2, 0.2], (90, 90, 100, 100)),
    ],
)
def test_normalized_box_to_pixels_clamps_to_bounds(bbox, expected):
    x_min, y_min, x_max, y_max = viz._normalized_box_to_pixels(bbox, width=100, height=100)
    assert (x_min, y_min, x_max, y_max) == expected


def test_annotate_detections_draws_green_border():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 1,
            "bbox": [0.5, 0.5, 0.4, 0.4],
            "confidence": 0.9,
        }
    ]
    annotated = viz._annotate_detections(rgb, detections, confidence_threshold=0.1)
    # Expect green borders (0, 255, 0) around the box edges
    assert (annotated[30, 35] == np.array([0, 255, 0])).all()
    assert (annotated[45, 30] == np.array([0, 255, 0])).all()


def test_dummy_frame_generates_consistent_shapes():
    frame = viz._build_dummy_frame(3)
    assert frame.rgb.shape[-1] == 3
    assert frame.segmentation is not None
    assert frame.segmentation.shape[:2] == frame.rgb.shape[:2]
    assert len(frame.detections) == 1
    detection = frame.detections[0]
    assert set(detection) == {"class_id", "bbox", "confidence"}


def test_run_visualization_dummy_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda _seconds: None)

    args = argparse.Namespace(
        detector_model="yolov12n",
        camera_name="0",
        confidence_threshold=0.2,
        segmentation_alpha=0.4,
        save_dir=tmp_path,
        max_frames=1,
        poll_interval=0.0,
        figure_size=(4.0, 2.0),
        dummy=True,
    )

    monkeypatch.setattr(viz, "_colorize_segmentation", lambda _mask: None)
    monkeypatch.setattr(viz.time, "sleep", lambda _seconds: None)
    viz.run_visualization(args)
    plt.close("all")


def test_parse_args_handles_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--detector-model",
            "yolov8n.pt",
            "--max-frames",
            "3",
            "--confidence-threshold",
            "0.7",
            "--dummy",
        ],
    )
    args = viz._parse_args()
    assert args.detector_model == "yolov8n.pt"
    assert args.max_frames == 3
    assert args.confidence_threshold == pytest.approx(0.7)
    assert args.dummy is True


def test_ensure_client_invokes_airsim_calls(monkeypatch):
    class StubClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None]] = []

        def confirmConnection(self) -> None:  # noqa: N802 - mimic AirSim casing
            self.calls.append(("confirm", None))

        def enableApiControl(self, flag: bool) -> None:  # noqa: N802 - AirSim style
            self.calls.append(("api", flag))

        def armDisarm(self, flag: bool) -> None:  # noqa: N802 - AirSim style
            self.calls.append(("arm", flag))

    class StubAirSim:
        CarClient = StubClient

    monkeypatch.setattr(viz, "airsim", StubAirSim())
    client = viz._ensure_client()
    assert ("api", True) in client.calls
    assert ("arm", True) in client.calls


def test_ensure_client_requires_airsim(monkeypatch):
    monkeypatch.setattr(viz, "airsim", None)
    with pytest.raises(RuntimeError):
        viz._ensure_client()


def test_fetch_rgb_frame_uses_airsim(monkeypatch):
    class StubImageType:
        Scene = 0

    class StubAirSim:
        ImageType = StubImageType

        class ImageRequest:
            def __init__(self, camera_name, image_type, pixels_as_float, compress) -> None:
                self.camera_name = camera_name
                self.image_type = image_type
                self.pixels_as_float = pixels_as_float
                self.compress = compress

    class StubClient:
        def __init__(self) -> None:
            self.requests = None

        def simGetImages(self, requests):  # noqa: N802 - mimic AirSim casing
            self.requests = requests
            return ["image"]

    client = StubClient()
    monkeypatch.setattr(viz, "airsim", StubAirSim())
    result = viz._fetch_rgb_frame(client, "0")
    assert result == "image"
    assert client.requests is not None
    assert isinstance(client.requests[0], StubAirSim.ImageRequest)


def test_fetch_rgb_frame_requires_airsim(monkeypatch):
    monkeypatch.setattr(viz, "airsim", None)
    with pytest.raises(RuntimeError):
        viz._fetch_rgb_frame(client=object(), camera_name="0")


def test_convert_to_rgb_supports_multiple_inputs():
    rgb_array = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_converted = viz._convert_to_rgb(rgb_array)
    assert rgb_converted is not None
    assert rgb_converted.shape == (4, 4, 3)

    grayscale = np.arange(9, dtype=np.uint8).reshape(3, 3)
    converted = viz._convert_to_rgb(grayscale)
    assert converted is not None
    assert converted.shape == (3, 3, 3)
    assert np.array_equal(converted[..., 0], grayscale)

    class StubImage:
        def __init__(self) -> None:
            self.height = 2
            self.width = 2
            self.image_data_uint8 = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])

    airsim_response = StubImage()
    converted = viz._convert_to_rgb(airsim_response)
    assert converted is not None
    assert converted.shape == (2, 2, 3)
    assert converted.dtype == np.uint8


def test_convert_to_rgb_handles_invalid_inputs():
    assert viz._convert_to_rgb(None) is None
    assert viz._convert_to_rgb(np.arange(5)) is None

    class MissingMeta:
        def __init__(self) -> None:
            self.height = 2
            self.width = None
            self.image_data_uint8 = bytes([0, 0])

    assert viz._convert_to_rgb(MissingMeta()) is None

    class BadBuffer:
        def __init__(self) -> None:
            self.height = 2
            self.width = 2
            self.image_data_uint8 = bytes([0] * 5)

    assert viz._convert_to_rgb(BadBuffer()) is None


def test_overlay_segmentation_validates_alpha():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    mask = np.zeros_like(rgb)
    with pytest.raises(ValueError):
        viz._overlay_segmentation(rgb, mask, alpha=1.1)


def test_overlay_segmentation_requires_matching_shape():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((3, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        viz._overlay_segmentation(rgb, mask, alpha=0.5)


def test_annotate_detections_validates_input_shape():
    with pytest.raises(ValueError):
        viz._annotate_detections(np.zeros((4, 4), dtype=np.uint8), [], confidence_threshold=0.1)


def test_annotate_detections_ignores_invalid_entries():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    detections = [
        {"class_id": 0, "confidence": 0.9},
        {"class_id": 1, "bbox": [0.5, 0.5], "confidence": 0.9},
        {"class_id": 2, "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.05},
    ]
    annotated = viz._annotate_detections(rgb, detections, confidence_threshold=0.1)
    assert np.array_equal(annotated, rgb)


def test_colorize_segmentation_rejects_non_2d():
    with pytest.raises(ValueError):
        viz._colorize_segmentation(np.zeros((2, 2, 2), dtype=np.int32))


def test_prepare_output_dir_creates_path(tmp_path):
    out_dir = tmp_path / "frames"
    result = viz._prepare_output_dir(out_dir)
    assert result == out_dir
    assert result is not None
    assert result.exists()


def test_prepare_output_dir_handles_none():
    assert viz._prepare_output_dir(None) is None


def test_load_class_names_accepts_dict_and_list():
    class DictModel:
        def __init__(self) -> None:
            self.names = {0: "car", 1: "pedestrian"}

    dict_detector = type("Det", (), {"_model": DictModel()})()
    dict_mapping = viz._load_class_names(cast(Any, dict_detector))
    assert dict_mapping == {0: "car", 1: "pedestrian"}

    class ListModel:
        def __init__(self) -> None:
            self.names = ["car", "pedestrian"]

    list_detector = type("Det", (), {"_model": ListModel()})()
    list_mapping = viz._load_class_names(cast(Any, list_detector))
    assert list_mapping == {0: "car", 1: "pedestrian"}


def test_capture_frame_filters_by_confidence(monkeypatch):
    class RawImage:
        def __init__(self) -> None:
            self.height = 2
            self.width = 2
            self.image_data_uint8 = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])

    raw = RawImage()

    class DummyPipeline:
        def build_observation_inputs(self, raw_rgb):
            assert raw_rgb is raw
            return {
                "segmentation_mask": np.zeros((2, 2), dtype=np.int32),
                "detections": [
                    {"class_id": 0, "bbox": [0.5, 0.5, 1.0, 1.0], "confidence": 0.95},
                    {"class_id": 1, "bbox": [0.5, 0.5, 1.0, 1.0], "confidence": 0.2},
                ],
            }

    monkeypatch.setattr(viz, "_fetch_rgb_frame", lambda _client, _camera_name: raw)
    pipeline = DummyPipeline()
    frame = viz._capture_frame(
        client=object(),
        pipeline=cast(Any, pipeline),
        camera_name="0",
        confidence_threshold=0.5,
    )

    assert frame.rgb.shape == (2, 2, 3)
    assert frame.segmentation is not None
    assert len(frame.detections) == 1


def test_capture_frame_raises_when_rgb_conversion_fails(monkeypatch):
    class DummyPipeline:
        def build_observation_inputs(self, raw_rgb):
            return {"segmentation_mask": np.zeros((1, 1), dtype=np.int32), "detections": []}

    monkeypatch.setattr(viz, "_fetch_rgb_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(viz, "_convert_to_rgb", lambda _raw: None)

    with pytest.raises(RuntimeError):
        viz._capture_frame(
            client=object(),
            pipeline=cast(Any, DummyPipeline()),
            camera_name="0",
            confidence_threshold=0.1,
        )


def test_capture_frame_wraps_segmentation_error(monkeypatch):
    class DummyPipeline:
        def build_observation_inputs(self, raw_rgb):
            raise viz.SegmentationCaptureError("boom")

    monkeypatch.setattr(viz, "_fetch_rgb_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(viz, "_convert_to_rgb", lambda _raw: np.zeros((1, 1, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError) as excinfo:
        viz._capture_frame(
            client=object(),
            pipeline=cast(Any, DummyPipeline()),
            camera_name="0",
            confidence_threshold=0.1,
        )
    assert "segmentation" in str(excinfo.value).lower()


def test_matplotlib_viewer_update_saves_frames(monkeypatch, tmp_path):
    set_titles: list[str] = []
    saved_paths: list[Path] = []

    class DummyImageArtist:
        def __init__(self) -> None:
            self.data = None

        def set_data(self, data) -> None:
            self.data = data

    class DummyAxis:
        def __init__(self) -> None:
            self.title = None
            self.axis_calls: list[str] = []
            self.images: list[DummyImageArtist] = []

        def set_title(self, title: str) -> None:
            self.title = title
            set_titles.append(title)

        def axis(self, arg: str) -> None:
            self.axis_calls.append(arg)

        def imshow(self, data) -> DummyImageArtist:
            artist = DummyImageArtist()
            artist.set_data(data)
            self.images.append(artist)
            return artist

    class DummyManager:
        def __init__(self) -> None:
            self.titles = []

        def set_window_title(self, title: str) -> None:
            self.titles.append(title)

    class DummyCanvas:
        def __init__(self) -> None:
            self.manager = DummyManager()

        def draw_idle(self) -> None:
            pass

        def flush_events(self) -> None:
            pass

    class DummyFigure:
        def __init__(self) -> None:
            self.canvas = DummyCanvas()

        def savefig(self, path: Path, bbox_inches: str) -> None:
            saved_paths.append(path)

    def fake_subplots(_rows: int, _cols: int, figsize: tuple[float, float]):
        fig = DummyFigure()
        axes = [DummyAxis(), DummyAxis(), DummyAxis()]
        return fig, axes

    monkeypatch.setattr(viz.plt, "subplots", fake_subplots)
    monkeypatch.setattr(viz.plt, "ion", lambda: None)
    monkeypatch.setattr(viz.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(viz.plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(viz.plt, "pause", lambda _seconds: None)

    viewer = viz._MatplotlibViewer(figure_size=(4.0, 2.0))
    save_path = tmp_path / "frame.png"
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    viewer.update(
        rgb=frame,
        detections_image=frame,
        segmentation_overlay=frame,
        frame_index=0,
        fps=30.0,
        save_path=save_path,
    )

    assert saved_paths == [save_path]
