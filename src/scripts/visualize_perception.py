"""Interactive visualization of perception inputs and outputs feeding the HRL agent.

This script connects to an AirSim instance, captures the RGB camera feed alongside the
segmentation mask and YOLO detections, and renders them side-by-side for manual
inspection. It can optionally save annotated frames to disk for later analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from perception.detector import ModelLoadError, YoloDetector
from perception.pipeline import PerceptionPipeline
from perception.segmentation import SegmentationAdapter, SegmentationCaptureError

try:  # pragma: no cover - AirSim is optional at test time
    import airsim
except ImportError:  # pragma: no cover - fallback for environments without AirSim
    airsim = None  # type: ignore


@dataclasses.dataclass
class VisualizationFrame:
    """Bundle of perception outputs for visualization."""

    rgb: np.ndarray
    detections: list[dict[str, Any]]
    segmentation: np.ndarray | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize perception pipeline outputs.")
    parser.add_argument(
        "--detector-model",
        default="yolov12n",
        help="Name or path of the YOLO model to load (default: yolov12n)",
    )
    parser.add_argument(
        "--camera-name",
        default="0",
        help="AirSim camera name to capture RGB and segmentation streams (default: 0)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence for drawing bounding boxes (default: 0.25)",
    )
    parser.add_argument(
        "--segmentation-alpha",
        type=float,
        default=0.45,
        help="Alpha blending factor when overlaying the segmentation mask (default: 0.45)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save annotated frames (PNG). Directory is created if missing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames captured. Default renders until interrupted.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Delay in seconds between frame polls (default: 0.05).",
    )
    parser.add_argument(
        "--figure-size",
        type=float,
        nargs=2,
        default=(16.0, 6.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size. Default: 16 6",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run in dummy mode without AirSim, generating synthetic frames for debugging.",
    )
    return parser.parse_args()


def _ensure_client() -> Any:
    if airsim is None:
        raise RuntimeError(
            "AirSim Python API is not available. Install the airsim package or use --dummy mode."
        )
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def _convert_to_rgb(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1).astype(np.uint8)
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


def _colorize_segmentation(
    mask: np.ndarray | None, *, colormap: str = "tab20"
) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be 2D")
    cmap = plt.get_cmap(colormap, 256)
    palette = (cmap(range(cmap.N))[:, :3] * 255).astype(np.uint8)
    indexed = np.abs(mask.astype(np.int64)) % palette.shape[0]
    return palette[indexed]


def _overlay_segmentation(rgb: np.ndarray, colored_mask: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")
    if rgb.shape != colored_mask.shape:
        raise ValueError("RGB image and colored mask must share shape")
    blended = alpha * colored_mask.astype(np.float32) + (1 - alpha) * rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _normalized_box_to_pixels(
    box: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x_c, y_c, box_w, box_h = box
    x_c = float(x_c) * width
    y_c = float(y_c) * height
    box_w = float(box_w) * width
    box_h = float(box_h) * height
    x_min = int(max(x_c - box_w / 2.0, 0))
    y_min = int(max(y_c - box_h / 2.0, 0))
    x_max = int(min(x_c + box_w / 2.0, width))
    y_max = int(min(y_c + box_h / 2.0, height))
    return x_min, y_min, x_max, y_max


def _annotate_detections(
    rgb: np.ndarray,
    detections: Iterable[dict[str, Any]],
    *,
    confidence_threshold: float,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3)")
    annotated = rgb.copy()
    height, width = rgb.shape[:2]
    for detection in detections:
        confidence = float(detection.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = _normalized_box_to_pixels(bbox, width, height)
        class_id = int(detection.get("class_id", -1))
        label = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        color = (0, 255, 0)
        annotated[y_min : y_min + 2, x_min:x_max] = color
        annotated[y_max - 2 : y_max, x_min:x_max] = color
        annotated[y_min:y_max, x_min : x_min + 2] = color
        annotated[y_min:y_max, x_max - 2 : x_max] = color
        text_y = max(y_min - 10, 0)
        text_x = max(x_min, 0)
        text = f"{label} {confidence:.2f}"
        _draw_text(annotated, text_x, text_y, text)
    return annotated


def _draw_text(canvas: np.ndarray, x: int, y: int, text: str) -> None:
    font = font_manager.FontProperties(family="DejaVu Sans", size=8)
    fig = Figure(figsize=(1, 1), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    ax.text(0.0, 1.0, text, fontproperties=font, color="white", fontsize=8)
    canvas_agg = FigureCanvasAgg(fig)
    canvas_agg.draw()
    frame_width, frame_height = canvas_agg.get_width_height()
    buf = np.frombuffer(canvas_agg.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(frame_height, frame_width, 4)
    plt.close(fig)
    overlay = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    y_end = min(y + frame_height, canvas.shape[0])
    x_end = min(x + frame_width, canvas.shape[1])
    region = canvas[y:y_end, x:x_end]
    overlay_crop = overlay[: y_end - y, : x_end - x]
    alpha_crop = alpha[: y_end - y, : x_end - x]
    canvas[y:y_end, x:x_end] = (
        (1 - alpha_crop) * region.astype(np.float32) + alpha_crop * overlay_crop.astype(np.float32)
    ).astype(np.uint8)


def _fetch_rgb_frame(client: Any, camera_name: str) -> Any:
    if airsim is None:
        raise RuntimeError("AirSim Python API is required to fetch frames unless --dummy is used")
    responses = client.simGetImages(
        [
            airsim.ImageRequest(
                camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False
            )
        ]
    )
    return responses[0] if responses else None


def _build_dummy_frame(index: int, width: int = 640, height: int = 360) -> VisualizationFrame:
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    rgb = np.stack(
        [np.sin(2 * np.pi * (xv + index / 10.0)), yv, np.cos(2 * np.pi * (yv + index / 20.0))],
        axis=-1,
    )
    rgb = ((rgb + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    mask = ((xv * 5 + index) % 5).astype(np.int32)
    detections = [
        {
            "class_id": 1,
            "bbox": [0.5, 0.5, 0.3 + 0.05 * np.sin(index / 5.0), 0.3],
            "confidence": 0.85,
        }
    ]
    return VisualizationFrame(rgb=rgb, detections=detections, segmentation=mask)


def _capture_frame(
    *,
    client: Any,
    pipeline: PerceptionPipeline,
    camera_name: str,
    confidence_threshold: float,
) -> VisualizationFrame:
    raw_rgb = _fetch_rgb_frame(client, camera_name)
    rgb_array = _convert_to_rgb(raw_rgb)
    try:
        perception_inputs = pipeline.build_observation_inputs(raw_rgb)
    except SegmentationCaptureError as exc:
        raise RuntimeError(f"Failed to capture segmentation mask: {exc}") from exc
    mask = perception_inputs.get("segmentation_mask")
    detections = perception_inputs.get("detections", [])
    if rgb_array is None:
        raise RuntimeError("Failed to convert AirSim image response to RGB array")
    filtered_detections = [
        detection
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= confidence_threshold
    ]
    return VisualizationFrame(rgb=rgb_array, detections=filtered_detections, segmentation=mask)


class _MatplotlibViewer:
    def __init__(self, *, figure_size: tuple[float, float]) -> None:
        self._fig, self._axes = plt.subplots(1, 3, figsize=figure_size)
        manager = getattr(self._fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Perception Visualization")
        titles = ["Raw RGB", "YOLO Detections", "Segmentation Overlay"]
        self._images: list[Any] = []
        for ax, title in zip(self._axes, titles, strict=True):
            ax.set_title(title)
            ax.axis("off")
            image = ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
            self._images.append(image)
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)

    def update(
        self,
        *,
        rgb: np.ndarray,
        detections_image: np.ndarray,
        segmentation_overlay: np.ndarray,
        frame_index: int,
        fps: float,
        save_path: Path | None,
    ) -> None:
        frames = [rgb, detections_image, segmentation_overlay]
        titles = [
            f"Raw RGB (frame {frame_index})",
            f"Detections (fps {fps:.1f})",
            "Segmentation Overlay",
        ]
        for ax, image_artist, frame, title in zip(
            self._axes, self._images, frames, titles, strict=True
        ):
            image_artist.set_data(frame)
            ax.set_title(title)
            ax.axis("off")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)
        if save_path is not None:
            self._fig.savefig(save_path, bbox_inches="tight")


def _prepare_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_class_names(detector: YoloDetector) -> dict[int, str] | None:
    names = getattr(detector._model, "names", None)  # noqa: SLF001 - accessing YOLO attribute
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return None


def run_visualization(args: argparse.Namespace) -> None:
    save_dir = _prepare_output_dir(args.save_dir)

    if args.dummy:
        viewer = _MatplotlibViewer(figure_size=tuple(args.figure_size))
        class_names = {1: "dummy"}
        start_time = time.perf_counter()
        frame = 0
        while args.max_frames is None or frame < args.max_frames:
            data = _build_dummy_frame(frame)
            colored_mask = _colorize_segmentation(data.segmentation)
            if colored_mask is None:
                colored_mask = np.zeros_like(data.rgb)
            overlay = _overlay_segmentation(data.rgb, colored_mask, args.segmentation_alpha)
            annotated = _annotate_detections(
                data.rgb,
                data.detections,
                confidence_threshold=args.confidence_threshold,
                class_names=class_names,
            )
            now = time.perf_counter()
            fps = frame / max(now - start_time, 1e-6)
            save_path = None
            if save_dir is not None:
                save_path = save_dir / f"dummy_{frame:05d}.png"
            viewer.update(
                rgb=data.rgb,
                detections_image=annotated,
                segmentation_overlay=overlay,
                frame_index=frame,
                fps=fps,
                save_path=save_path,
            )
            frame += 1
            time.sleep(args.poll_interval)
        return

    client = _ensure_client()
    segmentation = SegmentationAdapter(client, camera_name=args.camera_name)
    try:
        detector = YoloDetector(model_name=args.detector_model)
    except ModelLoadError as exc:
        raise RuntimeError(f"Failed to load YOLO detector: {exc}") from exc

    pipeline = PerceptionPipeline(segmentation=segmentation, detector=detector)
    viewer = _MatplotlibViewer(figure_size=tuple(args.figure_size))
    class_names = _load_class_names(detector)
    frame_index = 0
    start_time = time.perf_counter()

    while args.max_frames is None or frame_index < args.max_frames:
        try:
            frame = _capture_frame(
                client=client,
                pipeline=pipeline,
                camera_name=args.camera_name,
                confidence_threshold=args.confidence_threshold,
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            print(f"[visualize] Error capturing frame: {exc}")
            time.sleep(args.poll_interval)
            continue

        colored_mask = _colorize_segmentation(frame.segmentation)
        if colored_mask is None:
            colored_mask = np.zeros_like(frame.rgb)
        overlay = _overlay_segmentation(frame.rgb, colored_mask, args.segmentation_alpha)
        annotated = _annotate_detections(
            frame.rgb,
            frame.detections,
            confidence_threshold=args.confidence_threshold,
            class_names=class_names,
        )
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = (frame_index + 1) / elapsed
        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"frame_{frame_index:05d}.png"
        viewer.update(
            rgb=frame.rgb,
            detections_image=annotated,
            segmentation_overlay=overlay,
            frame_index=frame_index,
            fps=fps,
            save_path=save_path,
        )
        frame_index += 1
        time.sleep(args.poll_interval)


def main() -> None:
    args = _parse_args()
    run_visualization(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
