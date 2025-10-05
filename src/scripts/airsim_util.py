"""Shared helpers for interacting with AirSim from training scripts."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from typing import Any, Mapping, cast

import numpy as np

try:  # pragma: no cover - heavy dependency loaded lazily in scripts
    import airsim
except ImportError as exc:  # pragma: no cover - surfaced by caller scripts
    raise ImportError(
        "airsim is required for simulator integration. Install with: pip install airsim"
    ) from exc

from perception.async_pipeline import AsyncPerceptionPipeline
from perception.detector import ModelLoadError, YoloDetector
from perception.pipeline import PerceptionPipeline
from perception.segmentation import SegmentationAdapter
from utils.airsim_runner import airsim_session

__all__ = [
    "AirSimSimulatorAdapter",
    "build_perception",
    "extract_waypoints",
    "get_airsim_client",
    "sim_context",
]


class AirSimSimulatorAdapter:
    """Real AirSim simulator adapter for training and evaluation loops."""

    def __init__(
        self,
        client: Any,
        *,
        horizon: int,
        control_dt: float = 0.1,
        enable_rgb: bool = True,
        perception_pipeline: Any = None,
    ) -> None:
        self._client = client
        self._step = 0
        self._horizon = horizon
        self._initial_pose = None
        self._control_dt = control_dt
        self._previous_velocity = None
        self._previous_timestamp = None
        self._goal_pose = None
        self._waypoints = None
        self._current_waypoint_index = 0
        self._accumulator_time = 0.0
        self._enable_rgb = enable_rgb
        self._perception_pipeline = perception_pipeline

    @property
    def client(self) -> Any:
        return self._client

    def reset(self, experiment) -> dict[str, Any]:
        self._step = 0
        pose = experiment.start_pose
        self._goal_pose = experiment.goal_pose

        if hasattr(experiment, "waypoints") and experiment.waypoints:
            self._waypoints = experiment.waypoints
            self._current_waypoint_index = 0
        else:
            self._waypoints = None

        if self._initial_pose is None:
            self._initial_pose = pose

        position = airsim.Vector3r(pose.x, pose.y, pose.z)
        orientation = airsim.to_quaternion(0, 0, pose.yaw)
        vehicle_pose = airsim.Pose(position, orientation)

        try:
            self._client.simPause(True)
        except Exception:  # pragma: no cover - connection hiccups
            pass
        try:
            self._client.reset()
        except Exception:  # pragma: no cover
            pass
        try:
            self._client.simSetVehiclePose(vehicle_pose, True)
        except Exception:  # pragma: no cover
            pass
        try:
            self._client.simPause(False)
        except Exception:  # pragma: no cover
            pass

        target_xy = (float(pose.x), float(pose.y))
        settle_tol_m = 0.5
        timeout_s = 2.0
        poll_dt = 0.02
        start_t = time.monotonic()
        last_state = None
        while time.monotonic() - start_t < timeout_s:
            try:
                car_state = self._client.getCarState()
                last_state = car_state
                pos = car_state.kinematics_estimated.position
                dx = float(pos.x_val) - target_xy[0]
                dy = float(pos.y_val) - target_xy[1]
                if (dx * dx + dy * dy) ** 0.5 <= settle_tol_m:
                    break
            except Exception:  # pragma: no cover
                pass
            time.sleep(poll_dt)

        self._accumulator_time = 0.0

        try:
            car_state = self._client.getCarState()
        except Exception:  # pragma: no cover - fallback when RPC flaky
            car_state = last_state if last_state is not None else None
        if car_state is None:  # pragma: no cover - defensive fallback

            class _V3:
                def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
                    self.x_val = x
                    self.y_val = y
                    self.z_val = z

            class _Kin:
                def __init__(self) -> None:
                    self.position = _V3(target_xy[0], target_xy[1], float(pose.z))
                    self.linear_velocity = _V3(0.0, 0.0, 0.0)
                    self.orientation = airsim.to_quaternion(0, 0, pose.yaw)

            class _State:
                def __init__(self) -> None:
                    self.kinematics_estimated = _Kin()
                    self.timestamp = int(time.time() * 1e9)

            car_state = _State()

        self._previous_velocity = _vector_from_airsim(
            car_state.kinematics_estimated.linear_velocity
        )
        self._previous_timestamp = getattr(car_state, "timestamp", None)
        return self._build_state_dict(car_state, experiment)

    def step(self, action: Mapping[str, float]) -> dict[str, Any]:
        throttle = max(0.0, min(1.0, float(action.get("throttle", 0.0))))
        brake = max(0.0, min(1.0, float(action.get("brake", 0.0))))
        steering = max(-1.0, min(1.0, float(action.get("steering", 0.0))))

        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        car_controls.brake = brake
        car_controls.steering = steering

        self._client.setCarControls(car_controls)
        self._advance_simulation()

        self._step += 1
        car_state = self._client.getCarState()
        return self._build_state_dict(car_state)

    def _advance_simulation(self) -> None:
        try:
            if hasattr(self._client, "simContinueForTime"):
                self._client.simContinueForTime(self._control_dt)
                return
        except Exception:  # pragma: no cover
            pass
        time.sleep(self._control_dt)

    def _build_state_dict(self, car_state, experiment=None) -> dict[str, Any]:
        collision_info = self._client.simGetCollisionInfo()
        has_collision = collision_info.has_collided

        position = car_state.kinematics_estimated.position
        pos_xy = (float(position.x_val), float(position.y_val))
        distance_to_goal = 999.0
        goal_xy: tuple[float, float] | None = None

        if self._waypoints is not None and self._current_waypoint_index < len(self._waypoints):
            current_waypoint = self._waypoints[self._current_waypoint_index]
            wp_x, wp_y = float(current_waypoint.x), float(current_waypoint.y)
            dist_to_current = math.sqrt((position.x_val - wp_x) ** 2 + (position.y_val - wp_y) ** 2)

            if dist_to_current <= 5.0 and self._current_waypoint_index < len(self._waypoints) - 1:
                self._current_waypoint_index += 1
                current_waypoint = self._waypoints[self._current_waypoint_index]
                wp_x, wp_y = float(current_waypoint.x), float(current_waypoint.y)
                dist_to_current = math.sqrt(
                    (position.x_val - wp_x) ** 2 + (position.y_val - wp_y) ** 2
                )

            distance_to_goal = dist_to_current
            goal_xy = (wp_x, wp_y)
        elif self._goal_pose is not None or (
            experiment and hasattr(experiment, "goal_pose") and experiment.goal_pose
        ):
            goal_pose = getattr(experiment, "goal_pose", None) or self._goal_pose
            if goal_pose is not None:
                goal_xy = (float(goal_pose.x), float(goal_pose.y))
                distance_to_goal = math.sqrt(
                    (position.x_val - goal_pose.x) ** 2 + (position.y_val - goal_pose.y) ** 2
                )

        velocity = _vector_from_airsim(car_state.kinematics_estimated.linear_velocity)
        speed = float(np.linalg.norm(velocity))
        timestamp = getattr(car_state, "timestamp", None)
        acceleration = 0.0
        if (
            self._previous_velocity is not None
            and timestamp is not None
            and self._previous_timestamp
        ):
            dt = max((timestamp - self._previous_timestamp) * 1e-9, 1e-3)
            acceleration = float(np.linalg.norm((velocity - self._previous_velocity) / dt))
            self._accumulator_time += dt
        else:
            self._accumulator_time = max(self._accumulator_time, self._step * self._control_dt)

        self._previous_velocity = velocity
        self._previous_timestamp = timestamp

        heading = _compute_heading_deg(car_state.kinematics_estimated.orientation)
        progress_possible = not has_collision and speed >= 0.2

        # Calculate lane mask coverage from segmentation data
        lane_mask_coverage = self._calculate_lane_coverage()

        return {
            "telemetry": {
                "distance_to_goal": distance_to_goal,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "heading_deg": heading,
                "collision": has_collision,
                "lane_mask_coverage_ratio": lane_mask_coverage,
                "progress_possible": progress_possible,
                "sim_time_sec": self._accumulator_time,
                "position_xy": pos_xy,
                "goal_xy": goal_xy,
            },
            "image": self._get_camera_image(),
        }

    def _calculate_lane_coverage(self) -> float:
        """Calculate the ratio of road pixels in the segmentation mask."""
        if self._perception_pipeline is None:
            return 1.0  # Default to on-road if no perception

        try:
            # Get current camera image
            raw_image = self._get_camera_image()
            if raw_image is None:
                return 1.0

            # Get segmentation mask from perception pipeline
            perception_data = self._perception_pipeline.build_observation_inputs(raw_image)
            seg_mask = perception_data.get("segmentation_mask")

            if seg_mask is None:
                return 1.0  # Default to on-road if segmentation fails

            # Count road pixels (assuming road has specific segment IDs)
            # Common AirSim road segment IDs: 0 (road), 1 (road marking)
            # Adjust these IDs based on your AirSim environment
            road_pixels = np.isin(seg_mask, [0, 1])
            total_pixels = seg_mask.size

            if total_pixels == 0:
                return 1.0

            coverage = float(np.sum(road_pixels)) / float(total_pixels)
            return coverage

        except Exception:  # pragma: no cover - fallback on error
            return 1.0  # Default to on-road if calculation fails

    def _get_camera_image(self):
        if not self._enable_rgb:
            return None
        try:
            responses = self._client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses:
                return responses[0]
        except Exception:  # pragma: no cover
            pass
        return None


def build_perception(
    client: Any,
    camera_name: str,
    detector_model: str,
    *,
    async_mode: bool = False,
    enable_segmentation: bool = True,
):
    """Create a perception pipeline with YOLO and segmentation adapters."""

    segmentation = SegmentationAdapter(client, camera_name=camera_name)
    use_dummy = False
    try:
        detector = YoloDetector(model_name=detector_model)
    except ModelLoadError as exc:
        print(f"Warning: Could not load YOLO detector: {exc}")

        class DummyDetector:
            def run_detection(self, _image):
                return []

        detector = DummyDetector()
        use_dummy = True

    seg_enabled = enable_segmentation and not use_dummy
    if async_mode and not use_dummy:
        return AsyncPerceptionPipeline(
            segmentation=segmentation,
            detector=cast(YoloDetector, detector),
            enable_segmentation=seg_enabled,
        )
    return PerceptionPipeline(
        segmentation=segmentation,
        detector=cast(YoloDetector, detector),
        enable_segmentation=seg_enabled,
    )


def get_airsim_client() -> airsim.CarClient:
    """Connect to AirSim using optional AIRSIM_HOST / AIRSIM_PORT overrides."""

    host = os.environ.get("AIRSIM_HOST", "127.0.0.1")
    port = int(os.environ.get("AIRSIM_PORT", 41451))
    client = airsim.CarClient(ip=host, port=port)
    client.confirmConnection()
    return client


def sim_context(*, mode: str, settings_path: str):
    """Return context manager that launches AirSim unless orchestrator already did."""

    if os.environ.get("AIRSIM_PORT"):
        return nullcontext()
    return airsim_session(mode=mode, settings_path=settings_path)


def extract_waypoints(experiment) -> list[tuple[float, float]]:
    """Extract 2D waypoints from an experiment definition for plotting."""

    waypoints: list[tuple[float, float]] = []
    try:
        if hasattr(experiment, "waypoints") and experiment.waypoints:
            waypoints = [(float(wp.x), float(wp.y)) for wp in experiment.waypoints]
        elif hasattr(experiment, "goal_pose") and experiment.goal_pose:
            start = experiment.start_pose
            goal = experiment.goal_pose
            waypoints = [(float(start.x), float(start.y)), (float(goal.x), float(goal.y))]
    except Exception:  # pragma: no cover - defensive
        pass
    return waypoints


def _vector_from_airsim(vector) -> np.ndarray:
    return np.array([vector.x_val, vector.y_val, vector.z_val], dtype=np.float32)


def _compute_heading_deg(orientation) -> float:
    try:
        yaw = airsim.to_eularian_angles(orientation)[2]
    except Exception:  # pragma: no cover
        yaw = 0.0
    return math.degrees(yaw)
