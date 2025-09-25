"""Entry point for running seeded AirSim HRL experiments."""

from __future__ import annotations

import argparse
import json
from typing import Any

from airsim_env.env import AirSimEnv
from config.loader import load_experiment
from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.orchestrator import HRLOrchestrator
from hrl_agent.workers.sac_worker import SACWorker
from perception.pipeline import PerceptionPipeline
from perception.segmentation import SegmentationAdapter
from perception.detector import ModelLoadError, YoloDetector
from utils.airsim_runner import airsim_session
from utils.artifacts import ArtifactManager

try:  # pragma: no cover - airsim optional
    import airsim
except ImportError:  # pragma: no cover
    airsim = None  # type: ignore


class AirSimSimulatorAdapter:
    """Thin adapter that exposes reset/step for :class:`AirSimEnv`."""

    def __init__(self, client: Any, *, horizon: int) -> None:
        self._client = client
        self._step = 0
        self._horizon = horizon
        self._distance = float(horizon)

    def reset(self, experiment) -> dict[str, Any]:  # noqa: ANN001 - aligns with env expectations
        self._step = 0
        self._distance = float(experiment.horizon)
        pose = experiment.start_pose
        if (
            self._client is not None
            and hasattr(self._client, "simSetVehiclePose")
            and airsim is not None
        ):
            position = airsim.Vector3r(pose.x, pose.y, pose.z)
            orientation = airsim.to_quaternion(0, 0, pose.yaw)
            try:  # pragma: no cover - requires simulator
                self._client.simSetVehiclePose(airsim.Pose(position, orientation), True)
            except Exception:
                pass
        return _state_dict(self._distance, 0.0)

    def step(self, action: dict[str, float]) -> dict[str, Any]:
        throttle = float(action.get("throttle", 0.0))
        self._step += 1
        progress = max(throttle, 0.0)
        self._distance = max(0.0, self._distance - progress)
        return _state_dict(self._distance, throttle, progress_possible=True)


def _state_dict(distance: float, speed: float, *, progress_possible: bool = True) -> dict[str, Any]:
    return {
        "telemetry": {
            "distance_to_goal": distance,
            "speed_mps": speed,
            "collision": False,
            "lane_mask_coverage_ratio": 1.0,
            "progress_possible": progress_possible,
        },
        "image": None,
    }


def _create_coordinator() -> CommandCoordinator:
    commands = [
        "FOLLOW_LANE",
        "TURN_LEFT_AT_INTERSECTION",
        "TURN_RIGHT_AT_INTERSECTION",
        "STOP",
    ]
    policy = CommandPolicy(commands=commands)
    manager = DQNManager(policy)
    workers = {command: SACWorker() for command in commands}
    return CommandCoordinator(manager, workers)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seeded HRL AirSim experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    parser.add_argument("--settings", default="settings.json", help="Path to AirSim settings.json")
    parser.add_argument("--mode", choices=["gui", "headless"], default="headless")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="artifacts", help="Directory for artifacts")
    parser.add_argument("--detector-model", default="yolov12n")
    parser.add_argument("--camera-name", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_experiment(args.config)

    artifact_manager = ArtifactManager(args.output)
    artifact_manager.start_run(str(experiment.id))
    artifact_manager.write_json(
        "experiment.json",
        experiment.model_dump() if hasattr(experiment, "model_dump") else experiment.__dict__,
    )

    with airsim_session(mode=args.mode, settings_path=args.settings):
        client = _ensure_client()
        simulator = AirSimSimulatorAdapter(client, horizon=experiment.horizon)
        perception = _build_perception(client, args.camera_name, args.detector_model)
        env = AirSimEnv(experiment, simulator=simulator, perception=perception)
        coordinator = _create_coordinator()
        orchestrator = HRLOrchestrator(env, coordinator)
        result = orchestrator.run_episode(
            max_steps=args.max_steps, deterministic=args.deterministic
        )

    artifact_manager.write_json(
        "result.json",
        {
            "cumulative_reward": result.cumulative_reward,
            "steps": result.steps,
            "info": result.info,
            "config_hash": result.info.get("config_hash"),
        },
    )
    print(json.dumps({"reward": result.cumulative_reward, "steps": result.steps}, indent=2))
    return 0


def _ensure_client():
    if airsim is None:
        raise RuntimeError("AirSim Python API is required to run experiments")
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def _build_perception(client, camera_name: str, detector_model: str) -> PerceptionPipeline:
    segmentation = SegmentationAdapter(client, camera_name=camera_name)
    try:
        detector = YoloDetector(model_name=detector_model)
    except ModelLoadError as exc:
        raise RuntimeError("Failed to load YOLO model for detection") from exc
    return PerceptionPipeline(segmentation=segmentation, detector=detector)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
