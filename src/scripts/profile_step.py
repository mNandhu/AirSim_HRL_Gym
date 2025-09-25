"""Quick profiling helper for perception and environment steps."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Any

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator
from config.loader import load_experiment
from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.workers.sac_worker import SACWorker
from perception.detector import YoloDetector
from perception.pipeline import PerceptionPipeline
from perception.segmentation import SegmentationAdapter
from utils.airsim_runner import airsim_session
from utils.perf_metrics import LatencyTracker, time_block

try:  # pragma: no cover - optional AirSim dependency
    import airsim
except ImportError:  # pragma: no cover
    airsim = None  # type: ignore


class DummySimulator:
    def __init__(self, horizon: int) -> None:
        self._remaining = horizon

    def reset(self, experiment):
        self._remaining = experiment.horizon
        return {
            "telemetry": {
                "distance_to_goal": float(self._remaining),
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }

    def step(self, action):
        self._remaining = max(0, self._remaining - 1)
        return {
            "telemetry": {
                "distance_to_goal": float(self._remaining),
                "speed_mps": action.get("throttle", 0.0),
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile perception and environment step latency")
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", default="settings.json")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", choices=["gui", "headless"], default="headless")
    return parser.parse_args(argv)


def _build_env(
    experiment, detector_model: str = "yolov12n"
) -> tuple[AirSimEnv, CommandCoordinator]:
    if airsim is None:
        simulator = DummySimulator(experiment.horizon)
        perception = _dummy_perception()
    else:
        client = airsim.CarClient()
        client.confirmConnection()
        simulator = DummySimulator(experiment.horizon)
        perception = PerceptionPipeline(
            segmentation=SegmentationAdapter(client, camera_name="0"),
            detector=YoloDetector(model_name=detector_model),
        )
    env = AirSimEnv(
        experiment, simulator=simulator, perception=perception, reward_calculator=RewardCalculator()
    )
    commands = CommandPolicy(commands=["FOLLOW_LANE", "STOP"])
    manager = DQNManager(policy=commands)
    workers = {command: SACWorker() for command in commands.commands}
    coordinator = CommandCoordinator(manager, workers)
    return env, coordinator


def _dummy_perception():
    class DummyPerception:
        def build_observation_inputs(self, raw_rgb: Any) -> dict[str, Any]:
            return {"segmentation_mask": None, "detections": []}

    return DummyPerception()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_experiment(args.config)
    tracker = LatencyTracker()

    if airsim is not None:
        session_ctx = airsim_session(mode=args.mode, settings_path=args.settings)
    else:
        session_ctx = nullcontext()

    with session_ctx:
        env, coordinator = _build_env(experiment)
        observation = env.reset()
        for _ in range(args.iterations):
            with time_block(tracker, "coordination"):
                command, action = coordinator.act(observation)
            env.set_command(command)
            with time_block(tracker, "step"):
                observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation = env.reset()

    for name, metrics in tracker.summary().items():
        print(f"{name}: {metrics}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
