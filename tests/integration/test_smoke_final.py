from pathlib import Path

import pytest

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator
from config.experiment import ExperimentDefinition, Pose, SeedBundle
from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.orchestrator import HRLOrchestrator
from hrl_agent.workers.sac_worker import SACWorker
from utils.artifacts import ArtifactManager


class LinearSimulator:
    def __init__(self, horizon: int) -> None:
        self._distance = float(horizon)

    def reset(self, experiment):
        self._distance = float(experiment.horizon)
        return {
            "telemetry": {
                "distance_to_goal": self._distance,
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }

    def step(self, action):
        throttle = action.get("throttle", 0.0)
        self._distance = max(0.0, self._distance - throttle)
        return {
            "telemetry": {
                "distance_to_goal": self._distance,
                "speed_mps": throttle,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }


class StaticPerception:
    def build_observation_inputs(self, raw_rgb):
        return {"segmentation_mask": None, "detections": []}


@pytest.fixture
def experiment_definition():
    return ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=5,
        seeds=SeedBundle(python=11, numpy=12, torch=13, airsim=14, deterministic=True),
    )


def test_smoke_seeded_run_creates_artifacts(tmp_path: Path, experiment_definition):
    simulator = LinearSimulator(experiment_definition.horizon)
    perception = StaticPerception()
    env = AirSimEnv(
        experiment_definition,
        simulator=simulator,
        perception=perception,
        reward_calculator=RewardCalculator(),
    )

    policy = CommandPolicy(commands=["FOLLOW_LANE", "STOP"])
    manager = DQNManager(policy=policy)
    workers = {command: SACWorker() for command in policy.commands}
    coordinator = CommandCoordinator(manager, workers)
    orchestrator = HRLOrchestrator(env, coordinator)

    artifact_manager = ArtifactManager(tmp_path)
    run_paths = artifact_manager.start_run(str(experiment_definition.id))

    result = orchestrator.run_episode(max_steps=experiment_definition.horizon, deterministic=True)

    artifact_path = artifact_manager.write_json(
        "result.json",
        {
            "cumulative_reward": result.cumulative_reward,
            "steps": result.steps,
            "info": result.info,
        },
    )

    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8")
    assert run_paths.run_dir.exists()
