import pytest

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class CollisionSimulator:
    def __init__(self):
        self._step = 0

    def reset(self, experiment):
        self._step = 0
        return {
            "telemetry": {
                "distance_to_goal": 5.0,
                "speed_mps": 1.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }

    def step(self, action):
        self._step += 1
        collision = self._step >= 2
        return {
            "telemetry": {
                "distance_to_goal": max(0.0, 5.0 - self._step),
                "speed_mps": action.get("throttle", 0.0),
                "collision": collision,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }


class GoalSimulator:
    def reset(self, experiment):
        return {
            "telemetry": {
                "distance_to_goal": 0.4,
                "speed_mps": 1.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }

    def step(self, action):
        return {
            "telemetry": {
                "distance_to_goal": 0.2,
                "speed_mps": action.get("throttle", 0.0),
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }


class NoOpPerception:
    def build_observation_inputs(self, raw_rgb):
        return {"segmentation_mask": None, "detections": []}


@pytest.fixture
def experiment():
    return ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=5,
        seeds=SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=True),
    )


def test_collision_terminates_episode(experiment):
    env = AirSimEnv(
        experiment,
        simulator=CollisionSimulator(),
        perception=NoOpPerception(),
        reward_calculator=RewardCalculator(),
    )
    env.reset()
    _, _, terminated, truncated, info = env.step({"throttle": 1.0, "brake": 0.0, "steering": 0.0})
    assert not terminated
    _, _, terminated, truncated, info = env.step({"throttle": 1.0, "brake": 0.0, "steering": 0.0})
    assert terminated is True
    assert info["episode_step"] == 2


def test_goal_distance_triggers_termination(experiment):
    env = AirSimEnv(
        experiment,
        simulator=GoalSimulator(),
        perception=NoOpPerception(),
        reward_calculator=RewardCalculator(),
    )
    env.reset()
    _, _, terminated, truncated, info = env.step({"throttle": 0.0, "brake": 0.0, "steering": 0.0})
    assert terminated is True
    assert info["episode_step"] == 1
