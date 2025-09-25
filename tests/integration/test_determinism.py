import pytest

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class DeterministicSimulator:
    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
        self._progress = list(reversed(range(horizon)))

    def reset(self, experiment):
        return {
            "telemetry": {
                "distance_to_goal": float(experiment.horizon),
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": None,
        }

    def step(self, action):
        remaining = self._progress.pop() if self._progress else 0
        return {
            "telemetry": {
                "distance_to_goal": float(remaining),
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
def experiment_definition():
    return ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=3,
        seeds=SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=True),
    )


def _run_episode(env: AirSimEnv):
    obs = env.reset()
    done = False
    total = 0.0
    info = {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(
            {"throttle": 1.0, "brake": 0.0, "steering": 0.0}
        )
        total += reward
        done = terminated or truncated
    return total, info


def test_deterministic_runs_match(experiment_definition):
    simulator = DeterministicSimulator(experiment_definition.horizon)
    perception = NoOpPerception()
    reward = RewardCalculator()

    env1 = AirSimEnv(
        experiment_definition, simulator=simulator, perception=perception, reward_calculator=reward
    )
    total_1, info_1 = _run_episode(env1)

    simulator2 = DeterministicSimulator(experiment_definition.horizon)
    env2 = AirSimEnv(
        experiment_definition.model_copy(),
        simulator=simulator2,
        perception=perception,
        reward_calculator=reward,
    )
    total_2, info_2 = _run_episode(env2)

    assert total_1 == pytest.approx(total_2)
    assert info_1["config_hash"] == info_2["config_hash"]
