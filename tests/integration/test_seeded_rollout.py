from collections import deque

import pytest

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator, RewardConfig
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class DummyPerception:
    def __init__(self) -> None:
        self.calls = 0

    def capture_segmentation(self):
        self.calls += 1
        return None

    def run_detection(self, image):
        self.calls += 1
        return []

    def build_observation_inputs(self, raw_rgb):
        return {"segmentation_mask": None, "detections": []}


class DummySimulator:
    def __init__(self) -> None:
        self.steps = deque()

    def reset(self, experiment: ExperimentDefinition):
        self.steps = deque([{"progress_possible": True} for _ in range(experiment.horizon)])
        return {
            "telemetry": {
                "distance_to_goal": float(experiment.horizon),
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
            },
            "image": None,
        }

    def step(self, action):
        if not self.steps:
            raise RuntimeError("Simulator exceeded horizon")
        progress = self.steps.popleft()
        return {
            "telemetry": {
                "distance_to_goal": float(len(self.steps)),
                "speed_mps": action.get("throttle", 0.0),
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": progress.get("progress_possible", True),
            },
            "image": None,
        }


@pytest.fixture
def experiment_definition():
    return ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=5,
        seeds=SeedBundle(python=42, numpy=43, torch=44, airsim=45, deterministic=True),
    )


def run_episode(env: AirSimEnv, *, action=None):
    obs = env.reset()
    done = False
    cumulative_reward = 0.0
    info = {}

    while not done:
        obs, reward, terminated, truncated, info = env.step(
            action or {"throttle": 0.5, "brake": 0.0, "steering": 0.0}
        )
        cumulative_reward += reward
        done = terminated or truncated

    return cumulative_reward, info["config_hash"]


def test_seeded_rollout_is_deterministic(experiment_definition):
    simulator = DummySimulator()
    perception = DummyPerception()
    reward = RewardCalculator(
        RewardConfig(completion_bonus=2.0, idle_penalty_coef=0.5, idle_threshold=0.1)
    )

    env = AirSimEnv(
        experiment_definition, simulator=simulator, perception=perception, reward_calculator=reward
    )
    total_1, hash_1 = run_episode(env)

    # Recreate environment with identical seeds
    simulator2 = DummySimulator()
    perception2 = DummyPerception()
    env2 = AirSimEnv(
        experiment_definition.model_copy(),
        simulator=simulator2,
        perception=perception2,
        reward_calculator=reward,
    )

    total_2, hash_2 = run_episode(env2)

    assert total_1 == pytest.approx(total_2)
    assert hash_1 == hash_2


def test_environment_raises_when_exceeding_horizon(experiment_definition):
    simulator = DummySimulator()
    simulator.steps = deque()
    perception = DummyPerception()
    reward = RewardCalculator()

    env = AirSimEnv(
        experiment_definition, simulator=simulator, perception=perception, reward_calculator=reward
    )

    env.reset()
    simulator.steps.clear()
    with pytest.raises(RuntimeError):
        env.step({"throttle": 0.0, "brake": 0.0, "steering": 0.0})
