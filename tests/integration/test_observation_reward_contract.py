import numpy as np
import pytest

from airsim_env.env import AirSimEnv
from airsim_env.observation import ObservationPacket
from airsim_env.reward import RewardCalculator, RewardConfig
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class StaticPerception:
    def capture_segmentation(self):
        return np.ones((4, 4), dtype=np.uint8)

    def run_detection(self, image):
        return [
            {"class_id": 1, "bbox": [0.5, 0.5, 0.1, 0.1], "confidence": 0.9},
        ]

    def build_observation_inputs(self, raw_rgb):
        return {
            "segmentation_mask": self.capture_segmentation(),
            "detections": self.run_detection(raw_rgb),
        }


class StubSimulator:
    def __init__(self):
        self.distance = 10.0

    def reset(self, experiment):
        self.distance = 10.0
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 1.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
            },
            "image": np.zeros((3, 64, 64), dtype=np.float32),
        }

    def step(self, action):
        self.distance = max(0.0, self.distance - 1.5)
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": action.get("throttle", 0.0),
                "collision": False,
                "lane_mask_coverage_ratio": 0.9,
                "progress_possible": True,
            },
            "image": np.zeros((3, 64, 64), dtype=np.float32),
        }


@pytest.fixture
def experiment():
    return ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=3,
        seeds=SeedBundle(python=101, numpy=102, torch=103, airsim=104, deterministic=True),
    )


def test_observation_contains_expected_fields(experiment):
    env = AirSimEnv(
        experiment,
        simulator=StubSimulator(),
        perception=StaticPerception(),
        reward_calculator=RewardCalculator(RewardConfig()),
    )

    observation = env.reset()

    assert isinstance(observation, ObservationPacket)
    assert observation.segmentation_mask.shape == (4, 4)
    assert observation.detections and observation.detections[0]["class_id"] == 1
    assert set(observation.reward_components) == {
        "progress",
        "lane_adherence",
        "collision",
        "command_completion",
        "idle_penalty",
    }

    observation, reward, terminated, truncated, info = env.step(
        {"throttle": 0.5, "brake": 0.0, "steering": 0.0}
    )
    assert isinstance(observation, ObservationPacket)
    assert isinstance(reward, float)
    assert "config_hash" in info
    assert terminated or truncated or info["episode_step"] >= 1
