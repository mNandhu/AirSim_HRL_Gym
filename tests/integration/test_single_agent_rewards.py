import numpy as np
import pytest

from airsim_env.reward import RewardCalculator, RewardConfig
from airsim_env.single_agent_env import SingleAgentAirSimEnv
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class StaticPerception:
    def build_observation_inputs(self, _image):
        return {"segmentation_mask": None, "detections": []}


class ProgressSimulator:
    def __init__(self) -> None:
        self.distance = 12.0
        self.position = [0.0, 0.0]

    def reset(self, experiment):
        self.distance = 12.0
        self.position = [float(experiment.start_pose.x), float(experiment.start_pose.y)]
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
                "position_xy": tuple(self.position),
            },
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def step(self, control):
        speed = max(float(control.get("throttle", 0.0)) - float(control.get("brake", 0.0)), 0.0)
        stride = max(speed * 8.0, 8.0)
        self.position[0] += stride
        self.distance = max(0.0, self.distance - stride)
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": stride,
                "collision": False,
                "lane_mask_coverage_ratio": 0.95,
                "progress_possible": True,
                "heading_deg": 0.0,
                "position_xy": tuple(self.position),
            },
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
        }


@pytest.fixture
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition(
        scene="TestTown",
        vehicle="Sedan",
        start_pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        waypoints=[
            Pose(x=8.0, y=0.0, z=0.0, yaw=0.0),
            Pose(x=16.0, y=0.0, z=0.0, yaw=0.0),
        ],
        horizon=10,
        seeds=SeedBundle(python=10, numpy=11, torch=12, airsim=13, deterministic=True),
    )


@pytest.fixture
def env(experiment):
    simulator = ProgressSimulator()
    perception = StaticPerception()
    environment = SingleAgentAirSimEnv(
        experiment,
        simulator=simulator,
        perception=perception,
        reward_calculator=RewardCalculator(RewardConfig()),
        target_speed_range=(0.0, 8.0),
    )
    yield environment
    environment.close()


def test_reward_components_without_command(env):
    env.reset()
    action = np.array([0.8, 0.0], dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    components = info["reward_components"]

    assert pytest.approx(reward, rel=1e-6) == sum(components.values())
    assert pytest.approx(components["command_shaping"], abs=1e-6) == 0.0
    assert "collision_penalty" in components


def test_waypoint_completion_sets_goal_flag(env):
    env.reset()
    info = {}
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert info.get("goal_reached") is True
