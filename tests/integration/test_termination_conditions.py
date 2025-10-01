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
                "speed_mps": 0.0,
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
    """Simulator that starts near goal and moves to reach it."""

    def __init__(self):
        self._step_count = 0

    def reset(self, experiment):
        self._step_count = 0
        # For legacy goal_pose, PathManager creates path: [start_pose, goal_pose]
        # Start at start_pose (0, 0), which will be first waypoint
        return {
            "telemetry": {
                "distance_to_goal": 10.0,
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "position_xy": (0.0, 0.0),  # At start waypoint
            },
            "image": None,
        }

    def step(self, action):
        self._step_count += 1
        if self._step_count == 1:
            # First step: move past first waypoint (start) to advance to second waypoint (goal)
            return {
                "telemetry": {
                    "distance_to_goal": 5.0,
                    "speed_mps": 1.0,
                    "collision": False,
                    "lane_mask_coverage_ratio": 1.0,
                    "progress_possible": True,
                    "position_xy": (
                        3.0,
                        0.0,
                    ),  # Within threshold of start (0,0), will advance to goal waypoint
                },
                "image": None,
            }
        else:
            # Second step: reach goal waypoint at (10, 0)
            return {
                "telemetry": {
                    "distance_to_goal": 0.5,
                    "speed_mps": 1.0,
                    "collision": False,
                    "lane_mask_coverage_ratio": 1.0,
                    "progress_possible": True,
                    "position_xy": (9.5, 0.0),  # Within threshold of goal (10, 0)
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
    _, _, terminated, truncated, info = env.step({"target_speed": 2.0, "target_steering": 0.0})
    assert not terminated
    _, _, terminated, truncated, info = env.step({"target_speed": 2.0, "target_steering": 0.0})
    assert terminated is True
    assert info["episode_step"] == 2


def test_goal_distance_triggers_termination(experiment):
    """Test that reaching the final waypoint terminates the episode."""
    env = AirSimEnv(
        experiment,
        simulator=GoalSimulator(),
        perception=NoOpPerception(),
        reward_calculator=RewardCalculator(),
    )
    env.reset()

    # First step: advance past start waypoint
    _, _, terminated, truncated, info = env.step({"target_speed": 1.0, "target_steering": 0.0})
    assert not terminated, "Should not terminate after first step"

    # Second step: reach goal waypoint
    _, _, terminated, truncated, info = env.step({"target_speed": 1.0, "target_steering": 0.0})
    assert terminated is True, "Should terminate when final waypoint is reached"
    assert info["episode_step"] == 2
