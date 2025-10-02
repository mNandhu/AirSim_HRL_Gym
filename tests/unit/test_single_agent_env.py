import numpy as np
import pytest

from airsim_env.reward import RewardCalculator, RewardConfig
from airsim_env.single_agent_env import SingleAgentAirSimEnv
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class DummyPerception:
    def build_observation_inputs(self, _image):
        return {"segmentation_mask": None, "detections": []}


class DummySimulator:
    def __init__(self) -> None:
        self.last_control: dict[str, float] | None = None
        self._distance = 30.0
        self._position = (0.0, 0.0)

    def reset(self, experiment):
        self._distance = 30.0
        self._position = (float(experiment.start_pose.x), float(experiment.start_pose.y))
        self.last_control = None
        return {
            "telemetry": {
                "distance_to_goal": self._distance,
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
                "position_xy": self._position,
            },
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def step(self, control):
        self.last_control = dict(control)
        throttle = float(control.get("throttle", 0.0))
        brake = float(control.get("brake", 0.0))
        net_speed = max(throttle - brake, 0.0) * 5.0
        self._distance = max(0.0, self._distance - max(net_speed, 1.0))
        self._position = (self._position[0] + net_speed, self._position[1])
        return {
            "telemetry": {
                "distance_to_goal": self._distance,
                "speed_mps": net_speed,
                "collision": False,
                "lane_mask_coverage_ratio": 0.9,
                "progress_possible": True,
                "heading_deg": 5.0,
                "position_xy": self._position,
            },
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
        }


@pytest.fixture
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition(
        scene="TestTown",
        vehicle="Sedan",
        start_pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        waypoints=[
            Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            Pose(x=10.0, y=0.0, z=0.0, yaw=0.0),
        ],
        horizon=20,
        seeds=SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=True),
    )


@pytest.fixture
def env_setup(experiment):
    simulator = DummySimulator()
    perception = DummyPerception()
    env = SingleAgentAirSimEnv(
        experiment,
        simulator=simulator,
        perception=perception,
        reward_calculator=RewardCalculator(RewardConfig()),
        target_speed_range=(0.0, 10.0),
    )
    yield env, simulator
    env.close()


def test_reset_returns_observation_dict(env_setup):
    env, _ = env_setup
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "telemetry" in obs
    assert obs["telemetry"].shape == (8,)
    assert info["episode_step"] == 0
    assert set(info["reward_components"]) == {
        "command_shaping",
        "collision_penalty",
        "completion_bonus",
        "idle_penalty",
        "time_penalty",
    }
    assert env.render() is not None
    assert env.experiment.scene == "TestTown"
    assert env.last_info["episode_step"] == 0
    assert env.get_wrapped_env() is not None
    env.close()
    assert env.render() is None


def test_scale_action_clamps_values(env_setup):
    env, _ = env_setup
    env.reset()
    speed, steering, normalized = env._scale_action(np.array([3.0, -3.0], dtype=np.float32))
    assert pytest.approx(speed, rel=1e-6) == 10.0
    assert pytest.approx(steering, rel=1e-6) == -1.0
    assert normalized.tolist() == [1.0, -1.0]


def test_scale_action_accepts_mapping(env_setup):
    env, _ = env_setup
    env.reset()
    speed, steering, normalized = env._scale_action({"target_speed": -5.0, "target_steering": -2.0})
    assert speed == pytest.approx(0.0)
    assert steering == -1.0
    assert normalized.tolist() == [-1.0, -1.0]


def test_step_applies_control_and_returns_info(env_setup):
    env, simulator = env_setup
    env.reset()
    action = np.array([0.5, 0.25], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert simulator.last_control is not None
    assert 0.0 <= simulator.last_control["throttle"] <= 1.0
    assert -1.0 <= simulator.last_control["steering"] <= 1.0
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "applied_action" in info
    assert "telemetry" in info
    assert obs["telemetry"].dtype == np.float32


def test_observation_vector_is_bounded(env_setup):
    env, _ = env_setup
    obs, _ = env.reset()
    telemetry = obs["telemetry"]
    assert telemetry.shape == (8,)
    assert np.all(telemetry <= 1.0)
    assert np.all(telemetry >= -1.0)


def test_invalid_action_shape_raises(env_setup):
    env, _ = env_setup
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([0.1], dtype=np.float32))


def test_invalid_speed_range_raises(experiment):
    simulator = DummySimulator()
    perception = DummyPerception()
    with pytest.raises(ValueError):
        SingleAgentAirSimEnv(
            experiment,
            simulator=simulator,
            perception=perception,
            target_speed_range=(5.0, 2.0),
        )
