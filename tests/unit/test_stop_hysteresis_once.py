from airsim_env.env import AirSimEnv
from config.experiment import ExperimentDefinition, Pose, SeedBundle
from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.workers.sac_worker import SACWorker


class OscillatingSimulator:
    def __init__(self, speeds):
        self.speeds = speeds
        self.i = 0
        self.client = None

    def reset(self, experiment):  # noqa: ARG002
        self.i = 0
        return {
            "telemetry": {
                "distance_to_goal": 10.0,
                "progress_possible": True,
                "collision": False,
                "heading_deg": 0.0,
                "speed_mps": self.speeds[self.i] if self.speeds else 0.0,
                "lane_mask_coverage_ratio": 1.0,
            }
        }

    def step(self, action):  # noqa: ARG002
        self.i = min(self.i + 1, len(self.speeds) - 1)
        return {
            "telemetry": {
                "distance_to_goal": 10.0,
                "progress_possible": True,
                "collision": False,
                "heading_deg": 0.0,
                "speed_mps": self.speeds[self.i],
                "lane_mask_coverage_ratio": 1.0,
            }
        }


class DummyPerception:
    def build_observation_inputs(self, image):  # noqa: ARG002
        return {"segmentation_mask": None, "detections": []}


def _make_experiment():
    return ExperimentDefinition(
        scene="debug",
        vehicle="car",
        start_pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        goal_pose=Pose(x=10.0, y=0.0, z=0.0, yaw=0.0),
        horizon=50,
        seeds=SeedBundle(python=0, numpy=0, torch=0, airsim=0),
    )


def test_stop_completion_awarded_once_with_hysteresis():
    # Speeds oscillate around threshold (0.2), with some short dips
    speeds = [
        0.3,
        0.19,
        0.21,
        0.18,
        0.22,
        0.19,
        0.2,
        0.19,
        0.25,
        0.18,
        0.15,
        0.14,
        0.13,
        0.12,
        0.11,
        0.1,
        0.09,
        0.3,
        0.05,
        0.04,
    ]

    exp = _make_experiment()
    env = AirSimEnv(exp, simulator=OscillatingSimulator(speeds), perception=DummyPerception())

    # Minimal coordinator with STOP-only policy
    policy = CommandPolicy(commands=["STOP"])
    manager = DQNManager(policy)
    worker = SACWorker()
    coord = CommandCoordinator(manager, {"STOP": worker})

    obs = env.reset()

    completion_count = 0

    # Run a short rollout in deterministic mode (no learning side-effects)
    for _ in range(30):
        completed = coord.command_completed(obs)
        if completed:
            completion_count += 1
        cmd, action = coord.act(obs, deterministic=True)
        env.set_command(cmd, completed=completed)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    # With hysteresis and single-award-per-command, completion should occur at most once
    assert completion_count <= 1
