from airsim_env.env import AirSimEnv
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class StubSimulator:
    def __init__(self) -> None:
        self.client = None
        self._step = 0

    def reset(self, experiment: ExperimentDefinition):  # noqa: ARG002
        self._step = 0
        return {
            "telemetry": {
                "distance_to_goal": 10.0,
                "progress_possible": True,
                "collision": False,
                "heading_deg": 0.0,
                "speed_mps": 0.0,
                "lane_mask_coverage_ratio": 1.0,
            }
        }

    def step(self, action):  # noqa: ARG002
        self._step += 1
        return {
            "telemetry": {
                "distance_to_goal": 10.0,
                "progress_possible": True,
                "collision": False,
                "heading_deg": 0.0,
                "speed_mps": 0.0,
                "lane_mask_coverage_ratio": 1.0,
            }
        }


class StubPerception:
    def build_observation_inputs(self, image):  # noqa: ARG002
        return {"segmentation_mask": None, "detections": []}


class RecordingRewardCalculator:
    def __init__(self) -> None:
        self.command_completed_flags: list[bool] = []

    def compute(self, vehicle_state, *, active_command, command_completed, progress_possible):  # noqa: ARG002
        self.command_completed_flags.append(command_completed)
        return (
            {
                "completion_bonus": 100.0 if command_completed else 0.0,
                "command_shaping": 0.0,
                "collision_penalty": 0.0,
                "idle_penalty": 0.0,
            },
            0.0,
        )


def _make_experiment() -> ExperimentDefinition:
    return ExperimentDefinition(
        scene="debug",
        vehicle="car",
        start_pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        goal_pose=Pose(x=10.0, y=0.0, z=0.0, yaw=0.0),
        horizon=10,
        seeds=SeedBundle(python=0, numpy=0, torch=0, airsim=0),
    )


def test_completion_awarded_once_until_rearmed():
    experiment = _make_experiment()
    simulator = StubSimulator()
    perception = StubPerception()
    reward_calc = RecordingRewardCalculator()
    env = AirSimEnv(
        experiment, simulator=simulator, perception=perception, reward_calculator=reward_calc
    )

    env.reset()

    # Initial step: no completion
    env.set_command("STOP", completed=False)
    env.step({})

    # First completion should be honored
    env.set_command("STOP", completed=True)
    env.step({})

    # Subsequent completions without re-arming should be ignored
    env.set_command("STOP", completed=True)
    env.step({})
    env.set_command("STOP", completed=True)
    env.step({})

    # Once completion signal drops, the bonus can trigger again
    env.set_command("STOP", completed=False)
    env.step({})
    env.set_command("STOP", completed=True)
    env.step({})

    assert reward_calc.command_completed_flags == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]
