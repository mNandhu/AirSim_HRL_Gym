import numpy as np

from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.workers.sac_worker import SACWorker


class DeterministicWorker(SACWorker):
    def __init__(self, action):
        super().__init__()
        self._action = action

    def act(self, observation, *, deterministic=True):
        assert isinstance(observation, np.ndarray)
        return self._action


class DummyManager(DQNManager):
    def __init__(self):
        super().__init__(CommandPolicy(commands=["A", "B"]))
        self.attach_model(object())

    def select_command(self, observation, *, deterministic=True):
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (5,)
        return "A"


def test_coordinator_act_returns_command_and_action():
    manager = DummyManager()
    workers = {
        "A": DeterministicWorker({"throttle": 1.0}),
        "B": DeterministicWorker({"throttle": 0.0}),
    }
    coordinator = CommandCoordinator(manager, workers)

    observation = {"telemetry": {"distance_to_goal": 10.0, "speed_mps": 5.0}}
    command, action = coordinator.act(observation)

    assert command == "A"
    assert action["throttle"] == 1.0


def test_act_sequential_cycles_workers():
    manager = DummyManager()
    workers = {
        "A": DeterministicWorker({"throttle": 1.0}),
        "B": DeterministicWorker({"throttle": 2.0}),
    }
    coordinator = CommandCoordinator(manager, workers)

    observation = {"telemetry": {"distance_to_goal": 20.0, "speed_mps": 3.0}}
    cmd1, action1 = coordinator.act_sequential(observation)
    cmd2, action2 = coordinator.act_sequential(observation)

    assert (cmd1, action1["throttle"]) == ("A", 1.0)
    assert (cmd2, action2["throttle"]) == ("B", 2.0)
