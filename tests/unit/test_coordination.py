from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.workers.sac_worker import SACWorker


class DeterministicWorker(SACWorker):
    def __init__(self, action):
        super().__init__()
        self._action = action

    def act(self, observation, *, deterministic=True):
        return self._action


class DummyManager(DQNManager):
    def __init__(self):
        super().__init__(CommandPolicy(commands=["A", "B"]))
        self.attach_model(object())

    def select_command(self, observation, *, deterministic=True):
        return "A"


def test_coordinator_act_returns_command_and_action():
    manager = DummyManager()
    workers = {
        "A": DeterministicWorker({"throttle": 1.0}),
        "B": DeterministicWorker({"throttle": 0.0}),
    }
    coordinator = CommandCoordinator(manager, workers)

    command, action = coordinator.act({})

    assert command == "A"
    assert action["throttle"] == 1.0


def test_act_sequential_cycles_workers():
    manager = DummyManager()
    workers = {
        "A": DeterministicWorker({"throttle": 1.0}),
        "B": DeterministicWorker({"throttle": 2.0}),
    }
    coordinator = CommandCoordinator(manager, workers)

    cmd1, action1 = coordinator.act_sequential({})
    cmd2, action2 = coordinator.act_sequential({})

    assert (cmd1, action1["throttle"]) == ("A", 1.0)
    assert (cmd2, action2["throttle"]) == ("B", 2.0)
