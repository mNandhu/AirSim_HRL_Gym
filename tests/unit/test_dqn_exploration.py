import numpy as np

from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager


class _ActionSpaceStub:
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        # Always sample the last action for determinism
        return self.n - 1


class _ReplayBufferStub:
    def add(self, *args, **kwargs):
        return None


class _ModelStub:
    def __init__(self) -> None:
        self.exploration_rate = 1.0
        self.replay_buffer = _ReplayBufferStub()
        self.learning_starts = 0
        self.batch_size = 32
        self._total_timesteps = 0
        self._n_updates = 0

    def predict(self, observation, deterministic=True):  # noqa: ARG002
        # Greedy policy always returns action 0
        return 0, None

    def train(self, batch_size: int, gradient_steps: int):  # noqa: ARG002
        # No-op training
        return None


def test_select_command_uses_exploration_when_enabled(monkeypatch):
    # Arrange
    policy = CommandPolicy(commands=("A", "B"))
    mgr = DQNManager(
        policy,
        model=_ModelStub(),
        observation_space=None,
        action_space=_ActionSpaceStub(2),
        exploration_fraction=0.8,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        total_timesteps=100,
    )

    # Force np.random.rand to be < eps to take the exploratory branch
    monkeypatch.setattr(np.random, "rand", lambda *args, **kwargs: 0.0)

    # Act
    cmd = mgr.select_command(np.zeros(1, dtype=np.float32), deterministic=False)

    # Assert: with exploration, we should sample action index 1 -> command "B"
    assert cmd == "B"


def test_process_experience_updates_epsilon_schedule():
    # Arrange
    policy = CommandPolicy(commands=("A", "B"))
    model = _ModelStub()
    mgr = DQNManager(
        policy,
        model=model,
        observation_space=None,
        action_space=_ActionSpaceStub(2),
        exploration_fraction=0.5,  # exploration over first 50% of steps
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        total_timesteps=10,
    )

    # Act: one experience step
    mgr.process_experience(
        observation=np.zeros(1, dtype=np.float32),
        action_index=0,
        reward=0.0,
        next_observation=np.zeros(1, dtype=np.float32),
        done=False,
    )

    # For total_timesteps=10 and exploration_fraction=0.5 -> 5 steps in exploration phase
    # After 1 step, progress=0.2 -> eps ~= 0.82
    assert 0.8 < model.exploration_rate < 0.9
