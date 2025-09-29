from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.workers.sac_worker import SACWorker


class FakeModel:
    def predict(self, observation, deterministic=True):
        return 2, None

    def save(self, path):
        Path(path).write_text("model", encoding="utf-8")


def test_dqn_manager_selects_from_model(tmp_path):
    policy = CommandPolicy(commands=["LEFT", "RIGHT", "STRAIGHT"])
    manager = DQNManager(policy)
    manager.attach_model(FakeModel())

    command = manager.select_command({})
    assert command == "STRAIGHT"

    out_path = tmp_path / "model.zip"
    manager.save(str(out_path))
    assert out_path.exists()


def test_sac_worker_returns_default_action():
    worker = SACWorker()
    action = worker.act({}, deterministic=True)
    assert set(action.keys()) == {"target_speed", "target_steering"}


def test_sac_worker_handles_scalar_action():
    class ScalarModel:
        def predict(self, observation, deterministic=True):
            return 0.75, None

    worker = SACWorker()
    worker.attach_model(ScalarModel())
    action = worker.act(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action["target_speed"] == pytest.approx(0.75)
    assert action["target_steering"] == 0.0


def test_sac_worker_attach_and_save(tmp_path):
    class MockSAC:
        def __init__(self) -> None:
            self.saved = False

        def predict(self, observation, deterministic=True):
            return [0.1, -0.3], None

        def save(self, path):
            Path(path).write_text("saved", encoding="utf-8")

    worker = SACWorker()
    worker.attach_model(MockSAC())
    action = worker.act({}, deterministic=False)
    assert pytest.approx(action["target_speed"], rel=1e-6) == 0.1

    out_path = tmp_path / "worker.zip"
    worker.save(str(out_path))
    assert out_path.exists()


def test_sac_worker_save_requires_model():
    worker = SACWorker()
    with pytest.raises(RuntimeError):
        worker.save("unused.zip")


def test_sac_worker_load_requires_sb3(monkeypatch):
    worker = SACWorker()
    monkeypatch.setattr("hrl_agent.workers.sac_worker.SAC", None)
    with pytest.raises(RuntimeError):
        worker.load_from_path("model.zip")


def test_dqn_manager_policy_updates():
    manager = DQNManager(CommandPolicy(commands=["A", "B"]))
    manager.set_policy(CommandPolicy(commands=["X", "Y"]))
    assert manager.select_command({}) == "X"


def test_dqn_manager_load_requires_sb3(monkeypatch):
    manager = DQNManager(CommandPolicy(commands=["A", "B"]))
    monkeypatch.setattr("hrl_agent.manager.dqn_manager.DQN", None)
    with pytest.raises(RuntimeError):
        manager.load_from_path("model.zip")


def test_sac_worker_process_experience_trains_and_logs():
    class DummyReplayBuffer:
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
                ]
            ] = []

        def add(self, obs, next_obs, action, reward_arr, done_arr, infos):
            self.calls.append((obs, next_obs, action, reward_arr, done_arr, infos))

    class DummyLogger:
        def __init__(self) -> None:
            self.records: list[tuple[str, float]] = []
            self.dumps: list[int] = []

        def record(self, key: str, value: float) -> None:
            self.records.append((key, value))

        def dump(self, step: int) -> None:  # noqa: D401 - mimic sb3 logger API
            self.dumps.append(step)

    class DummyModel:
        def __init__(self) -> None:
            self.replay_buffer = DummyReplayBuffer()
            self.learning_starts = 1
            self.batch_size = 4
            self.logger = DummyLogger()
            self._total_timesteps = 0
            self._n_updates = 0
            self.train_args: list[tuple[int, int]] = []

        def train(self, *, batch_size: int, gradient_steps: int) -> None:
            self.train_args.append((batch_size, gradient_steps))

    worker = SACWorker()
    model = DummyModel()
    worker.attach_model(model)
    worker._steps = 99  # Force logging branch on next experience

    worker.process_experience(
        observation=np.zeros(5, dtype=np.float32),
        action=np.array([0.1, -0.3], dtype=np.float32),
        reward=1.5,
        next_observation=np.ones(5, dtype=np.float32),
        done=True,
    )

    assert len(model.replay_buffer.calls) == 1
    assert model.train_args == [(model.batch_size, 1)]
    assert model._total_timesteps == 1
    assert model._n_updates == 1
    assert model.logger.dumps == [100]
    assert any(key == "train/reward" for key, _ in model.logger.records)


def test_sac_worker_handles_empty_and_overflow_actions():
    class EmptyModel:
        def predict(self, observation, deterministic=True):
            return np.array([], dtype=np.float32), None

    class WideModel:
        def predict(self, observation, deterministic=True):
            return np.array([0.9, 0.8, -0.4, 0.3], dtype=np.float32), None

    worker = SACWorker()
    worker.attach_model(EmptyModel())
    empty_action = worker.act({}, deterministic=True)
    assert empty_action == {"target_speed": 0.0, "target_steering": 0.0}

    worker.attach_model(WideModel())
    rich_action = worker.act({}, deterministic=True)
    assert rich_action["target_speed"] == pytest.approx(0.9 * worker.TARGET_SPEED_MAX)
    assert rich_action["target_steering"] == pytest.approx(-0.4)


def test_dqn_manager_process_experience_trains_and_logs():
    class DummyReplayBuffer:
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
                ]
            ] = []

        def add(self, obs, next_obs, action, reward_arr, done_arr, infos):
            self.calls.append((obs, next_obs, action, reward_arr, done_arr, infos))

    class DummyLogger:
        def __init__(self) -> None:
            self.records: list[tuple[str, float]] = []
            self.dumps: list[int] = []

        def record(self, key: str, value: float) -> None:
            self.records.append((key, value))

        def dump(self, step: int) -> None:
            self.dumps.append(step)

    class DummyModel:
        def __init__(self) -> None:
            self.replay_buffer = DummyReplayBuffer()
            self.learning_starts = 1
            self.batch_size = 2
            self.logger = DummyLogger()
            self._total_timesteps = 0
            self._n_updates = 0
            self.train_args: list[tuple[int, int]] = []

        def train(self, *, batch_size: int, gradient_steps: int) -> None:
            self.train_args.append((batch_size, gradient_steps))

    manager = DQNManager(CommandPolicy(commands=["LEFT", "RIGHT"]))
    model = DummyModel()
    manager.attach_model(model)
    manager._steps = 99

    manager.process_experience(
        observation=np.zeros(5, dtype=np.float32),
        action_index=1,
        reward=2.0,
        next_observation=np.ones(5, dtype=np.float32),
        done=False,
    )

    assert len(model.replay_buffer.calls) == 1
    assert model.train_args == [(model.batch_size, 1)]
    assert model._total_timesteps == 1
    assert model._n_updates == 1
    assert model.logger.dumps == [100]
    assert any(key == "train/command_selection_step" for key, _ in model.logger.records)
