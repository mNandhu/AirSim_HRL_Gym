from pathlib import Path

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
    assert set(action.keys()) == {"throttle", "brake", "steering"}


def test_sac_worker_handles_scalar_action():
    class ScalarModel:
        def predict(self, observation, deterministic=True):
            return 0.75, None

    worker = SACWorker()
    worker.attach_model(ScalarModel())
    action = worker.act(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action["throttle"] == pytest.approx(0.75)
    assert action["brake"] == 0.0
    assert action["steering"] == 0.0


def test_sac_worker_attach_and_save(tmp_path):
    class MockSAC:
        def __init__(self) -> None:
            self.saved = False

        def predict(self, observation, deterministic=True):
            return [0.1, 0.2, -0.3], None

        def save(self, path):
            Path(path).write_text("saved", encoding="utf-8")

    worker = SACWorker()
    worker.attach_model(MockSAC())
    action = worker.act({}, deterministic=False)
    assert pytest.approx(action["throttle"], rel=1e-6) == 0.1

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
