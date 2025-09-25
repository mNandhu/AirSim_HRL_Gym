from pathlib import Path

import pytest

from utils import airsim_runner
from utils.airsim_runner import SimulatorLaunchError, SimulatorSession


class DummyProcess:
    def __init__(self, pid: int, alive: bool) -> None:
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_restart_after_crash(monkeypatch, settings_path):
    launches = []
    processes = iter(
        [
            DummyProcess(pid=111, alive=False),
            DummyProcess(pid=222, alive=True),
        ]
    )

    def fake_launch(mode, settings_path):  # noqa: ANN001
        process = next(processes)
        session = SimulatorSession(
            pid=process.pid, process=process, mode=mode, settings_path=settings_path
        )
        launches.append(session)
        return session

    monkeypatch.setattr(airsim_runner, "launch", fake_launch)
    monkeypatch.setattr(airsim_runner, "terminate", lambda session, force=False: None)

    session = airsim_runner.launch("headless", str(settings_path))

    with pytest.raises(RuntimeError):
        airsim_runner.ensure_running(session)

    restarted = airsim_runner.restart(session, mode="headless")
    assert restarted.pid == 222
    assert len(launches) == 2


def test_restart_respects_max_attempts(monkeypatch, settings_path):
    monkeypatch.setattr(
        airsim_runner,
        "launch",
        lambda *args, **kwargs: (_ for _ in ()).throw(SimulatorLaunchError("fail")),
    )
    monkeypatch.setattr(airsim_runner, "terminate", lambda session, force=False: None)

    with pytest.raises(SimulatorLaunchError):
        airsim_runner.restart(
            SimulatorSession(
                pid=999,
                process=DummyProcess(pid=999, alive=False),
                mode="headless",
                settings_path=str(settings_path),
            ),
            mode="headless",
            max_attempts=2,
        )
