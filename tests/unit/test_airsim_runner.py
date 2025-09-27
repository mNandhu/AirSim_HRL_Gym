from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import airsim_runner
from utils.airsim_runner import (
    SimulatorLaunchError,
    SimulatorSession,
    airsim_session,
    ensure_running,
    launch,
)


class DummyProcess:
    def __init__(self, pid: int = 1234, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive
        self._terminated = False

    def poll(self):
        return None if self._alive and not self._terminated else 0

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_launch_headless_includes_offscreen_flag(monkeypatch, settings_path):
    created_cmds = []

    def fake_popen(cmd, *args, **kwargs):  # noqa: ANN001 - signature matches subprocess
        created_cmds.append(cmd)
        return DummyProcess()

    monkeypatch.setattr(airsim_runner, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.setattr(airsim_runner, "_wait_for_start", lambda process, timeout: None)

    session = launch("headless", str(settings_path))

    cmd = created_cmds[0]
    assert "-RenderOffScreen" in cmd
    assert "-settings" in cmd
    assert session.mode == "headless"
    assert session.settings_path == str(settings_path)


def test_build_command_linux_headless(monkeypatch, settings_path):
    created_cmds = []

    def fake_popen(cmd, *args, **kwargs):  # noqa: ANN001 - mimic subprocess signature
        created_cmds.append(cmd)
        return DummyProcess()

    monkeypatch.setattr(airsim_runner, "IS_WINDOWS", False)
    monkeypatch.setattr(airsim_runner, "DEFAULT_EXECUTABLE", "./AirSimNH.sh")
    monkeypatch.setattr(airsim_runner, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.setattr(airsim_runner, "_wait_for_start", lambda process, timeout: None)

    launch("headless", str(settings_path))

    cmd = created_cmds[0]
    assert cmd[:2] == ["bash", "./AirSimNH.sh"]
    assert "-windowed" in cmd
    assert "-nosound" in cmd


def test_launch_retries_then_succeeds(monkeypatch, settings_path):
    attempts = []

    def fake_popen(cmd, *args, **kwargs):  # noqa: ANN001 - mimic subprocess signature
        attempts.append(datetime.now())
        if len(attempts) < 3:
            raise OSError("startup failed")
        return DummyProcess(pid=4321)

    monkeypatch.setattr(airsim_runner, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.setattr(airsim_runner, "_wait_for_start", lambda process, timeout: None)

    session = launch("headless", str(settings_path))
    assert session.pid == 4321
    assert len(attempts) == 3


def test_launch_raises_after_max_attempts(monkeypatch, settings_path):
    def fake_popen(*args, **kwargs):  # noqa: ANN001
        raise OSError("fail")

    monkeypatch.setattr(airsim_runner, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.setattr(airsim_runner, "time", SimpleNamespace(sleep=lambda _: None))

    with pytest.raises(SimulatorLaunchError):
        launch("gui", str(settings_path))


def test_ensure_running_checks_process(monkeypatch, settings_path):
    process = DummyProcess(alive=False)
    session = SimulatorSession(
        pid=process.pid, process=process, mode="headless", settings_path=str(settings_path)
    )

    with pytest.raises(RuntimeError):
        ensure_running(session)


def test_context_manager_terminates(monkeypatch, settings_path):
    process = DummyProcess()
    terminate_calls = []

    def fake_launch(mode, settings_path):  # noqa: ANN001
        return SimulatorSession(
            pid=process.pid, process=process, mode=mode, settings_path=settings_path
        )

    def fake_terminate(session, *, force=False):  # noqa: ANN001
        terminate_calls.append((session.pid, force))

    monkeypatch.setattr(airsim_runner, "launch", fake_launch)
    monkeypatch.setattr(airsim_runner, "terminate", fake_terminate)

    with airsim_session(mode="headless", settings_path=str(settings_path)) as session:
        assert session.pid == process.pid

    assert terminate_calls == [(process.pid, False)]
