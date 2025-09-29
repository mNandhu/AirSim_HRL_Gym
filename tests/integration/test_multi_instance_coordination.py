"""Test multi-instance coordination to ensure trainers connect to the correct simulators."""

from __future__ import annotations

import subprocess
from pathlib import Path
import os

import pytest


from scripts.parallel_orchestrator import main as orchestrator_main


@pytest.mark.integration
def test_multi_instance_coordination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that multiple trainers connect to their assigned simulators without port conflicts."""
    # Create dummy settings.json
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    # Create a dummy script for the training job
    trainer_script_path = tmp_path / "dummy_trainer.py"
    trainer_script_path.write_text(
        """
import os
import airsim

port = os.environ.get("AIRSIM_PORT")
with open(f"trainer_{port}.log", "w") as f:
    f.write(f"Connected to port {port}")
client = airsim.CarClient(port=int(port))
client.confirmConnection()
""",
        encoding="utf-8",
    )

    # Create parallel config
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        f"""
telemetry:
  interval_seconds: 1
airsim_instances:
  - name: sim1
    settings: {settings_path.as_posix()}
  - name: sim2
    settings: {settings_path.as_posix()}
training_jobs:
  - name: trainer1
    command: ["python", "{trainer_script_path.as_posix()}"]
  - name: trainer2
    command: ["python", "{trainer_script_path.as_posix()}"]
""",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    # Mock airsim.CarClient to avoid real connection attempts
    class MockCarClient:
        def __init__(self, ip="127.0.0.1", port=41451):
            self.port = port

        def confirmConnection(self):
            pass

    monkeypatch.setattr("airsim.CarClient", MockCarClient)

    # Mock the subprocess.Popen to simulate trainer behavior and avoid external deps
    original_popen = subprocess.Popen
    popen_calls: list[list[str]] = []

    class DummyProcess:
        def __init__(self, pid: int):
            self.pid = pid
            self._polls = [None, 0]

        def poll(self):
            if len(self._polls) > 1:
                return self._polls.pop(0)
            return self._polls[0]

        def wait(self):
            self._polls = [0]
            return 0

        def send_signal(self, _signal: int) -> None:
            self._polls = [0]

    def mock_popen(command, *args, **kwargs):
        # Detect our dummy trainer by filename even if full path is provided
        if any(str(arg).endswith("dummy_trainer.py") for arg in command):
            env = kwargs.get("env", os.environ)
            port = env.get("AIRSIM_PORT", "41451")
            log_path = tmp_path / f"trainer_{port}.log"
            log_path.write_text(f"Connected to port {port}", encoding="utf-8")
            popen_calls.append(["python", command[-1]])
            return DummyProcess(pid=3000)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr("scripts.parallel_orchestrator.subprocess.Popen", mock_popen)

    # Mock the airsim_runner.launch to avoid launching real simulators
    class MockSimulatorSession:
        def __init__(self, pid):
            self.pid = pid
            self.process = subprocess.Popen(["python", "-c", "import time; time.sleep(2)"])

    def mock_launch(*args, **kwargs):
        return MockSimulatorSession(pid=1234)

    monkeypatch.setattr("utils.airsim_runner.launch", mock_launch)

    # Run the orchestrator
    exit_code = orchestrator_main(["--config", str(config_path), "--run-id", "test-multi-instance"])
    assert exit_code == 0

    # Check that the dummy trainer scripts were run and connected to the correct ports
    log1_path = tmp_path / "trainer_41451.log"
    log2_path = tmp_path / "trainer_41452.log"
    assert log1_path.exists()
    assert log2_path.exists()
    assert log1_path.read_text() == "Connected to port 41451"
    assert log2_path.read_text() == "Connected to port 41452"
    assert len(popen_calls) == 2
    assert popen_calls[0][0] == "python"
    assert popen_calls[1][0] == "python"
