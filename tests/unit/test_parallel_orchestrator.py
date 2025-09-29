from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, cast

import pytest

import scripts.parallel_orchestrator as orchestrator_script
from scripts.parallel_orchestrator import (
    launch_airsim_instances,
    launch_training_jobs,
    load_config,
    monitor_telemetry,
    query_gpu_utilization,
)
from utils.airsim_runner import SimulatorSession
from utils.artifacts import ArtifactManager


class DummyProcess:
    def __init__(self, pid: int, polls: list[Any] | None = None) -> None:
        self.pid = pid
        self._polls = polls or [None]

    def poll(self) -> Any:
        if len(self._polls) > 1:
            return self._polls.pop(0)
        return self._polls[0]

    def wait(self) -> int:
        self._polls = [0]
        return 0

    def send_signal(self, _signal: int) -> None:
        self._polls = [0]


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("airsim_instances: []\n", encoding="utf-8")
    data = load_config(config_path)
    assert data["airsim_instances"] == []


def test_launch_training_jobs_sets_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[dict[str, Any]] = []

    def fake_popen(command, env=None, cwd=None):  # type: ignore[no-untyped-def]
        captures.append({"command": command, "env": env, "cwd": cwd})
        return DummyProcess(pid=4321, polls=[None])  # type: ignore[return-value]

    monkeypatch.setattr("scripts.parallel_orchestrator.subprocess.Popen", fake_popen)

    jobs = launch_training_jobs(
        [
            {
                "name": "trainer",
                "command": ["echo", "hello"],
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
                "cwd": None,
            }
        ],
        base_port=41451,
    )

    assert "trainer" in jobs
    assert captures[0]["command"] == ["echo", "hello"]
    assert captures[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_launch_airsim_instances_invokes_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_launch(mode, settings_path, **kwargs):  # type: ignore[no-untyped-def]
        return SimulatorSession(
            pid=1234,
            process=DummyProcess(pid=1234),
            mode=mode,
            settings_path=settings_path,
        )

    monkeypatch.setattr("scripts.parallel_orchestrator.airsim_runner.launch", fake_launch)

    sessions = launch_airsim_instances(
        [
            {
                "name": "sim0",
                "mode": "headless",
                "settings": "settings.json",
            }
        ],
        run_dir=tmp_path,
        base_port=41451,
    )

    assert "sim0" in sessions
    assert sessions["sim0"].pid == 1234


def test_launch_airsim_instances_platform_specific_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str] | None] = []

    def fake_launch(mode, settings_path, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(kwargs.get("command"))
        return SimulatorSession(
            pid=5555,
            process=DummyProcess(pid=5555),
            mode=mode,
            settings_path=settings_path,
        )

    monkeypatch.setattr("scripts.parallel_orchestrator.airsim_runner.launch", fake_launch)
    monkeypatch.setattr("scripts.parallel_orchestrator.IS_WINDOWS", False)

    launch_airsim_instances(
        [
            {
                "name": "sim-linux",
                "mode": "headless",
                "settings": "settings.json",
                "command_windows": ["windows"],
                "command_linux": ["linux"],
            }
        ],
        run_dir=tmp_path,
        base_port=41451,
    )

    assert commands[0] == ["linux"]


def test_monitor_telemetry_records_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process = DummyProcess(pid=999, polls=[None, 0])
    jobs = {"job": process}
    stop_event = threading.Event()
    artifact_manager = ArtifactManager(tmp_path)
    paths = artifact_manager.start_run("unit-test-run")
    monitor_telemetry(
        cast(dict[str, subprocess.Popen[Any]], jobs),
        artifact_manager,
        interval=0.001,
        gpu_command=None,
        stop_event=stop_event,
    )
    log_path = paths.logs_dir / "telemetry.jsonl"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents


def test_query_gpu_utilization_missing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.parallel_orchestrator.subprocess.check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert query_gpu_utilization("nvidia-smi") == []


def test_query_gpu_utilization_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = "GPU-123, RTX, 50 %, 100 MiB, 200 MiB"
    monkeypatch.setattr(
        "scripts.parallel_orchestrator.subprocess.check_output",
        lambda *args, **kwargs: sample,
    )
    stats = query_gpu_utilization("nvidia-smi")
    assert stats[0]["uuid"] == "GPU-123"
    assert stats[0]["utilization"] == "50 %"


def test_monitor_telemetry_includes_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process = DummyProcess(pid=111, polls=[None, 0])
    jobs = {"job": process}
    artifact_manager = ArtifactManager(tmp_path)
    artifact_manager.start_run("telemetry")

    records: list[dict[str, Any]] = []

    def fake_append(name: str, payload: dict[str, Any]) -> None:  # noqa: ANN001
        records.append(payload)

    monkeypatch.setattr(artifact_manager, "append_jsonl", fake_append)
    monkeypatch.setattr(
        "scripts.parallel_orchestrator.query_gpu_utilization", lambda cmd: [{"uuid": "gpu"}]
    )

    stop_event = threading.Event()
    monitor_telemetry(
        cast(dict[str, subprocess.Popen[Any]], jobs),
        artifact_manager,
        interval=0.001,
        gpu_command="nvidia-smi",
        stop_event=stop_event,
    )

    assert records
    assert records[0]["gpu"][0]["uuid"] == "gpu"


def test_main_runs_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
airsim_instances:
  - name: sim0
    settings: {settings_path.as_posix()}
training_jobs:
  - name: job0
    command:
      - python
      - -c
      - print('hello')
telemetry:
  interval_seconds: 0.001
  gpu_command: fake-gpu
""",
        encoding="utf-8",
    )

    sessions_created: list[SimulatorSession] = []

    def fake_launch(mode, settings_path, **kwargs):  # noqa: ANN001
        session = SimulatorSession(
            pid=1000 + len(sessions_created),
            process=DummyProcess(2000 + len(sessions_created), polls=[None, 0]),
            mode=mode,
            settings_path=settings_path,
        )
        sessions_created.append(session)
        return session

    monkeypatch.setattr("scripts.parallel_orchestrator.airsim_runner.launch", fake_launch)

    popen_calls: list[list[str]] = []

    def fake_popen(command, env=None, cwd=None):  # noqa: ANN001
        popen_calls.append(command)
        return DummyProcess(pid=3000, polls=[None, 0])

    monkeypatch.setattr("scripts.parallel_orchestrator.subprocess.Popen", fake_popen)

    monkeypatch.setattr(
        "scripts.parallel_orchestrator.subprocess.check_output",
        lambda *args, **kwargs: "GPU-1, RTX, 20 %, 100 MiB, 200 MiB",
    )

    terminated_sessions: list[int] = []

    def fake_terminate(session, force=True):  # noqa: ANN001
        terminated_sessions.append(session.pid)

    monkeypatch.setattr("scripts.parallel_orchestrator.airsim_runner.terminate", fake_terminate)

    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, **unused):  # noqa: ANN001
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self) -> None:
            self._target(*self._args, **self._kwargs)

        def join(self, timeout: float | None = None) -> None:  # noqa: ANN001
            return None

    monkeypatch.setattr("scripts.parallel_orchestrator.threading.Thread", ImmediateThread)

    exit_code = orchestrator_script.main(["--config", str(config_path), "--run-id", "test-run"])

    assert exit_code == 0
    assert len(sessions_created) == 1
    assert popen_calls[0][0] == "python"
    assert terminated_sessions == [sessions_created[0].pid]

    run_dirs = list((tmp_path / "artifacts").glob("test-run*"))
    assert run_dirs, "expected artifact directory to be created"
    telemetry_path = run_dirs[0] / "logs" / "telemetry.jsonl"
    summary_path = run_dirs[0] / "summary.json"

    telemetry_entries = [
        json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    assert telemetry_entries[0]["gpu"][0]["uuid"] == "GPU-1"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["jobs"]["job0"] == 0
