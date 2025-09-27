"""Utilities for launching and supervising AirSim simulator processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from dotenv import load_dotenv

__all__ = [
    "SimulatorSession",
    "SimulatorLaunchError",
    "launch",
    "ensure_running",
    "terminate",
    "restart",
    "airsim_session",
]

load_dotenv()

LOG_PATH = Path("simulator_failures.log")
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")


def _default_executable() -> str:
    env_override = os.environ.get("AIRSIM_EXECUTABLE")
    if env_override:
        return env_override
    if IS_WINDOWS:
        return "AirSimNH.exe"
    return "./AirSimNH.sh"


DEFAULT_EXECUTABLE = _default_executable()


class SimulatorLaunchError(RuntimeError):
    """Raised when the AirSim process cannot be started after retries."""


@dataclass
class SimulatorSession:
    pid: int
    process: Any
    mode: str
    settings_path: str
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    retries: int = 0


def launch(
    mode: str,
    settings_path: str,
    *,
    command: Sequence[str] | None = None,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
) -> SimulatorSession:
    """Launch AirSim with retry semantics."""

    settings_path = str(settings_path)
    if not Path(settings_path).exists():
        raise FileNotFoundError(f"Settings file does not exist: {settings_path}")

    exe_command = list(command or _build_command(mode, settings_path))
    attempts = 0
    last_error: Exception | None = None

    while attempts < max_attempts:
        attempts += 1
        try:
            process = subprocess.Popen(exe_command)  # noqa: S603 - subprocess needed for simulator
            _wait_for_start(process, timeout)
            return SimulatorSession(
                pid=process.pid,
                process=process,
                mode=mode,
                settings_path=settings_path,
                retries=attempts - 1,
            )
        except Exception as exc:  # pragma: no cover - failure path tested via monkeypatch
            last_error = exc
            _log_failure("launch_failed", mode, attempts, exc)
            if attempts >= max_attempts:
                break
            time.sleep(backoff_seconds * attempts)

    message = f"Failed to launch AirSim after {max_attempts} attempts"
    if last_error is not None:
        raise SimulatorLaunchError(message) from last_error
    raise SimulatorLaunchError(message)


def ensure_running(session: SimulatorSession) -> None:
    """Ensure the simulator process is alive."""

    if session.process.poll() is not None:
        raise RuntimeError("AirSim process is not running")


def terminate(session: SimulatorSession, *, force: bool = False) -> None:
    """Terminate the AirSim process gracefully or forcefully."""

    if session.process.poll() is not None:
        return
    try:
        session.process.terminate()
        session.process.wait(timeout=10)
    except Exception:  # pragma: no cover - fallback path
        if force:
            session.process.kill()
        else:
            session.process.kill()


def restart(
    session: SimulatorSession,
    *,
    mode: str | None = None,
    max_attempts: int = 3,
) -> SimulatorSession:
    """Restart the simulator by terminating the current process and re-launching."""

    terminate(session, force=True)
    return _launch_with_compat(
        mode or session.mode, session.settings_path, max_attempts=max_attempts
    )


@contextmanager
def airsim_session(
    *,
    mode: str = "headless",
    settings_path: str,
    max_attempts: int = 3,
) -> Iterator[SimulatorSession]:
    session = _launch_with_compat(mode, settings_path, max_attempts=max_attempts)
    try:
        yield session
    finally:
        terminate(session)


def _build_command(mode: str, settings_path: str) -> list[str]:
    executable = DEFAULT_EXECUTABLE
    cmd = _resolve_base_command(executable)
    cmd.extend(["-settings", settings_path])
    if mode == "headless":
        cmd.extend(_headless_flags())
    return cmd


def _resolve_base_command(executable: str) -> list[str]:
    if IS_WINDOWS:
        return [executable]
    if executable.endswith(".sh"):
        return ["bash", executable]
    return [executable]


def _headless_flags() -> list[str]:
    if IS_WINDOWS:
        return ["-RenderOffScreen"]
    # On Linux, provide additional hints for offscreen rendering
    return ["-RenderOffScreen", "-windowed", "-nosound"]


def _wait_for_start(process: Any, timeout: float) -> None:
    # A simple time-based wait is not sufficient, as the simulator may still be
    # initializing its RPC server even after the process is running.
    # A better approach would be to poll the API server, but that requires
    # knowing the port, which is configured in the settings file.
    # For now, we'll stick with a generous sleep.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SimulatorLaunchError("Simulator exited during startup")
        time.sleep(1)
    # If we exit the loop normally, assume process running


def _log_failure(event: str, mode: str, attempt: int, exc: Exception) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "event": event,
        "mode": mode,
        "attempt": attempt,
        "error": repr(exc),
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _launch_with_compat(mode: str, settings_path: str, *, max_attempts: int) -> SimulatorSession:
    try:
        return launch(mode, settings_path, max_attempts=max_attempts)
    except TypeError as exc:  # pragma: no cover - compatibility for monkeypatched functions
        if "max_attempts" in str(exc):
            return launch(mode, settings_path)
        raise
