"""Launch and monitor parallel AirSim training jobs with telemetry collection."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from utils import airsim_runner
from utils.artifacts import ArtifactManager

IS_WINDOWS = os.name == "nt"

__all__ = ["main"]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Parallel orchestration config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def _generate_settings_file(
    base_settings_path: Path, output_dir: Path, instance_index: int, base_port: int
) -> Path:
    if not base_settings_path.exists():
        raise FileNotFoundError(f"Base settings file not found: {base_settings_path}")

    with base_settings_path.open("r", encoding="utf-8") as handle:
        settings_data = json.load(handle)

    settings_data["ApiServerPort"] = base_port + instance_index
    output_path = output_dir / f"settings_{instance_index}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(settings_data, handle, indent=2)

    return output_path


def launch_airsim_instances(
    definitions: Iterable[dict[str, Any]],
    *,
    run_dir: Path,
    base_port: int,
) -> dict[str, airsim_runner.SimulatorSession]:
    sessions: dict[str, airsim_runner.SimulatorSession] = {}
    for i, definition in enumerate(definitions):
        name = str(definition.get("name", f"sim_{i}"))
        mode = str(definition.get("mode", "headless"))
        base_settings = definition.get("settings")
        if base_settings is None:
            raise ValueError(f"A settings path is required for AirSim instance {name!r}")

        assigned_port = base_port + i
        settings_path = _generate_settings_file(Path(base_settings), run_dir, i, base_port)

        platform_key = "command_windows" if IS_WINDOWS else "command_linux"
        command_template = definition.get(platform_key) or definition.get("command")

        # Replace placeholder with the actual settings path
        if command_template:
            command = [
                arg.replace("{settings_path}", str(settings_path)) for arg in command_template
            ]
            # Normalize "-settings", "<path>" into a single token "-settings=<path>"
            try:
                idx = command.index("-settings")
                if idx + 1 < len(command):
                    path_arg = command[idx + 1]
                    command[idx : idx + 2] = [f"-settings={path_arg}"]
            except ValueError:
                # No "-settings" token found; leave as-is
                pass
        else:
            command = None

        if command is not None and not isinstance(command, (list, tuple)):
            raise ValueError(f"command for AirSim instance {name!r} must be a sequence")

        # Mapping print for clarity
        print(f"• Mapping: AirSim {name} -> port {assigned_port}, settings={settings_path}")

        session = airsim_runner.launch(
            mode,
            str(settings_path),
            command=command,
            max_attempts=int(definition.get("max_attempts", 3)),
            backoff_seconds=float(definition.get("backoff_seconds", 2.0)),
            timeout=float(definition.get("timeout", 30.0)),
        )
        sessions[name] = session
        print(f"✓ AirSim instance {name} launched (pid={session.pid})")
    return sessions


def launch_training_jobs(
    definitions: Iterable[dict[str, Any]],
    *,
    base_port: int,
    run_dir: Path | None = None,
) -> dict[str, subprocess.Popen[Any]]:
    jobs: dict[str, subprocess.Popen[Any]] = {}
    for i, definition in enumerate(definitions):
        name = str(definition.get("name", f"job_{i}"))
        command = definition.get("command")
        if not command or not isinstance(command, (list, tuple)):
            raise ValueError(f"command for training job {name!r} must be a non-empty sequence")

        env_overrides = {str(k): str(v) for k, v in (definition.get("env", {}) or {}).items()}
        env = os.environ.copy()
        env.update(env_overrides)

        assigned_port = base_port + i
        env["AIRSIM_PORT"] = str(assigned_port)
        env["AIRSIM_HOST"] = "127.0.0.1"
        # Provide the actual settings path this trainer should reference (if available)
        if run_dir is not None:
            env["AIRSIM_SETTINGS_PATH"] = str(run_dir / f"settings_{i}.json")

        # Mapping print for clarity
        gpu_info = env_overrides.get("CUDA_VISIBLE_DEVICES")
        gpu_suffix = f", gpu={gpu_info}" if gpu_info is not None else ""
        print(f"• Mapping: Trainer {name} -> {env['AIRSIM_HOST']}:{env['AIRSIM_PORT']}{gpu_suffix}")

        cwd = definition.get("cwd")
        process = subprocess.Popen(command, env=env, cwd=cwd)  # noqa: S603
        jobs[name] = process
        print(f"→ Launched training job {name} (pid={process.pid})")
    return jobs


def query_gpu_utilization(command: str = "nvidia-smi") -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                command,
                "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    stats: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 5:
            continue
        gpu_uuid, name, util, mem_used, mem_total = parts
        stats.append(
            {
                "uuid": gpu_uuid,
                "name": name,
                "utilization": util,
                "memory_used": mem_used,
                "memory_total": mem_total,
            }
        )
    return stats


def monitor_telemetry(
    jobs: dict[str, subprocess.Popen[Any]],
    artifact_manager: ArtifactManager,
    *,
    interval: float,
    gpu_command: str | None,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        timestamp = time.time()
        job_status = {
            name: {
                "pid": process.pid,
                "returncode": process.poll(),
            }
            for name, process in jobs.items()
        }
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "jobs": job_status,
        }
        if gpu_command:
            payload["gpu"] = query_gpu_utilization(gpu_command)
        artifact_manager.append_jsonl("telemetry.jsonl", payload)
        if all(process.poll() is not None for process in jobs.values()):
            break
        stop_event.wait(interval)


def terminate_airsim_sessions(sessions: dict[str, airsim_runner.SimulatorSession]) -> None:
    for name, session in sessions.items():
        try:
            airsim_runner.terminate(session, force=True)
            print(f"✓ Terminated AirSim instance {name} (pid={session.pid})")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"! Failed to terminate AirSim instance {name}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch parallel AirSim training jobs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/parallel/sample_parallel.yaml"),
        help="YAML file describing AirSim instances and training jobs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="parallel_orchestration",
        help="Identifier appended to artifact directory",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=41451,
        help="Base port for AirSim API servers",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    # Use cwd/artifacts if running in test context
    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        artifact_root = os.path.join(os.getcwd(), "artifacts")
    else:
        artifact_root = "artifacts"
    artifact_manager = ArtifactManager(artifact_root)
    run_paths = artifact_manager.start_run(args.run_id, timestamp_first=False)
    artifact_manager.write_json(
        "config_snapshot.json", {"config_path": str(args.config), "config": config}
    )

    airsim_defs = config.get("airsim_instances", []) or []
    job_defs = config.get("training_jobs", []) or []
    telemetry_cfg = config.get("telemetry", {}) or {}

    sessions = launch_airsim_instances(
        airsim_defs, run_dir=run_paths.run_dir, base_port=args.base_port
    )
    jobs = launch_training_jobs(job_defs, base_port=args.base_port, run_dir=run_paths.run_dir)

    stop_event = threading.Event()
    telemetry_thread = threading.Thread(
        target=monitor_telemetry,
        name="telemetry-thread",
        args=(jobs, artifact_manager),
        kwargs={
            "interval": float(telemetry_cfg.get("interval_seconds", 5.0)),
            "gpu_command": telemetry_cfg.get("gpu_command"),
            "stop_event": stop_event,
        },
        daemon=True,
    )
    telemetry_thread.start()

    try:
        exit_code = 0
        for name, process in jobs.items():
            returncode = process.wait()
            if returncode != 0 and exit_code == 0:
                exit_code = returncode
            artifact_manager.append_jsonl(
                "job_events.jsonl",
                {
                    "timestamp": time.time(),
                    "event": "job_completed",
                    "job": name,
                    "returncode": returncode,
                },
            )
    except KeyboardInterrupt:
        print("! Received interrupt, terminating jobs...")
        exit_code = 130
        for process in jobs.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
    finally:
        stop_event.set()
        telemetry_thread.join(timeout=10.0)
        # Ensure artifact directory is flushed before returning
        # Use time module for sleep, ensure not shadowed
        import time as _time

        for _ in range(10):
            if run_paths.run_dir.exists():
                break
            _time.sleep(0.1)
        terminate_airsim_sessions(sessions)

    artifact_manager.write_json(
        "summary.json",
        {
            "run_dir": str(run_paths.run_dir),
            "jobs": {name: proc.poll() for name, proc in jobs.items()},
            "airsim_instances": list(sessions.keys()),
        },
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
