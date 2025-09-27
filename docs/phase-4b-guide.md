# Phase 4B Parallel Training Guide

## Overview

Phase 4B extends the hierarchical RL stack so that multiple AirSim instances can be orchestrated in parallel while maintaining determinism, telemetry, and reproducible artifact logging. The new tooling lets you:

-   Pre-warm a pool of seeded simulator environments that can be leased across rollouts.
-   Coordinate batched manager/worker decisions across those environments.
-   Run perception asynchronously so vision workloads keep up with several simulators.
-   Launch and supervise parallel training jobs with unified telemetry on Windows _and_ Linux.

Use this guide as a quick reference for the code added in `src/airsim_env/pool.py`, `src/hrl_agent/coordination.py`, `src/hrl_agent/orchestrator.py`, `src/perception/async_pipeline.py`, `src/scripts/parallel_orchestrator.py`, and the accompanying tests.

## Environment Pooling (`src/airsim_env/pool.py`)

`EnvironmentPool` builds and manages a deterministic pool of `AirSimEnv` (or compatible) instances.

-   Call `EnvironmentPool.build(size, base_experiment, factory, seed_stride)` to construct the pool.
    -   `factory` should accept `(ExperimentDefinition, index)` and return `(env, session)`.
    -   Each entry derives a disjoint `SeedBundle` via `derive_seed_bundle`, offset by `index` and `seed_stride` to ensure reproducibility.
-   Acquire environments with `with pool.acquire(timeout=...) as lease:` to get a `PooledEnvironment`.
    -   `lease.entry.env` exposes the simulator, `lease.entry.seed_bundle` reveals the applied seeds, and `lease.entry.session` may hold the underlying AirSim process.
-   If a simulator misbehaves, call `pool.mark_unhealthy(index)` to tear it down and rebuild it with the same deterministic seed slice.
-   The pool keeps track of in-use entries and blocks until a simulator becomes available, so long-running rollouts do not starve others.

This design guarantees that parallel rollouts never reuse the same seed bundle, preserving determinism guarantees documented in `specs/002-feature-title-airsim/contracts/`.

## Parallel Command Coordination (`src/hrl_agent/coordination.py`, `src/hrl_agent/orchestrator.py`)

`CommandCoordinator` now supports per-environment state via `reset_env`, `act_for_env`, `command_completed_for_env`, and `observe_transition_for_env`. The orchestrator can therefore drive several environments simultaneously.

`HRLOrchestrator.run_parallel(envs: Mapping[str, AirSimEnv], ...)` handles:

-   Resetting each environment and coordinating manager/worker pairs per environment ID.
-   Logging telemetry and reward components through the existing metrics tracker interface.
-   Feeding transitions back into experience buffers when `deterministic=False`.
-   Gracefully removing finished environments while others continue.

Keep using `run_episode` for single-environment rollouts; both paths share the same coordinator wiring.

## Asynchronous Perception (`src/perception/async_pipeline.py`)

`AsyncPerceptionPipeline` wraps `SegmentationAdapter` and `YoloDetector` with a background worker thread.

-   `build_observation_inputs` captures segmentation masks immediately and queues RGB frames for deferred detection.
-   A bounded queue (default size 2) introduces frame skipping when overloaded instead of blocking control loops.
-   The worker gracefully stops on `close()`; callers should invoke this when tearing down a simulator.

Swap in `AsyncPerceptionPipeline` wherever the synchronous `PerceptionPipeline` was used to keep inference latency low across multiple simulators.

## Parallel Orchestration CLI (`src/scripts/parallel_orchestrator.py`)

`parallel_orchestrator.py` is the entry point for launching fleets of simulators and training jobs.

### Configuration layout

Point the script at a YAML file (see `configs/parallel/sample_parallel.yaml`) that defines three sections:

1. `telemetry`: sampling interval and an optional GPU command (defaults to `nvidia-smi`).
2. `airsim_instances`: metadata per simulator—name, mode (`headless` vs GUI), `settings` file, optional platform-specific launch commands, retry/backoff/timeout knobs.
3. `training_jobs`: shell command arrays for each learner plus optional `env` overrides and working directories.

### Running the orchestrator

```pwsh
uv run python src/scripts/parallel_orchestrator.py --config configs/parallel/sample_parallel.yaml --run-id phase4b-demo
```

What happens under the hood:

-   AirSim instances launch via `utils.airsim_runner.launch`, honoring Windows and Linux offscreen flags or any overrides supplied in the config.
-   Each training job starts with `subprocess.Popen` and is tracked until completion.
-   Telemetry (job status, optional GPU utilization) is appended to `logs/telemetry.jsonl` for the run.
-   Summary metadata is written to `summary.json` once all jobs finish.

Artifacts land under `artifacts/<run-id>_<timestamp>/`. The timestamp is still included for chronological sorting, but prefixing with `run-id` keeps automation-friendly names for CI and tests.

Stop the script with `Ctrl+C` to propagate SIGINT to running jobs while simulators are force-terminated.

## Resource & Platform Enhancements

-   `utils.airsim_runner` now:
    -   Detects Windows vs Linux defaults, applying `-RenderOffScreen` plus Linux-specific hints (`-windowed`, `-nosound`).
    -   Accepts an `AIRSIM_EXECUTABLE` override.
    -   Logs structured failure entries to `simulator_failures.log` when retries are exhausted.
-   The orchestrator script ensures artifact directories are flushed and simulator sessions are terminated even if telemetry threads are still draining.

## Testing & Regression Coverage

Phase 4B introduced targeted tests to prevent regressions:

-   `tests/unit/test_environment_pool.py` exercises pool sizing, leasing, and unhealthy recovery.
-   `tests/unit/test_coordinator_parallel.py` validates per-environment command coordination.
-   `tests/unit/test_parallel_orchestrator.py` stubs out subprocesses to verify telemetry, artifact creation, and cleanup.
-   `tests/integration/test_parallel_scaling.py` benchmarks throughput vs the baseline configuration.

Run the full suite with:

```pwsh
uv run pytest
```

or target an individual test module for quicker feedback, e.g.:

```pwsh
uv run pytest tests/unit/test_parallel_orchestrator.py
```

## Operational Checklist

1. Prepare your AirSim `settings.json` and confirm the simulator binary is reachable (or set `AIRSIM_EXECUTABLE`).
2. Customize `configs/parallel/sample_parallel.yaml`:
    - Duplicate the file per environment group, adjusting `command_windows`/`command_linux` execs and CUDA bindings.
    - For headless Linux runs, ensure `xvfb-run` or EGL support is available if you override the defaults.
3. Launch the orchestrator with a meaningful `--run-id`. The final artifact path will be `artifacts/<run-id>_<timestamp>/`.
4. Inspect telemetry in `logs/telemetry.jsonl` and job completions in `logs/job_events.jsonl`.
5. Review `summary.json` for a quick snapshot of return codes and launched simulator names.

## When to Fall Back

-   Single-simulator experiments can continue using `train_and_eval.py`; no configuration changes are required—the new pooling and orchestration utilities are additive.
-   If you need strict frame-by-frame detections (no dropping), revert to the synchronous perception pipeline by swapping out `AsyncPerceptionPipeline`.

## Troubleshooting Tips

-   **Simulators fail to launch**: confirm `settings` paths in the YAML exist on disk and increase `timeout` or `max_attempts` if the VM boots slowly. Check `simulator_failures.log` for timestamped failure details.
-   **No telemetry output**: ensure `telemetry.gpu_command` resolves on the host. The script falls back silently if the command is missing—set it to `null` to skip GPU sampling.
-   **Artifacts missing**: verify write permissions to the `artifacts/` folder. During tests the orchestrator automatically targets the working directory’s `artifacts` subfolder.

With these components in place, you can scale hierarchical RL training across multiple AirSim instances while keeping reproducibility, logging, and usability on par with single-instance workflows.
