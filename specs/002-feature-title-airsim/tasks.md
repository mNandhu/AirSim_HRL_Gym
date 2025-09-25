# Tasks: AirSim HRL Training Framework

**Input**: Design documents from `/specs/002-feature-title-airsim/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/

## Execution Flow (summary)

1. Setup project scaffold & configuration
2. Author failing tests (unit + integration) per TDD
3. Implement core modules (config, env, perception, runner, agents)
4. Integrate perception + reward + agent hierarchy
5. Validate reproducibility & performance targets
6. Polish documentation, coverage, and profiling

## Phase 3.1: Setup

-   [x] T001 Create directory scaffold: `src/airsim_env/`, `src/hrl_agent/manager/`, `src/hrl_agent/workers/`, `src/perception/`, `src/utils/`, `src/config/`, `src/scripts/`
-   [x] T002 Initialize uv project & pyproject with pinned deps (python==3.12, stable-baselines3, torch, pydantic, pytest, pytest-cov)
-   [x] T003 [P] Add tooling configs: `.editorconfig`, `pyproject.toml` lint/format (ruff/black), coverage ≥90% gate
-   [x] T004 [P] Add baseline `settings.json` and hash utility in `src/config/settings_hash.py`
-   [x] T005 Create experiment schema models in `src/config/experiment.py` (ExperimentDefinition, SeedBundle)
-   [x] T006 Implement seed application helper `src/config/seeds.py`

## Phase 3.2: Tests First (TDD)

-   [x] T007 [P] Unit test experiment schema validation: `tests/unit/test_experiment_schema.py`
-   [x] T008 [P] Unit test seed application (reproducibility): `tests/unit/test_seeds.py`
-   [x] T009 [P] Unit test reward component placeholders: `tests/unit/test_reward_components.py`
-   [x] T010 [P] Unit test perception interfaces (segmentation + detector mocks): `tests/unit/test_perception_interfaces.py`
-   [x] T011 [P] Unit test simulator runner lifecycle (mock subprocess): `tests/unit/test_airsim_runner.py`
-   [x] T012 [P] Integration test seeded rollout skeleton (env reset/step loop deterministic): `tests/integration/test_seeded_rollout.py`
-   [x] T013 [P] Integration test restart after simulated crash: `tests/integration/test_restart_recovery.py`
-   [x] T014 [P] Integration test observation structure & reward components shape: `tests/integration/test_observation_reward_contract.py`

## Phase 3.3: Core Implementation

-   [x] T015 [P] Implement reward module scaffold `src/airsim_env/reward.py` (compute components listed in contract)
-   [x] T016 [P] Implement observation builder `src/airsim_env/observation.py`
-   [x] T017 [P] Implement perception segmentation adapter `src/perception/segmentation.py`
-   [x] T018 [P] Implement YOLO detector wrapper (torch.hub) `src/perception/detector.py`
-   [x] T019 Implement simulator runner with retries `src/utils/airsim_runner.py`
-   [x] T020 Implement environment core API `src/airsim_env/env.py` (reset, step, compute_reward delegation)
-   [x] T021 Implement artifact & logging utilities `src/utils/artifacts.py`
-   [x] T022 Implement experiment loader & hashing `src/config/loader.py`
-   [x] T023 Implement DQN manager wrapper `src/hrl_agent/manager/dqn_manager.py`
-   [x] T024 Implement SAC worker wrapper `src/hrl_agent/workers/sac_worker.py`
-   [x] T025 Implement command dispatch & coordination layer `src/hrl_agent/coordination.py`
-   [x] T026 Implement reward contract doc generator `src/utils/reward_doc.py`

## Phase 3.4: Integration

-   [x] T027 Integrate perception into observation builder (compose segmentation + detections)
-   [x] T028 Integrate reward module into environment step pipeline
-   [x] T029 Integrate manager→worker command flow
-   [x] T030 Add seeded rollout script `src/scripts/run_experiment.py`
-   [x] T031 Add pytest fixture `airsim_session` in `tests/conftest.py` (headless, retries, logs)
-   [x] T032 Validate deterministic reruns (compare config hash + cumulative reward) `tests/integration/test_determinism.py`
-   [x] T033 Add latency sampling instrumentation (perception + step) `src/utils/perf_metrics.py`
-   [x] T034 Implement multi-worker sequential scheduling placeholder in coordination module

## Phase 3.5: Polish

-   [x] T035 [P] Extend unit tests for edge cases (missing segmentation, detector timeout) `tests/unit/test_perception_edge_cases.py`
-   [x] T036 [P] Add collision and goal termination integration test `tests/integration/test_termination_conditions.py`
-   [x] T037 [P] Generate initial `docs/reward-contract.md` from reward module
-   [x] T038 [P] Add README reproducibility + architecture overview
-   [x] T039 [P] Add profiling script `src/scripts/profile_step.py`
-   [x] T040 [P] Coverage & quality gate enforcement script `scripts/ci_verify.py`
-   [x] T041 Final seeded smoke test & artifact snapshot `tests/integration/test_smoke_final.py`
-   [x] T042 Clean up TODO comments and ensure type hints across public APIs

## Dependencies

-   Setup (T001-T006) precedes all tests
-   Tests (T007-T014) precede implementation (T015+)
-   T020 depends on T015, T016, T017, T018
-   T025 depends on T023, T024
-   Integration tasks (T027-T034) depend on core implementation tasks
-   Polish tasks (T035+) depend on integration completion

## Parallel Execution Examples

```
# Example parallel batch (early unit tests):
run_task T007 & run_task T008 & run_task T009 & run_task T010 & run_task T011

# Example parallel batch (core perception + reward):
run_task T015 & run_task T016 & run_task T017 & run_task T018

# Example parallel batch (polish docs/tests):
run_task T035 & run_task T036 & run_task T037 & run_task T038
```

## Validation Checklist

-   [ ] All contract interfaces covered by unit tests
-   [ ] Environment step + reward deterministic under fixed seeds
-   [ ] Perception p50 latency <50ms (documented)
-   [ ] Non-learning coverage ≥90%
-   [ ] Reward contract doc matches implemented components
-   [ ] Two consecutive seeded runs produce identical config hash & cumulative reward

## Notes

-   YOLO model name kept generic (`yolov12n` placeholder) — pin exact variant when weights confirmed.
-   Scheduling policy upgrade (priority queue) deferred; placeholder ensures deterministic sequential order.

## Phase 4: Pending Implementations

> Goal: finish hardening the single-agent training workflow before layering in parallelized rollout support.

### Phase 4A — Solidify Training Framework

-   [x] T043 Stabilize AirSim adapter loop — tighten reset/step contracts, move from fixed sleeps to simulator time checks, and expose low-level telemetry needed by rewards and workers.
-   [x] T044 Integrate SB3 training hooks — replace placeholder Gym policies with AirSim-backed environments, wire replay buffers, and invoke `.learn()` inside the coordinator/orchestrator loop.
-   [x] T045 Expand evaluation artifacts — capture episode summaries, config hashes, and perception snapshots during `eval` runs to track regressions.
-   [x] T046 Command completion feedback — add heuristics or perception triggers that mark commands as complete so the reward bonus becomes meaningful.
-   [x] T047 Determinism audit — rerun seeded tests and profiling to confirm the hardened loop preserves existing reproducibility guarantees.

### Phase 4B — Parallelized Training Enablement (next phase)

-   **T048 Environment pooling design** — outline how multiple AirSim instances or replay buffers can be multiplexed without violating seed isolation.
-   **T049 Parallel coordinator interface** — extend `CommandCoordinator`/`HRLOrchestrator` to accept batched observations and distribute actions across workers.
-   **T050 Async perception pipeline** — introduce frame skipping or worker threads so perception keeps up with multiple concurrent simulators.
-   **T051 Resource orchestration tooling** — add scripts/configs for launching and monitoring parallel jobs (Multi-GPU SB3, multiple AirSim heads) with telemetry on utilization.
-   **T052 Regression & scaling tests** — craft pytest/integration scenarios that exercise the parallel path and compare throughput against the single-run baseline.
