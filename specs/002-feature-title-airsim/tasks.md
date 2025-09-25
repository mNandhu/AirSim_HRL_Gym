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

- [ ] T001 Create directory scaffold: `src/airsim_env/`, `src/hrl_agent/manager/`, `src/hrl_agent/workers/`, `src/perception/`, `src/utils/`, `src/config/`, `src/scripts/`
- [ ] T002 Initialize uv project & pyproject with pinned deps (python==3.12, stable-baselines3, torch, pydantic, pytest, pytest-cov)
- [ ] T003 [P] Add tooling configs: `.editorconfig`, `pyproject.toml` lint/format (ruff/black), coverage ≥90% gate
- [ ] T004 [P] Add baseline `settings.json` and hash utility in `src/config/settings_hash.py`
- [ ] T005 Create experiment schema models in `src/config/experiment.py` (ExperimentDefinition, SeedBundle)
- [ ] T006 Implement seed application helper `src/config/seeds.py`

## Phase 3.2: Tests First (TDD)

- [ ] T007 [P] Unit test experiment schema validation: `tests/unit/test_experiment_schema.py`
- [ ] T008 [P] Unit test seed application (reproducibility): `tests/unit/test_seeds.py`
- [ ] T009 [P] Unit test reward component placeholders: `tests/unit/test_reward_components.py`
- [ ] T010 [P] Unit test perception interfaces (segmentation + detector mocks): `tests/unit/test_perception_interfaces.py`
- [ ] T011 [P] Unit test simulator runner lifecycle (mock subprocess): `tests/unit/test_airsim_runner.py`
- [ ] T012 [P] Integration test seeded rollout skeleton (env reset/step loop deterministic): `tests/integration/test_seeded_rollout.py`
- [ ] T013 [P] Integration test restart after simulated crash: `tests/integration/test_restart_recovery.py`
- [ ] T014 [P] Integration test observation structure & reward components shape: `tests/integration/test_observation_reward_contract.py`

## Phase 3.3: Core Implementation

- [ ] T015 [P] Implement reward module scaffold `src/airsim_env/reward.py` (compute components listed in contract)
- [ ] T016 [P] Implement observation builder `src/airsim_env/observation.py`
- [ ] T017 [P] Implement perception segmentation adapter `src/perception/segmentation.py`
- [ ] T018 [P] Implement YOLO detector wrapper (torch.hub) `src/perception/detector.py`
- [ ] T019 Implement simulator runner with retries `src/utils/airsim_runner.py`
- [ ] T020 Implement environment core API `src/airsim_env/env.py` (reset, step, compute_reward delegation)
- [ ] T021 Implement artifact & logging utilities `src/utils/artifacts.py`
- [ ] T022 Implement experiment loader & hashing `src/config/loader.py`
- [ ] T023 Implement DQN manager wrapper `src/hrl_agent/manager/dqn_manager.py`
- [ ] T024 Implement SAC worker wrapper `src/hrl_agent/workers/sac_worker.py`
- [ ] T025 Implement command dispatch & coordination layer `src/hrl_agent/coordination.py`
- [ ] T026 Implement reward contract doc generator `src/utils/reward_doc.py`

## Phase 3.4: Integration

- [ ] T027 Integrate perception into observation builder (compose segmentation + detections)
- [ ] T028 Integrate reward module into environment step pipeline
- [ ] T029 Integrate manager→worker command flow
- [ ] T030 Add seeded rollout script `src/scripts/run_experiment.py`
- [ ] T031 Add pytest fixture `airsim_session` in `tests/conftest.py` (headless, retries, logs)
- [ ] T032 Validate deterministic reruns (compare config hash + cumulative reward) `tests/integration/test_determinism.py`
- [ ] T033 Add latency sampling instrumentation (perception + step) `src/utils/perf_metrics.py`
- [ ] T034 Implement multi-worker sequential scheduling placeholder in coordination module

## Phase 3.5: Polish

- [ ] T035 [P] Extend unit tests for edge cases (missing segmentation, detector timeout) `tests/unit/test_perception_edge_cases.py`
- [ ] T036 [P] Add collision and goal termination integration test `tests/integration/test_termination_conditions.py`
- [ ] T037 [P] Generate initial `docs/reward-contract.md` from reward module
- [ ] T038 [P] Add README reproducibility + architecture overview
- [ ] T039 [P] Add profiling script `src/scripts/profile_step.py`
- [ ] T040 [P] Coverage & quality gate enforcement script `scripts/ci_verify.py`
- [ ] T041 Final seeded smoke test & artifact snapshot `tests/integration/test_smoke_final.py`
- [ ] T042 Clean up TODO comments and ensure type hints across public APIs

## Dependencies

- Setup (T001-T006) precedes all tests
- Tests (T007-T014) precede implementation (T015+)
- T020 depends on T015, T016, T017, T018
- T025 depends on T023, T024
- Integration tasks (T027-T034) depend on core implementation tasks
- Polish tasks (T035+) depend on integration completion

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

- [ ] All contract interfaces covered by unit tests
- [ ] Environment step + reward deterministic under fixed seeds
- [ ] Perception p50 latency <50ms (documented)
- [ ] Non-learning coverage ≥90%
- [ ] Reward contract doc matches implemented components
- [ ] Two consecutive seeded runs produce identical config hash & cumulative reward

## Notes

- YOLO model name kept generic (`yolov12n` placeholder) — pin exact variant when weights confirmed.
- Scheduling policy upgrade (priority queue) deferred; placeholder ensures deterministic sequential order.
