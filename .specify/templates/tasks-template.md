# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)

```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: environment module scaffolding, dependency injection, config schema
   → Tests: pytest unit coverage for utilities + reward logic, integration harness for AirSim
   → Environment: `AirSimEnv` implementation, telemetry adapters, reward encapsulation
   → Agent: manager/worker hooks that consume environment interfaces only
   → Reproducibility & Docs: experiment configs, seed pipelines, reward documentation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → Environment/agent boundary maintained?
   → Reproducible experiment artifacts defined (start/end pose, seeds)?
   → Mandatory pytest suites (unit + integration) authored before implementation?
   → Reward/observation documentation tasks included?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- Environment module lives under `src/airsim_env/` (or repo-specific equivalent) with pytest suites under `tests/airsim_env/`.
- HRL agent logic resides under `src/hrl_agent/` and MUST consume environment interfaces only.
- Experiment configurations stored in `configs/experiments/` with seeds persisted alongside outputs.
- Adjust paths based on plan.md structure while preserving environment/agent separation.

## Phase 3.1: Setup

- [ ] T001 Create `src/airsim_env/` module and register it as standalone package
- [ ] T002 Establish dependency injection container for AirSim client (`src/infra/clients.py`)
- [ ] T003 [P] Define reproducible experiment schema in `configs/experiments/base.yaml`
- [ ] T004 [P] Configure linting, formatting, and pytest coverage thresholds (>=90% for non-learning code)

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] T005 [P] Unit test `AirSimEnv.reset`/`step` lifecycle in `tests/airsim_env/test_env_loop.py`
- [ ] T006 [P] Unit test reward calculus module in `tests/airsim_env/test_reward.py`
- [ ] T007 [P] Integration test seeded rollout using simulator stub in `tests/integration/test_seeded_rollout.py`
- [ ] T008 [P] Configuration validator test for experiment schema in `tests/integration/test_config_validation.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] T009 [P] Implement `AirSimEnv` interface in `src/airsim_env/env.py`
- [ ] T010 [P] Implement reward module encapsulation `src/airsim_env/reward.py`
- [ ] T011 [P] Add observation builder with normalization in `src/airsim_env/observation.py`
- [ ] T012 Create agent-facing command bus consuming environment interface in `src/hrl_agent/manager.py`
- [ ] T013 Implement experiment config loader applying seeds in `src/infra/config_loader.py`
- [ ] T014 Persist rollout metadata and telemetry to `artifacts/` bundle

## Phase 3.4: Integration

- [ ] T015 Connect AirSim client adapter with retry/backoff logic
- [ ] T016 Wire telemetry streaming to logging + metrics sink
- [ ] T017 Expose CLI/CLI command `scripts/run_experiment.py` with deterministic seed inputs
- [ ] T018 Verify environment compatibility against latest agent branch (no reverse dependencies)

## Phase 3.5: Polish

- [ ] T019 [P] Extend pytest suite for edge-case telemetry failures
- [ ] T020 Generate/refresh `docs/reward-contract.md`
- [ ] T021 [P] Publish reproducibility checklist (config hash + seeds) in release notes
- [ ] T022 Enforce coverage gate and archive pytest HTML report
- [ ] T023 Conduct manual smoke with seeded AirSim session and record findings

## Dependencies

- Tests (T005-T008) before implementation (T009-T014)
- T009 blocks T015 (AirSim client integration)
- T010/T011 block agent consumption (T012)
- T013 blocks reproducibility documentation (T021)
- Implementation complete before polish (T019-T023)

## Parallel Example

```
# Launch T005-T008 together:
Task: "Unit test AirSimEnv.reset/step lifecycle in tests/airsim_env/test_env_loop.py"
Task: "Unit test reward calculus module in tests/airsim_env/test_reward.py"
Task: "Integration test seeded rollout in tests/integration/test_seeded_rollout.py"
Task: "Configuration validator test for experiment schema in tests/integration/test_config_validation.py"
```

## Notes

- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules

_Applied during main() execution_

1. **From Environment Interfaces**:

   - Each interface change → corresponding pytest unit task [P]
   - Reward/observation updates → documentation + test tasks

2. **From Experiment Configurations**:

   - Each scenario → config file task + seeded smoke test
   - Seeds/poses → artifact persistence task

3. **From Agent Requirements**:

   - Each agent-worker interaction → adapter task consuming environment API only
   - Concurrent workers → telemetry coordination tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist

_GATE: Checked by main() before returning_

- [ ] Environment boundary tasks exist (no agent imports in environment tasks)
- [ ] Reproducibility tasks cover scene, poses, seeds, and artifact logging
- [ ] Pytest tasks precede implementation tasks and enforce coverage thresholds
- [ ] Reward/observation documentation tasks included
- [ ] Parallel tasks operate on disjoint files
- [ ] Each task specifies exact file path or configuration artifact
