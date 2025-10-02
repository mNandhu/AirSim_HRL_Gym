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

-   [x] **T048 Environment pooling design** — outline how multiple AirSim instances or replay buffers can be multiplexed without violating seed isolation.
-   [x] **T049 Parallel coordinator interface** — extend `CommandCoordinator`/`HRLOrchestrator` to accept batched observations and distribute actions across workers.
-   [x] **T050 Async perception pipeline** — introduce frame skipping or worker threads so perception keeps up with multiple concurrent simulators.
-   [x] **T051 Resource orchestration tooling** — add scripts/configs for launching and monitoring parallel jobs (Multi-GPU SB3, multiple AirSim heads) with telemetry on utilization.
-   [x] **T052 Cross-platform headless support** — implement Linux-compatible AirSim headless launcher alongside existing Windows support, ensuring consistent simulator behavior across platforms. Ensure that all functionality is supported in both linux and windows.
-   [x] **T053 Platform-agnostic orchestration** — extend resource orchestration tooling to detect and manage AirSim instances on both Windows and Linux environments with unified configuration.
-   [x] **T054 Regression & scaling tests** — craft pytest/integration scenarios that exercise the parallel path and compare throughput against the single-run baseline.
-   [x] **T055 Port-based simulator routing** — implement port assignment logic in parallel orchestrator to enable multiple AirSim instances on single machine with unique ApiServerPort configurations.
-   [x] **T056 Trainer-to-simulator connection mapping** — update training jobs to accept AIRSIM_HOST/AIRSIM_PORT environment variables and modify CarClient initialization to use these parameters for targeted simulator connections.
-   [x] **T057 Dynamic settings.json generation** — add orchestrator capability to generate unique settings.json files per AirSim instance with different ports and pass these to simulator launch commands.
-   [x] **T058 Multi-instance coordination validation** — create integration tests that verify multiple trainers can connect to their assigned simulator instances simultaneously without port conflicts.

---

## Phase 5: Single-Agent SAC Pivot (Post-80-Episode Analysis)

> **Context**: After 80 episodes of hierarchical RL training, empirical evidence shows workers cannot learn with insufficient data (1,500 steps each vs 10,000+ required), manager learns from random worker outputs (diverging policy), and zero task progress (0/80 episodes reached any waypoint). Mathematical proof: P(hierarchical success | untrained workers) = 0. **Decision**: Pivot to proven single-agent SAC approach to establish baseline, then optionally return to hierarchical with proper pre-training.

### Phase 5A — Single-Agent SAC Implementation (Fast Baseline)

> **Note**: Single-agent keeps existing PID control for low-level dynamics. Agent outputs [target_speed, target_steering] just like hierarchical workers did—we're only removing the manager→command layer. This is simpler to learn (2D action space) and proven in autonomous driving research.

-   [x] **T059 Single-agent environment wrapper** — Create `src/airsim_env/single_agent_env.py` that exposes flat action space WITHOUT command hierarchy but WITH PID control.

    -   Input: Observation (same as current: images, telemetry, waypoints)
    -   Output: `Box(2)` actions [target_speed, target_steering] ∈ [0-10, -1 to 1] (same as workers)
    -   PID: Keep existing speed PID (converts target_speed → throttle/brake)
    -   Reward: Use existing reward components (progress, lane deviation, collision, etc.)
    -   Success criteria: Drop-in Gym API compatibility for SB3

-   [x] **T060 Single-agent training script** — Create `src/scripts/train_single_agent.py` based on existing `train_and_eval.py` but instantiate single SAC policy instead of hierarchical coordinator.

    -   Remove: DQNManager, SAC workers, CommandCoordinator
    -   Add: Single `SAC(policy="MultiInputPolicy", env=SingleAgentEnv, ...)`
    -   Hyperparameters: `learning_rate=3e-4`, `buffer_size=100000`, `learning_starts=10000`, `batch_size=256`, `tau=0.005`, `gamma=0.99`
    -   Logging: Reuse artifact manager, metrics tracker (same as hierarchical)

-   [x] **T061 Action space validation** — Unit test that action bounds, scaling, and control mapping work correctly.

    -   Test: Actions ∈ [-1, 1] map to valid throttle/brake/steering
    -   Test: Edge cases (simultaneous throttle+brake, extreme steering)
    -   Test: Deterministic action→control mapping

-   [x] **T062 Single-agent reward validation** — Integration test that reward components fire correctly without command context.

    -   Test: Progress velocity reward works with direct actions
    -   Test: Lane deviation penalty applies
    -   Test: Collision penalty triggers
    -   Test: Waypoint progress tracked correctly

-   [ ] **T063 Baseline training run** — Execute 100-200 episode training to establish convergence baseline.

    -   Target: 50%+ episodes reach ≥1 waypoint by episode 200
    -   Metrics: Episode length, cumulative reward, waypoints reached, collision rate
    -   Success criteria: Upward trend in waypoints, decreasing collision rate
    -   Artifacts: Save checkpoints every 25 episodes, generate learning curves

-   [x] **T064 Single-agent evaluation script** — Extend `train_and_eval.py eval` mode or create separate evaluator for trained single-agent checkpoints.
    -   Load: Trained SAC policy from checkpoint
    -   Run: Deterministic evaluation episodes (no exploration noise)
    -   Report: Success rate, waypoint completion, trajectory visualization

### Phase 5B — Documentation & Analysis

-   [ ] **T065 Update architecture docs** — Revise `docs/model-structure.md` and README to reflect single-agent approach as primary training mode.

    -   Mark hierarchical mode as "experimental/pre-training required"
    -   Document single-agent as "validated baseline"
    -   Add performance comparison table (when available)

-   [ ] **T066 Training comparison analysis** — Generate comparative report between hierarchical (80 ep) and single-agent (200 ep) results.

    -   Metrics: Waypoints reached, episode length, collision rate, training time
    -   Graphs: Learning curves, trajectory quality, convergence speed
    -   Document: `docs/training-comparison-hierarchical-vs-single-agent.md`

-   [ ] **T067 Hyperparameter tuning guide** — Document SAC hyperparameters and tuning recommendations for this task.
    -   Learning rate: 1e-4 to 5e-4 range
    -   Buffer size: 50K-200K (vs episode length × episodes)
    -   Tau: 0.005-0.01 (soft update rate)
    -   Gamma: 0.95-0.99 (discount factor for waypoint sequences)

### Phase 5C — Optional: Return to Hierarchical (If Time Permits)

> **Only proceed if single-agent succeeds AND hierarchical interpretability is required**

-   [ ] **T068 Worker pre-training infrastructure** — Create `src/scripts/pretrain_workers.py` to train individual SAC workers in isolation.

    -   Per-worker environments: Simplified tasks (straight lane for FOLLOW_LANE, turn circle for TURN_LEFT/RIGHT, deceleration for STOP)
    -   Training: 50K steps per worker with shaped rewards
    -   Save: Worker checkpoints for hierarchical initialization

-   [ ] **T069 Imitation learning bootstrap** — Use trained single-agent SAC to generate demonstrations for worker pre-training.

    -   Extract: Trajectories from trained single-agent policy
    -   Label: Commands based on trajectory context (lane following, turning, stopping)
    -   Train: Workers via behavior cloning on labeled demonstrations
    -   Validate: Workers achieve >70% task success on isolated tasks

-   [ ] **T070 Hierarchical training with pre-trained workers** — Resume hierarchical training with competent workers frozen or fine-tuning.

    -   Load: Pre-trained worker checkpoints
    -   Freeze: Workers during initial manager training (50-100 episodes)
    -   Train: DQN manager to sequence competent workers
    -   Fine-tune: Optionally unfreeze workers for joint optimization

-   [ ] **T071 Hierarchical validation** — Compare hierarchical performance with single-agent baseline.
    -   Metrics: Match or exceed single-agent waypoint completion
    -   Interpretability: Analyze command sequences for debugging/understanding
    -   Generalization: Test on unseen waypoint configurations
    -   Decision: Keep hierarchical if interpretability benefit outweighs complexity cost

### Phase 5D — Cleanup & Deprecation

-   [ ] **T072 Mark hierarchical training as deprecated** — Update CLI help text and docs to guide users to single-agent mode.

    -   Add warning: "Hierarchical mode requires worker pre-training. See docs/hierarchical-training-guide.md"
    -   Default: Single-agent mode in training scripts
    -   Flag: `--hierarchical` to opt-in to hierarchical (with pre-training check)

-   [ ] **T073 Archive 80-episode analysis** — Move hierarchical training analysis to `docs/archive/` for historical reference.

    -   Files: `training-analysis-20ep-not-converging.md`, `training-analysis-80ep-final-verdict.md`
    -   Add: Forward pointer to single-agent approach in archive files
    -   Preserve: Lessons learned about hierarchical RL bootstrapping

-   [ ] **T074 Update reward contract** — Ensure reward documentation reflects single-agent usage (no command-specific shaping).
    -   Clarify: Shaping rewards apply to all actions (not command-conditioned)
    -   Update: Examples show direct action → reward mapping
    -   Test: Reward calculations match documented formulas

## Dependencies (Phase 5)

-   **T059-T062** can run in parallel (single-agent implementation)
-   **T063** depends on T059-T062 (need working single-agent before training)
-   **T064** depends on T063 (need trained checkpoint to evaluate)
-   **T065-T067** can run in parallel with T063 (documentation during training)
-   **T068-T071** are optional and depend on T063 success + decision to pursue hierarchical
-   **T072-T074** are cleanup tasks after primary approach is validated

## Validation Checklist (Phase 5)

-   [ ] Single-agent SAC converges within 200 episodes (≥1 waypoint reached in 50%+ episodes)
-   [ ] Action space validated (no NaN, control bounds respected)
-   [ ] Reward components fire correctly without command hierarchy
-   [ ] Training artifacts generated (checkpoints, metrics, graphs)
-   [ ] Documentation updated to reflect primary training mode
-   [ ] Evaluation script produces deterministic results
-   [ ] (Optional) Hierarchical training matches single-agent performance after pre-training

## Success Metrics (Phase 5A Baseline)

| Metric                | Target (Episode 200)      | Measured | Status |
| --------------------- | ------------------------- | -------- | ------ |
| **Waypoints Reached** | ≥1 in 50%+ episodes       | TBD      | 🔄     |
| **Collision Rate**    | <50%                      | TBD      | 🔄     |
| **Episode Length**    | >150 steps avg            | TBD      | 🔄     |
| **Cumulative Reward** | >500 avg                  | TBD      | 🔄     |
| **Learning Trend**    | Upward over 50-ep windows | TBD      | 🔄     |

## Notes (Phase 5)

-   **Why single-agent first**: 80 episodes of hierarchical training proved workers cannot learn from scratch (zero loss, zero waypoints). Single SAC establishes viable baseline in ~1 week vs months of hierarchical debugging.
-   **Hierarchical optional**: Only revisit if interpretability is critical AND single-agent succeeds. Pre-training is mandatory for hierarchical RL.
-   **Reward rebalancing preserved**: Collision penalty (-50), progress rewards (+2.0 coef), command persistence (10 steps) were correct—problem was hierarchical bootstrapping, not rewards.
-   **Timeline estimate**: T059-T064 (~3-5 days implementation + 2-3 days training) = 1 week to working baseline. Compare to hierarchical pre-training (T068-T070: 3-4 weeks minimum).
