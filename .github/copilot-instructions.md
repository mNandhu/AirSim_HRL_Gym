# GitHub Copilot Project Instructions

Purpose: Enable AI agents to contribute safely and productively to the AirSim HRL Training Framework.

## 1. Architectural Overview

- This repo (branch example: `002-feature-title-airsim`) documents a Hierarchical RL framework (no runtime code yet) targeting AirSim.
- Hierarchy: DQN Manager (discrete commands) → SAC Workers (continuous control) → `AirSimEnv` (Gym-like wrapper) → AirSim simulator.
- Perception stack: AirSim segmentation (ground-truth) + YOLOv12 detection (torch.hub) -> fused into ObservationPacket.
- Reproducibility core: ExperimentDefinition (scene, vehicle, start/goal pose, seeds, horizon) + SeedBundle ensures deterministic runs.
- Reward shaping isolated in `src/airsim_env/reward.py`; documentation generated to `docs/reward-contract.md`.

## 2. Constitution Enforcement (DO NOT VIOLATE)

1. Environment must not import agent code (one‑way dependency only).
2. All reproducibility inputs (scene, vehicle, poses, seeds) must appear in experiment config YAML before execution.
3. Non-learning infrastructure (env, perception adapters, sim runner, config loader) requires ≥90% pytest coverage.
4. State / action / reward contract changes require simultaneous doc + test updates.

## 3. Planned Source Layout (when implemented)

```
src/
  airsim_env/        # env.py, reward.py, observation.py
  hrl_agent/
    manager/         # dqn_manager.py
    workers/         # sac_worker.py
  perception/        # segmentation.py, detector.py
  utils/             # airsim_runner.py, artifacts.py, reward_doc.py, perf_metrics.py
  config/            # experiment.py, loader.py, seeds.py, settings_hash.py
  scripts/           # run_experiment.py, profile_step.py
tests/
  unit/              # Fine-grained tests
  integration/       # Seeded rollout, restart, determinism
  performance/       # Optional latency sampling
configs/experiments/  # YAML experiment definitions
```

## 4. Key Documents (Spec Phase)

- `specs/.../spec.md`: Feature requirements & clarifications.
- `plan.md`: Technical context + constitution checks (already PASS).
- `data-model.md`: Entities and relationships (ExperimentDefinition, ObservationPacket, etc.).
- `contracts/`: API contracts for environment, perception, simulator runner.
- `tasks.md`: Ordered TDD-first implementation plan (42 tasks).

## 5. Implementation Priorities

Follow `tasks.md` strictly in order groups:

1. Scaffold + config (T001–T006)
2. Failing tests first (T007–T014) – ensure they FAIL before implementing.
3. Core modules (T015–T026) – keep reward logic ONLY in reward module.
4. Integration wiring (T027–T034) – add seeded rollout + determinism check.
5. Polish & docs (T035–T042) – generate reward-contract doc last after reward module stable.

## 6. Testing & Quality

- Use pytest + pytest-cov; enforce `--cov=src --cov-report=term-missing`.
- Add unit tests per entity or module boundary; integration tests must set explicit seeds before env reset.
- Determinism test: Run same config twice → identical cumulative reward & config hash.
- Add latency measurement helpers for perception and step loop (<50ms p50 target for perception path).

## 7. Perception Guidelines

- Segmentation: call AirSim `ImageType.Segmentation` → integer mask (no training step).
- Detection: torch.hub load YOLO; lazy load inside first call to reduce startup time.
- Observation builder merges: RGB → normalization; produce segmentation mask & detection list; embed command & telemetry.

## 8. Simulator Management

- `airsim_runner.py` launches `AirSimNH.exe` with `-settings` and optionally `-RenderOffScreen`.
- Retry policy: up to 3 attempts; log structured JSON + append to `simulator_failures.log`.
- Provide pytest fixture `airsim_session` (session scope) for headless use.

## 9. Reproducibility & Seeds

- Apply seeds (python, numpy, torch, airsim) BEFORE first `reset()`.
- Store seeds + config hash in artifacts; do not rely on implicit defaults.

## 10. Reward Components (Initial Formulas)

Reference (keep configurable constants centralized):

- progress = prev_distance_to_goal - curr_distance_to_goal
- lane_adherence = lane_mask_coverage_ratio
- collision = -1 \* int(collision_occurred)
- command_completion = completion_bonus \* int(command_objective_met)
- idle_penalty = -idle_penalty_coef \* int(speed_mps < idle_threshold & progress_possible)

## 11. Contribution Conventions

- One task per commit where practical; include task ID (e.g., "T015: implement reward module scaffold").
- Never move reward logic into env.step body; call reward.compute.\* functions instead.
- Keep environment free of any stable-baselines3 imports.
- Update `docs/reward-contract.md` and add/adjust tests whenever observation or reward semantics change.

## 12. Common Pitfalls to Avoid

- Forgetting to seed AirSim before reset (breaks determinism test).
- Hardcoding detection class IDs without documenting mapping.
- Logging only human-readable text (always include structured JSON fields for artifacts).
- Allowing perception code to silently swallow model load failures (raise explicit error).

## 13. When Unsure

- Check contracts first → data-model → tasks ordering.
- If new reward component introduced: add test, update contract table + doc, increment doc version.

---

Feedback welcome: Identify unclear areas (e.g., artifact directory naming, performance thresholds) for iterative refinement.
