# Research: AirSim HRL Training Framework

## 1. Algorithms

- Decision: High-level DQN (stable-baselines3)
- Rationale: Mature discrete algorithm, stable target network & replay; fits limited action space (FOLLOW_LANE, TURN_LEFT_AT_INTERSECTION, etc.)
- Alternatives: PPO (slower convergence on discrete navigation), custom PyTorch DQN (reinvent maintenance), A2C (less sample efficient)

- Decision: Low-level SAC workers (stable-baselines3)
- Rationale: Continuous control robustness, entropy-driven exploration; widely benchmarked
- Alternatives: TD3 (less robust exploration), PPO (slower adaptation), custom actor-critic

## 2. Perception

- Decision: AirSim Segmentation camera for drivable mask
- Rationale: Perfect ground-truth -> zero training time
- Alternatives: Train segmentation network (adds data pipeline)

- Decision: YOLOv12 torch.hub detector
- Rationale: Latest available YOLO release; rapid model acquisition
- Alternatives: YOLOv8 local weights (manual mgmt), custom detector, classical CV

## 3. Reward Shaping

- Decision: Central module `src/airsim_env/reward.py` with composable components
- Rationale: Test isolation + change traceability
- Alternatives: Inline logic in `step` (harder to test)

## 4. Reproducibility

- Decision: Pydantic experiment schema (scene, vehicle, start_pose, goal_pose, horizon, seeds)
- Rationale: Deterministic reconstruction; validation on load
- Alternatives: Ad-hoc argparse flags

## 5. Simulator Lifecycle

- Decision: `src/utils/airsim_runner.py` controlling subprocess w/ retry (3 attempts), logs to `simulator_failures.log`
- Alternatives: Manual GUI; external shell scripts (less portable)

## 6. Configuration & Dependency Management

- Decision: `uv` for Python 3.12 env + lockfile; pinned versions for stable-baselines3, torch
- Alternatives: pip + requirements.txt (less reproducibility)

## 7. Telemetry & Logging

- Decision: JSON logs (run metadata, reward breakdown) + artifact bundles per run
- Alternatives: Plain text only (hard to parse)

## 8. Open Items (Non-blocking)

- Multi-worker scheduling (initial sequential; later priority queue)
- Performance threshold refinement after prototype profiling

## 9. Risks & Mitigations

| Risk                            | Impact            | Mitigation                             |
| ------------------------------- | ----------------- | -------------------------------------- |
| AirSim startup flakiness        | Delays CI         | Structured retries + fail-fast log     |
| YOLO model load latency         | Slower test start | Lazy load in fixture; cache weights    |
| Reward shaping complexity creep | Hard to test      | Enforce component pattern + unit tests |

## 10. Acceptance for Phase 0

- All critical unknowns resolved.
- Library & architectural choices enumerated.
- Deterministic execution path defined.
