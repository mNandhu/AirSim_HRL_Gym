# Implementation Plan: AirSim HRL Training Framework

**Branch**: `002-feature-title-airsim` | **Date**: 2025-09-25 | **Spec**: `G:/Projects/HRL_SelfDrivingCar/AirSim_HRL_Gym/specs/002-feature-title-airsim/spec.md`
**Input**: Feature specification from `/specs/002-feature-title-airsim/spec.md`

## Execution Flow (/plan command scope)

```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:

- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Implement a reproducible hierarchical reinforcement learning framework in AirSim with a discrete DQN high-level manager and multiple SAC low-level workers. The environment wrapper (`AirSimEnv`) remains agent-agnostic and Gym-compliant, providing observation dictionaries containing processed segmentation masks, YOLOv12 detections, telemetry, and the active high-level command. The system emphasizes deterministic experiment configuration (scene, vehicle, poses, seeds) and >90% pytest coverage for all non-learning infrastructure (environment wrapper, simulator runner, perception adapters, configuration loader). Perception leverages AirSim’s perfect ground-truth segmentation (no custom segmentation training) and a torch.hub YOLOv12 model for object detection. Simulator lifecycle management is automated via a subprocess-based runner with a pytest fixture for headless CI usage.

## Technical Context

**Language/Version**: Python 3.12 (managed by `uv`)  
**Primary Dependencies**: stable-baselines3 (DQN, SAC), torch + torch.hub (YOLOv12), msgpack/json (config artifacts), AirSim Python API, numpy, pydantic (config & schema validation), pytest + pytest-cov  
**Storage**: Filesystem artifact bundles (configs/, artifacts/, logs/) — no database required initially  
**Testing**: pytest (unit + integration + smoke seeded rollout), coverage threshold ≥90% for non-learning code  
**Target Platform**: Developer workstations & CI runners (Windows primary; Linux headless where supported by AirSim)  
**Project Type**: single  
**Performance Goals**: Environment step (excluding NN inference) median <10ms; perception pipeline end-to-end <50ms per frame (segmentation retrieval + detection); simulator startup retry window <120s total  
**Constraints**: Deterministic seeds for Python/NumPy/PyTorch/AirSim; headless mode via `-RenderOffScreen`; graceful simulator restart ≤3 attempts  
**Scale/Scope**: Single vehicle, Neighborhood map baseline; extensible to multi-worker concurrency (>1 SAC instance) by command routing layer

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- [x] **Modularity**: `AirSimEnv` defined in isolation (`src/airsim_env/`); agents consume only public methods (`reset`, `step`, `get_observation`, `close`). No reciprocal imports.
- [x] **Reproducibility**: Experiment schema (scene, vehicle, start pose, goal pose, seeds, horizon) to reside in `configs/experiments/*.yaml`; loader seeds Python/NumPy/PyTorch/AirSim before `reset`.
- [x] **Testability**: Planned pytest suites: unit (env methods, reward calc, config loader), integration (seeded rollout, simulator lifecycle), coverage gate ≥90% enforced in CI.
- [x] **State-Action-Reward Clarity**: Reward module isolated in `src/airsim_env/reward.py`; doc generation to `docs/reward-contract.md` with versioned updates.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```
src/
├── airsim_env/              # Environment wrapper (reset/step/reward/observation)
├── hrl_agent/
│   ├── manager/             # DQN high-level manager wrapper
│   └── workers/             # SAC low-level workers
├── perception/              # Segmentation + detection adapters
├── utils/                   # Simulator runner, artifacts, reward doc tooling
├── config/                  # Experiment schema, seed handling, loader
└── scripts/                 # run_experiment.py, profiling scripts

tests/
├── unit/                    # Reward, perception, config, sim runner
├── integration/             # Seeded rollout, restart recovery
└── performance/             # Optional profiling / latency checks

configs/
└── experiments/             # YAML experiment definitions

artifacts/                   # Generated run outputs (gitignored)
docs/
└── reward-contract.md       # Generated reward documentation
```

**Structure Decision**: Option 1 (single project). Custom subpackages: `src/airsim_env/`, `src/hrl_agent/manager/`, `src/hrl_agent/workers/`, `src/perception/`, `src/utils/`.

## Phase 0: Outline & Research

Key Decision Areas & Outcomes (recorded in `research.md`):

| Topic                   | Decision                            | Rationale                                                          | Alternatives                                                    |
| ----------------------- | ----------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| High-Level Algorithm    | DQN via stable-baselines3           | Mature, discrete action support, replay buffer & target net stable | Custom PyTorch DQN (slower), PPO (less discrete specialization) |
| Low-Level Control       | SAC via stable-baselines3           | Continuous action robustness, entropy regularization               | TD3 (less robust exploration), PPO                              |
| Perception Segmentation | Use AirSim `ImageType.Segmentation` | Zero training time, perfect labels accelerate iteration            | Train Enet/DeepLab (longer pipeline)                            |
| Object Detection        | YOLOv12 torch.hub weights           | Latest available YOLO release; quick load                          | YOLOv8 local weights (maintenance), custom detector             |
| Reward Shaping          | Centralized module `reward.py`      | Maintain transparency & testability                                | Inline reward logic in step (harder to test)                    |
| Reproducibility         | Config schema + seeded loader       | Deterministic experiment recreation                                | Ad-hoc script args (error prone)                                |
| Simulator Mgmt          | `airsim_runner.py` + pytest fixture | Automates headless CI lifecycle                                    | Manual GUI startup                                              |
| Config Validation       | pydantic models                     | Strict schema + type coercion                                      | Manual dict validation                                          |
| Logging & Telemetry     | Structured JSON + log files         | Machine parsable; reproducible artifacts                           | Plain text only                                                 |

Open Items (tracked, not blockers):

- Performance tuning thresholds (may refine after first profiling pass).
- Multi-worker scheduling policy (initially sequential dispatch, upgrade to priority queue later).

Research Tasks Completed Indicators:

- All algorithm/library choices recorded with alternatives.
- Deterministic seed orchestration flow drafted.
- Perception inference path latency target drafted (<50ms).

`research.md` will contain the expanded rationale paragraphs.

## Phase 1: Design & Contracts

_Prerequisites: research.md complete_

1. Data Model (`data-model.md`): Define entities:

   - ExperimentDefinition (fields: id/hash, scene, vehicle, poses, seeds, horizon, timestamp)
   - HighLevelCommand (enum + priority)
   - ObservationPacket (image tensor spec, segmentation mask shape, detections structure, command, telemetry fields)
   - TelemetryArtifact (episode_id, returns, reward_components, failure_reason, config_hash)
   - SimulatorSession (pid, mode, start_time, retries, status)

2. Environment Contract (`contracts/environment.md`):

   - Methods: `reset(seed_bundle) -> ObservationPacket`, `step(action_dict) -> (ObservationPacket, reward: float, terminated: bool, truncated: bool, info: dict)`, `compute_reward(prev_obs, action, new_obs) -> float`, `close()`.
   - Action space definition: throttle [-1,1], brake [0,1], steering [-1,1].
   - Discrete command space enumeration reference.

3. Perception Contract (`contracts/perception.md`):

   - Interfaces: `capture_segmentation()`, `run_detection(image)`, `build_observation()`.
   - Detection output schema (list[ {class_id:int, bbox:[x,y,w,h], confidence:float} ]).

4. Simulator Management Contract (`contracts/sim_runner.md`):

   - `launch(mode, settings_path) -> SimulatorSession`.
   - `ensure_running(session)`, `terminate(session)`, context manager semantics.

5. Quickstart (`quickstart.md`):

   - Steps: install via uv, download YOLO weights (auto via torch.hub), run seeded experiment, view artifacts & reward documentation.

6. Initial failing tests (stub placeholders) enumerated in plan (actual files generated during implementation, not by /plan command):

   - Unit: reward calculation, observation builder, config loader, sim runner lifecycle.
   - Integration: seeded rollout, restart after failure injection.

7. Agent context file update via script (will record stable-baselines3, YOLOv12 usage).

Outputs for this phase: `data-model.md`, `contracts/` directory with environment, perception, sim_runner markdown, `quickstart.md`.

## Phase 2: Task Planning Approach

_This section describes what the /tasks command will do - DO NOT execute during /plan_

**Task Generation Strategy**:

- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 35-40 tasks (expanded for perception + reproducibility + coverage gates) in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

_These phases are beyond the scope of the /plan command_

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

_Fill ONLY if Constitution Check has violations that must be justified_

| Violation                  | Why Needed         | Simpler Alternative Rejected Because |
| -------------------------- | ------------------ | ------------------------------------ |
| [e.g., 4th project]        | [current need]     | [why 3 projects insufficient]        |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient]  |

## Progress Tracking

_This checklist is updated during execution flow_

**Phase Status**:

- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---

_Based on Constitution v1.0.0 - See `/memory/constitution.md`_
