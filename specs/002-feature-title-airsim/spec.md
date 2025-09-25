# Feature Specification: AirSim HRL Training Framework

**Feature Branch**: `002-feature-title-airsim`  
**Created**: 2025-09-25  
**Status**: Draft  
**Input**: User description: "Build a Python framework to train an HRL agent in AirSim. The framework must support a two-level hierarchy: 1. High-Level Manager (DQN) outputs discrete high-level commands (FOLLOW_LANE, TURN_LEFT_AT_INTERSECTION). 2. Low-Level Workers (SAC) consume the high-level command as part of the state and emit continuous controls (throttle, brake, steering). 3. Environment Wrapper (`AirSimEnv`) conforms to Gym, step accepts continuous controls, reset returns initial observation dictionary with processed camera image + active command. 4. Perception module processes AirSim segmentation camera for drivable area and a YOLOv12 detector for obstacles. 5. Simulator management utility starts/stops AirSim in GUI and headless (-RenderOffScreen) modes for pytest automation."

## Execution Flow (main)

```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines

- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements

- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")
- **Reproducibility reminder**: Capture AirSim environment, start/end poses, and required seeds when defining experiments.

### For AI Generation

When creating this spec from a user prompt:

1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## User Scenarios & Testing _(mandatory)_

### Primary User Story

An autonomy engineer needs to run repeatable HRL training sessions in AirSim where a discrete-command manager and continuous-control workers collaborate through a shared environment wrapper and perception stack.

### Acceptance Scenarios

1. **Given** a configured AirSim map with defined start and goal poses and saved random seeds, **When** the engineer launches the simulator via the management utility and triggers a training run, **Then** the system must spin up the environment wrapper, perception pipeline, DQN manager, and SAC workers with those deterministic settings and log reproducible rollout metadata.
2. **Given** an ongoing HRL training episode, **When** the DQN manager issues a command such as FOLLOW_LANE, **Then** the active worker must receive that command within the observation dictionary, produce continuous throttle/brake/steering outputs, and the environment must apply them, returning the next observation and reward shaped by the encapsulated reward module.

### Edge Cases

- What happens when AirSim fails to start in headless mode or disconnects mid-episode? The framework must surface a recoverable error and allow automated retry without manual intervention.
- How does the system handle missing or delayed perception inputs (e.g., segmentation camera unavailable or YOLO inference timeout)? The environment must provide fallback signals or abort with clear diagnostics to keep training metrics trustworthy.
- How are conflicting high-level commands resolved when multiple workers are scheduled concurrently? Document prioritisation or queuing rules to preserve determinism.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST provide a reusable `AirSimEnv` wrapper that conforms to the OpenAI Gym API, keeps reward logic encapsulated, and exposes a dictionary observation containing processed perception data and the current high-level command.
- **FR-002**: System MUST enable a DQN-based high-level manager to issue discrete navigation commands that are logged, timestamped, and forwarded in deterministic order to worker agents.
- **FR-003**: System MUST enable multiple SAC-based worker agents to consume the high-level command as part of their state input and emit continuous control signals applied by `AirSimEnv.step`.
- **FR-004**: System MUST include a perception module that fuses AirSim segmentation camera output with YOLOv12 detections to generate drivable-area masks and obstacle annotations used in observations and reward shaping.
- **FR-005**: System MUST offer a simulator management utility to start, monitor, and stop AirSim in GUI and headless (`-RenderOffScreen`) modes to support automated pytest execution.
- **FR-006**: System MUST allow experiment definitions to declare AirSim map, vehicle asset, weather profile, start pose, goal pose, episode horizon, and all random seeds, and persist these with rollout artifacts for reproducibility.
- **FR-007**: System MUST expose configuration hooks so tests can substitute mock AirSim backends or recorded sensor data for rapid CI execution.
- **FR-008**: System MUST capture telemetry (episode returns, failure reasons, perception statistics) and store it alongside configuration hashes for auditability.
- **FR-009**: System MUST document observation, action, and reward contracts, updating the documentation each time those semantics change.
- **FR-010**: System MUST surface alerts for simulator/startup failures by logging each failure event (timestamp + reason + retry count) to `simulator_failures.log` and exiting with non-zero status after 3 consecutive failed restart attempts (configurable).

- **FR-011**: System MUST target the "Neighborhood" map with the default sedan vehicle model as the canonical baseline environment for initial rollout validation.

- **FR-012**: System MUST pass the repository-managed `settings.json` to `AirSimNH.exe` via the `-settings` argument for both GUI and headless launches to guarantee deterministic simulator configuration.

- **FR-013**: System MUST achieve >=90% pytest coverage for all non-learning modules, including the environment wrapper, simulator utility, and configuration loaders, enforced via the CI pipeline.

### Key Entities _(include if feature involves data)_

- **ExperimentDefinition**: Captures AirSim scene metadata, vehicle type, start/goal poses, deterministic seeds, and episode horizon for each reproducible run.
- **HighLevelCommand**: Enumerates discrete manager outputs (e.g., FOLLOW_LANE, TURN_LEFT_AT_INTERSECTION) with priority metadata and command-to-worker routing rules.
- **ObservationPacket**: Aggregates processed camera tensors, command context, environment telemetry, and safety flags delivered to manager and workers.
- **TelemetryArtifact**: Stores rollout summaries, reward traces, perception diagnostics, and configuration hashes archived after each experiment.
- **SimulatorSession**: Represents an AirSim process lifecycle, including mode (GUI/headless), status, logs, and restart counters for automated management.

---

## Clarifications

### Session 2025-09-25

- Q: What is the default escalation channel for repeated simulator start failures (FR-010)? → A: Log to `simulator_failures.log` + non-zero exit after 3 retries
- Q: Is the Neighborhood map with default sedan the correct baseline (FR-011)? → A: Yes, Neighborhood + default sedan
- Q: Should a repository settings.json be enforced and passed to AirSim on startup? → A: Yes, via `-settings` argument

## Review & Acceptance Checklist

_GATE: Automated checks run during main() execution_

### Content Quality

- [x] No implementation details (languages, frameworks, APIs) — Spec describes capabilities (manager/worker hierarchy, perception roles) without prescribing libraries or code structures.
- [x] Focused on user value and business needs — Emphasizes reproducible HRL training workflow for autonomy engineer persona.
- [x] Written for non-technical stakeholders — Uses domain language (commands, episodes) while avoiding low-level architectural patterns.
- [x] All mandatory sections completed — Includes User Story, Acceptance Scenarios, Edge Cases, Functional Requirements, Entities, Clarifications.

### Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain — All former markers (FR-010, FR-011) resolved; none remain in text.
- [x] Requirements are testable and unambiguous — Each FR states observable behavior (e.g., log file creation, passing settings.json, deterministic seeds).
- [ ] Success criteria are measurable — Some metrics (e.g., retry count) present, but performance thresholds (latency, throughput) not yet quantified.
- [x] Scope is clearly bounded — Limits to HRL training framework (manager, workers, perception, environment, simulator management).
- [ ] Dependencies and assumptions identified — AirSim + YOLOv12 referenced but versions and resource assumptions not explicitly stated.

### Constitution Alignment

- [x] Environment/agent boundaries preserved (no HRL logic leaking into environment wrapper requirements) — FR-001 confines wrapper responsibilities; agent logic separate.
- [x] Experiment definition includes AirSim scene, vehicle, start/end pose, and deterministic seed plan — FR-006/FR-011 define scene, vehicle, seeds, poses; reproducibility emphasized.
- [x] State, action, and reward documentation updates captured (docs/tests noted) — FR-009 mandates documentation updates on semantic changes.

---

## Execution Status

_Updated by main() during processing_

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---
