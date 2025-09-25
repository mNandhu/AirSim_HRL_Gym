<!--
Sync Impact Report
Version change: N/A → 1.0.0
Modified principles: N/A → I. Modular Separation of Environment and Agent; N/A → II. Deterministic Simulation Reproducibility; N/A → III. Pytest Coverage for Non-Learning Infrastructure; N/A → IV. Documented State-Action-Reward Contract
Added sections: Core Principles; Engineering Constraints for AirSim HRL; Workflow & Quality Gates; Governance
Removed sections: Placeholder Principle 5 section
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ✅ updated
- .specify/templates/tasks-template.md ✅ updated
Follow-up TODOs: None
-->

# AirSim HRL Project Constitution

## Core Principles

### I. Modular Separation of Environment and Agent (NON-NEGOTIABLE)

- The `AirSimEnv` wrapper MUST only expose simulator-facing contracts (`reset`, `step`, `observe`, `close`, telemetry access) and MUST NOT import or depend on HRL agent implementations, managers, or workers.
- HRL agents (DQN manager, SAC workers, or future variants) MUST interact with the AirSim environment exclusively through typed interfaces defined in the environment package.
- Changes to environment APIs MUST maintain backward compatibility or include a migration note that documents breaking changes and provides upgrade guidance.
  Rationale: Maintaining a strict boundary keeps the simulator wrapper reusable across multiple HRL agents, preserves testability, and eliminates cyclic dependencies that slow experimentation.

### II. Deterministic Simulation Reproducibility

- Every experiment configuration MUST declare the AirSim map, vehicle asset, weather profile, time-of-day, initial pose, and termination conditions (goal pose or rollout horizon).
- Random seeds for Python, NumPy, PyTorch (or equivalent agent framework), and AirSim’s internal simulation MUST be set before `AirSimEnv.reset()` and persisted with the run artifact.
- Execution metadata (environment commit, agent commit, configuration hash, seed values) MUST be logged at rollout start and attached to experiment artifacts to enable bitwise reruns.
  Rationale: Explicit initial conditions and deterministic seeds ensure experiments can be reproduced, compared, and audited across machines and contributors.

### III. Pytest Coverage for Non-Learning Infrastructure

- The `AirSimEnv` wrapper, configuration loaders, telemetry utilities, and data pipelines MUST be covered by pytest unit tests that validate happy-path flows and failure handling.
- End-to-end pytest integration suites MUST exercise simulator connection bootstrapping, environment reset, step loops, and reward calculation using AirSim stubs or a controllable simulator harness.
- Continuous integration MUST fail if coverage for non-learning modules drops below 90% or if mandated tests are skipped.
  Rationale: Reinforcement learning stacks rely on deterministic scaffolding; rigorous tests on infrastructure prevent silent regressions that would invalidate performance claims.

### IV. Documented State-Action-Reward Contract

- Reward shaping logic MUST reside inside the environment package and be reachable via a dedicated module or method (e.g., `AirSimEnv.compute_reward`).
- Observations, action spaces, and reward components MUST be described in synchronized documentation (docstrings plus `docs/reward-contract.md`) generated or updated with every change.
- Any pull request modifying observation or reward semantics MUST update the documentation and associated unit or integration tests demonstrating the new behavior.
  Rationale: A transparent state-action-reward contract enables agent authors to reason about learning signals, prevents hidden coupling, and accelerates onboarding.

## Engineering Constraints for AirSim HRL

- Package the environment wrapper as a standalone module that can be imported without requiring the HRL agent stack; use dependency injection for AirSim clients.
- Maintain experiment configuration schemas under version control (`configs/experiments/*.yaml`) with machine-validated linting before execution.
- Persist telemetry (episode returns, safety events, failure reasons) alongside the configuration and seed bundle in a reproducible artifact store.
- Provide simulation fixtures that allow offline replay of logged trajectories for debugging without needing a running AirSim server.

## Workflow & Quality Gates

- Every plan or implementation proposal MUST include a Constitution Check confirming adherence to all four principles, with explicit remediation steps for any deviations.
- Code reviews MUST reject changes where agent modules import environment internals, or where reward logic moves outside the environment boundary.
- Pre-merge CI MUST run the full pytest suite (unit + integration), configuration schema validation, and a deterministic smoke rollout seeded from the latest reproducible experiment definition.
- Release notes MUST summarize experiment configurations affected, with links to updated reward documentation and coverage reports.

## Governance

- This constitution governs all work on the AirSim HRL project and supersedes conflicting team conventions.
- Amendments require consensus from project maintainers, an updated version number following semantic rules, and a migration plan covering environment interfaces, experiment schemas, and documentation impacts.
- Compliance is reviewed quarterly; unresolved violations MUST be documented with follow-up tasks and addressed before the next release.

**Version**: 1.0.0 | **Ratified**: 2025-09-25 | **Last Amended**: 2025-09-25
