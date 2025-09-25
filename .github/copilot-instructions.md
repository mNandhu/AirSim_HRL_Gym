# GitHub Copilot Project Instructions

Purpose: help AI contributors extend the AirSim hierarchical RL stack safely and productively.

## 1. Big-picture architecture

-   **Hierarchy:** `hrl_agent.manager.DQNManager` selects discrete commands → `hrl_agent.workers.SACWorker` outputs throttle/brake/steering → `AirSimEnv` (`src/airsim_env/env.py`) applies them through an AirSim client adapter.
-   **Perception pipeline:** `perception.segmentation.SegmentationAdapter` captures masks, `perception.detector.YoloDetector` loads Ultralytics models, and `perception.pipeline.PerceptionPipeline` converts AirSim frames to `(H,W,3)` NumPy arrays before fusing results into an `ObservationPacket`.
-   **Reproducibility:** Experiment YAMLs flow through `config.loader.load_experiment` into immutable `ExperimentDefinition`s that embed a `SeedBundle`; `config.seeds.apply_seed_bundle` seeds Python/NumPy/Torch/AirSim on every reset.
-   **Artifacts:** `utils.artifacts.ArtifactManager` writes JSONL logs & summaries under `artifacts/<timestamp>_<id>/`; reward behavior is documented in `docs/reward-contract.md` and must stay in sync with code.

## 2. Source layout highlights

-   `src/airsim_env/`: environment façade, reward calculator, observation helpers. Keep this layer free of agent imports.
-   `src/hrl_agent/`: command coordination (`coordination.py`), manager/workers, `HRLOrchestrator` for episode rollouts.
-   `src/perception/`: segmentation and YOLO adapters (loading now uses `from ultralytics import YOLO` with a dummy fallback when downloads fail).
-   `src/utils/airsim_runner.py`: launch AirSim via `airsim_session`, retries with structured logs in `simulator_failures.log`.
-   `src/scripts/train_and_eval.py`: primary CLI (`train` / `eval`) that wires simulator adapter, perception pipeline, artifacts, and coordination.
-   Specs & contracts live under `specs/002-feature-title-airsim/`; consult `plan.md`, `data-model.md`, `contracts/*`, and `tasks.md` for intent before modifying interfaces.

## 3. Working locally

-   Install deps with `uv sync` (Python ≥3.12). Ensure AirSim is installed and `settings.json` plus `AIRSIM_EXECUTABLE` are configured.
-   Run the quality gate with `uv run pytest` (pyproject enforces `--cov=src --cov-report=term-missing --cov-fail-under=85`). Integration tests mock AirSim but still expect deterministic seeds.
-   Smoke training by launching AirSim, then `uv run python src/scripts/train_and_eval.py train --config configs/experiments/training.yaml --episodes 1 --max-steps 200`. Use `--detector-model <weights>` to pick YOLO checkpoints; on load failure the dummy detector returns no boxes so training proceeds.
-   Evaluate saved models with `uv run python src/scripts/train_and_eval.py eval --config configs/experiments/baseline.yaml --models models --continuous`.

## 4. Coding conventions & guardrails

-   Preserve one-way dependencies: env/perception/utils must not import `hrl_agent`. Keep reward math inside `src/airsim_env/reward.py`; update `docs/reward-contract.md` and related tests upon changes.
-   Always convert AirSim image responses to NumPy arrays before calling Ultralytics to avoid “Unsupported image type” errors (`PerceptionPipeline._to_numpy_image`).
-   Logging and artifacts should remain structured JSON and include config hashes from `config.loader.compute_config_hash`.
-   When adding experiment fields or seeds, extend `ExperimentDefinition`, update YAML fixtures, and ensure determinism tests stay green.
-   Expand existing pytest suites (`tests/unit`, `tests/integration`) instead of ad-hoc scripts; perception/runner additions generally need both unit coverage and targeted integration tests.

## 5. When unsure

-   Trace API contracts in `specs/002-feature-title-airsim/contracts/` before altering data flows, then cross-check `tasks.md` for sequencing.
-   Run `python run.py` for an interactive checklist the repo presents to contributors.
-   Leave notes here if new non-obvious workflows emerge (extra scripts, simulator setup quirks, YOLO weight management).

Feedback welcome—flag unclear areas (artifact naming, AirSim launch expectations, YOLO usage) so we can refine these instructions.
