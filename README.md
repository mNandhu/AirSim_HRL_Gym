# AirSim HRL Training Framework

Hierarchical reinforcement learning stack targeting the AirSim simulator. A discrete DQN manager selects high-level commands which are executed by SAC-based low-level workers through a structured environment wrapper.

## Project Layout

```
src/
  airsim_env/       # Environment wrapper, reward, observation helpers
  hrl_agent/        # Manager, workers, coordination orchestration
  perception/       # Segmentation and detection adapters
  utils/            # Simulator runner, artifacts, reward docs, perf metrics
  config/           # Experiment schema, loader, seeds
  scripts/          # Experiment runners and profiling tools
tests/
  unit/             # Unit tests for config, reward, perception, runner
  integration/      # Determinism, termination, seeded rollout checks
configs/
  experiments/      # YAML experiment definitions
```

## Getting Started

1. Install Python with [uv](https://github.com/astral-sh/uv) (Python 3.12).
2. Install dependencies:
    ```pwsh
    uv sync
    ```
3. Run the seeded experiment script (requires AirSim simulator and Python API):
    ```pwsh
    uv run python src/scripts/run_experiment.py --config configs/experiments/baseline.yaml
    ```

Artifacts are written to `artifacts/<timestamp>_<experiment_id>/` including configuration, telemetry, and reward summaries.

## Reproducibility

-   Experiments are defined via `ExperimentDefinition` (scene, vehicle, poses, seeds, horizon).
-   Seeds are applied consistently across Python, NumPy, Torch, and AirSim using `config.seeds.apply_seed_bundle`.
-   Configuration hashes are stored with run metadata for deterministic reruns.

## Testing

Run the full suite with coverage (≥90% for non-learning code):

```pwsh
uv run pytest --cov=src --cov-report=term-missing
```

## Documentation

Reward component behavior is described in `docs/reward-contract.md`. Update the document by invoking `utils.reward_doc.generate_reward_contract` after changing reward logic.
