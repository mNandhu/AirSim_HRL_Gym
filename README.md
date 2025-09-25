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

### Quick Start (Training & Evaluation)

1. **Install dependencies:**

    ```pwsh
    uv sync
    ```

2. **Check setup and get instructions:**

    ```pwsh
    python run.py
    ```

3. **Launch AirSim**: Start the AirSim simulator with Neighbourhood environment

4. **Train the HRL agent** (watch the car learn to drive):

    ```pwsh
    uv run python src/scripts/train_and_eval.py train --config configs/experiments/training.yaml
    ```

5. **Evaluate trained model** (watch the trained car drive):
    ```pwsh
    uv run python src/scripts/train_and_eval.py eval --config configs/experiments/baseline.yaml --models models
    ```

### Training Mode Options

```pwsh
# Basic training with GUI (recommended for first runs)
uv run python src/scripts/train_and_eval.py train --config configs/experiments/training.yaml

# Training options
--episodes 100           # Number of training episodes
--save-interval 10       # Save models every N episodes
--mode gui              # Show AirSim window (use 'headless' to hide)
--resume                # Resume from previously saved models
--models models         # Directory to save/load models
```

### Evaluation Mode Options

```pwsh
# Run trained model once
uv run python src/scripts/train_and_eval.py eval --config configs/experiments/baseline.yaml --models models

# Evaluation options
--continuous            # Run multiple episodes continuously
--mode gui             # Show AirSim window
--max-steps 1000       # Maximum steps per episode
```

### Legacy Single-Episode Mode

For single experiment runs (original implementation):

```pwsh
uv run python src/scripts/run_experiment.py --config configs/experiments/baseline.yaml
```

Artifacts are written to `artifacts/<timestamp>_<experiment_id>/` including configuration, telemetry, and reward summaries.

## Prerequisites

-   **AirSim Simulator**: Download and install AirSim with a driving environment (Neighbourhood recommended)
-   **Python 3.12+**: Install via [uv](https://github.com/astral-sh/uv) or standard Python installer
-   **Dependencies**: Automatically installed via `uv sync` or manually: `airsim`, `stable-baselines3`

## Training Process

The HRL system trains two types of models:

-   **DQN Manager**: Selects high-level commands (FOLLOW_LANE, TURN_LEFT, TURN_RIGHT, STOP)
-   **SAC Workers**: Execute low-level continuous control (throttle, brake, steering) for each command

During training, you'll see:

-   Real-time reward scores and episode statistics
-   Periodic model saving (every 10 episodes by default)
-   Live visualization in AirSim window showing the car learning to drive

## Model Files

Trained models are saved to `models/` directory:

-   `dqn_manager.zip`: High-level command selection policy
-   `sac_follow_lane.zip`, `sac_turn_left_at_intersection.zip`, etc.: Low-level control policies

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
