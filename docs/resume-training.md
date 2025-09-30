# Resuming Training from Previous Runs

## Overview

Training runs now save models in run-specific directories under `models/`:

```
models/
  20250930T154745Z_train_8591ac4c.../
    dqn_manager.zip
    sac_follow_lane.zip
    sac_turn_left_at_intersection.zip
    ...
    best-models/
      dqn_manager.zip
      ...
```

## Usage

### Resume from latest models (most recent checkpoint)

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume-from 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7 \
  --episodes 100 \
  --save-interval 10
```

This will:

1. Load **latest models** from `models/20250930T154745Z_train_8591ac4c.../` (top-level in run dir)
2. Continue training from those weights
3. Save new checkpoints to a **new** run directory (e.g., `models/20250930T160102Z_train_newid/`)

### Resume from best models (highest reward checkpoint)

**Option 1: Using --resume-best flag**

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume-from 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7 \
  --resume-best \
  --episodes 100
```

**Option 2: Explicit path to best-models/**

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume-from 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7/best-models \
  --episodes 100
```

Both load from `models/{run_id}/best-models/` subdirectory.

### Legacy resume (backward compatibility)

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume \
  --models models \
  --episodes 100
```

This tries to load from top-level `models/dqn_manager.zip`, `models/sac_*.zip` (old structure).

## Finding Available Runs

List all runs:

```bash
ls models/
```

If you specify a non-existent run ID, the script will show you the last 10 available runs:

```
❌ Error: Resume directory not found: models/nonexistent_id

Available runs in models:
  - 20250930T153517Z_train_e020805e-54c0-490e-b6b8-7ae5339872ab
  - 20250930T153637Z_train_5da85237-29f3-4278-9677-4c79cc68428c
  - 20250930T154546Z_train_5c4b5d3a-89fc-4dc2-b493-09590d859d5f
  ...
```

## Key Distinctions

**Latest vs Best Models:**

-   **Latest models** (`models/{run_id}/`): Most recent checkpoint (saved every `--save-interval` episodes)
-   **Best models** (`models/{run_id}/best-models/`): Checkpoint with highest cumulative reward seen so far

**Which should I use?**

-   Use **latest** to continue from where training left off (preserves exploration state)
-   Use **best** to start from the highest-performing checkpoint (may have better initial performance)

## Notes

-   The `--resume-from` flag takes precedence over `--resume`
-   `--resume-best` only works with `--resume-from` (not with legacy `--resume`)
-   When resuming, models are loaded from the specified run directory
-   New checkpoints are always saved to a fresh run directory (no overwrites)
-   You can use explicit paths like `{run_id}/best-models` instead of `--resume-best`
