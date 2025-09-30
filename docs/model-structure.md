# Model Structure and Resume Options

## Directory Structure

Each training run creates the following structure:

```
models/
└── 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7/
    ├── dqn_manager.zip                      ← Latest checkpoint
    ├── sac_follow_lane.zip                  ← Latest checkpoint
    ├── sac_turn_left_at_intersection.zip    ← Latest checkpoint
    ├── sac_turn_right_at_intersection.zip   ← Latest checkpoint
    ├── sac_stop.zip                         ← Latest checkpoint
    └── best-models/                         ← Best checkpoint (highest reward)
        ├── dqn_manager.zip
        ├── sac_follow_lane.zip
        ├── sac_turn_left_at_intersection.zip
        ├── sac_turn_right_at_intersection.zip
        └── sac_stop.zip
```

## Resume Options

| Command                                | Loads From                           | Use Case                                |
| -------------------------------------- | ------------------------------------ | --------------------------------------- |
| `--resume-from {run_id}`               | `models/{run_id}/` (top-level)       | Continue from most recent checkpoint    |
| `--resume-from {run_id} --resume-best` | `models/{run_id}/best-models/`       | Continue from highest-reward checkpoint |
| `--resume-from {run_id}/best-models`   | `models/{run_id}/best-models/`       | Same as above (explicit path)           |
| `--resume`                             | `models/` (top-level, old structure) | Backward compatibility                  |

## Quick Examples

**Resume from latest checkpoint:**

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume-from 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7 \
  --episodes 50
```

**Resume from best checkpoint:**

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --resume-from 20250930T154745Z_train_8591ac4c-07d7-4069-b327-cfdf72b999a7 \
  --resume-best \
  --episodes 50
```

## Decision Guide

**When to use latest models:**

-   ✅ You want to continue training naturally from where it stopped
-   ✅ Preserves the exact exploration/exploitation state
-   ✅ Best for long training runs that were interrupted

**When to use best models:**

-   ✅ Previous training may have degraded after the best checkpoint
-   ✅ You want to start from known good performance
-   ✅ Fine-tuning or transfer learning scenarios

**Pro tip:** Check `artifacts/{run_id}/training_stats.json` to see reward trends and decide which checkpoint to use.
