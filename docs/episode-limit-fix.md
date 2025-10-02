# Episode Limit Fix for Single-Agent Training

**Date**: October 2, 2025  
**Issue**: Training ran past the specified `--episodes` count  
**Status**: ✅ Fixed

---

## Problem

When running:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 20 \
  --save-interval 10 \
  --mode headless \
  --detector-model yolo12n
```

Training continued past episode 20 (went to 23+ before being interrupted).

---

## Root Cause

The `--episodes` parameter was being used to calculate `total_timesteps` for SAC's `model.learn()`:

```python
total_timesteps = args.episodes * experiment.horizon
# With --episodes 20 and horizon=1000:
# total_timesteps = 20 * 1000 = 20,000 steps
```

However, SAC trains for a **fixed number of steps**, not episodes. If episodes end early (due to collisions, truncations, etc.), 20,000 steps could easily be 25-30 episodes.

**Example**:

-   Episode 1: 100 steps (collision)
-   Episode 2: 80 steps (collision)
-   Episode 3: 120 steps (collision)
-   ...
-   Episode 23: Still under 20,000 total steps!

---

## Solution

Modified `MetricsCallback` to explicitly track and limit the number of episodes:

### 1. Added `max_episodes` parameter to callback

```python
class MetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        metrics_tracker: MetricsTracker,
        artifact_manager: ArtifactManager,
        model_dir: Path,
        save_interval: int,
        waypoints: Sequence[tuple[float, float]],
        max_episodes: int | None = None,  # NEW
    ) -> None:
        # ...
        self._max_episodes = max_episodes
```

### 2. Stop training when limit reached

```python
def _on_step(self) -> bool:
    # ... episode completion logic ...

    if done:
        self._episode_index += 1
        # ...

        # Stop training if max episodes reached
        if self._max_episodes is not None and self._episode_index >= self._max_episodes:
            print(f"\n✅ Reached maximum episodes ({self._max_episodes}), stopping training...")
            return False  # Returning False stops training

    return True
```

### 3. Pass episodes count to callback

```python
callback = MetricsCallback(
    metrics_tracker=metrics_tracker,
    artifact_manager=artifact_manager,
    model_dir=model_dir,
    save_interval=args.save_interval,
    waypoints=waypoints,
    max_episodes=args.episodes,  # NEW
)
```

---

## Behavior After Fix

### With `--episodes 20`:

1. SAC trains with `total_timesteps = 20,000` (upper bound)
2. Callback counts episodes as they complete
3. After episode 20 finishes, callback returns `False`
4. Training stops immediately with message:
    ```
    ✅ Reached maximum episodes (20), stopping training...
    ```

### Why keep large `total_timesteps`?

-   Ensures we don't run out of timesteps before reaching the episode limit
-   Episode lengths vary (collisions end early, successful runs go longer)
-   Acts as a safety upper bound
-   Callback provides the actual episode-based stopping condition

---

## Testing

### Unit/Integration Tests

All existing tests pass:

```bash
uv run pytest tests/unit/test_single_agent_env.py \
             tests/integration/test_single_agent_rewards.py \
             --no-cov -q
# Result: 9 passed in 5.53s
```

### Manual Test

Created `test_episode_limit.py` to verify callback stops after exactly 5 episodes when `max_episodes=5` is set. Test passed successfully.

---

## Related Files Modified

1. `src/scripts/train_single_agent.py`
    - Added `max_episodes` parameter to `MetricsCallback`
    - Added early stopping logic in `_on_step()`
    - Pass `args.episodes` to callback
    - Added comment explaining `total_timesteps` calculation

---

## Usage

### Training with episode limit (default behavior):

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 20
```

→ Stops after exactly 20 episodes

### Training with timestep limit (advanced):

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --total-timesteps 50000
```

→ Stops after 50,000 steps (ignores episode count)

### Combining both:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100 \
  --total-timesteps 50000
```

→ Stops at whichever comes first (100 episodes OR 50,000 steps)

---

## Backward Compatibility

✅ **No breaking changes**

-   `max_episodes` defaults to `None` (no limit)
-   Existing code that doesn't pass `max_episodes` works unchanged
-   `--episodes` parameter now works as users expect

---

## Future Improvements

Could add:

1. Progress bar showing episode count: `Episode 15/20`
2. Estimated time remaining based on average episode length
3. Early stopping based on performance metrics (e.g., "stop if 10 consecutive episodes reach goal")

---

**Status**: ✅ **Fixed and tested** - Training now respects `--episodes` limit
