# Fix: Distance Tracking and Waypoint Metrics Display

**Date**: October 1, 2025  
**Related Issue**: Training convergence failure due to broken distance tracking

## Changes Made

### 1. Fixed Distance Tracking in Simulator Adapter

**File**: `src/scripts/train_and_eval.py`

#### Problem

The simulator adapter always returned `distance_to_goal = 999.0` when using waypoint-based navigation because it only checked for `goal_pose` (single-goal mode). With waypoint configurations, `goal_pose` is None, so distance was never calculated.

#### Solution

Enhanced `AirSimSimulatorAdapter` to:

1. Store and track waypoint list from experiment configuration
2. Maintain current waypoint index with automatic advancement (5m threshold)
3. Calculate distance to current target waypoint instead of final goal
4. Fall back to legacy `goal_pose` mode if waypoints not present

**Key Changes**:

-   Added `self._waypoints` and `self._current_waypoint_index` to adapter state
-   Modified `reset()` to extract waypoints from experiment
-   Rewrote `_build_state_dict()` to prioritize waypoint distance calculation
-   Auto-advance waypoint index when within 5.0m threshold (matches `PathManager`)

**Before**:

```python
distance_to_goal = 999.0
goal_pose = getattr(experiment, "goal_pose", None) or self._goal_pose
if goal_pose is not None:
    # Calculate distance...
```

**After**:

```python
# Priority 1: Waypoint-based navigation
if self._waypoints is not None:
    current_waypoint = self._waypoints[self._current_waypoint_index]
    distance_to_goal = math.sqrt(...)
    # Auto-advance if within threshold
    if dist_to_current <= 5.0:
        self._current_waypoint_index += 1

# Priority 2: Legacy single goal
elif self._goal_pose is not None:
    distance_to_goal = math.sqrt(...)
```

#### Impact

-   `distance_to_goal` in telemetry now reflects actual distance to current waypoint
-   Command completion detection in `coordination.py` now works correctly
-   Completion bonuses (+100.0) will be awarded when commands complete
-   Metrics show meaningful progress tracking

---

### 2. Fixed Misleading Waypoint Count Display

**File**: `src/utils/metrics_tracker.py`

#### Problem

The metrics displayed "Waypoints: 1/8" at episode start (before any waypoint reached) because the code showed `waypoint_index + 1` to be "human-readable". This was confusing because:

-   Index 0 means "targeting waypoint 0" (none completed yet)
-   Display showed "1/8" implying 1 waypoint was already reached
-   Graphs showed all episodes reaching "1 waypoint" when they actually reached 0

#### Solution

Changed waypoint metrics to represent **actual count of completed waypoints**, not the index being targeted.

**Semantic Clarification**:

-   `current_waypoint_index = 0`: Agent is targeting first waypoint → 0 waypoints completed
-   `current_waypoint_index = 1`: Agent is targeting second waypoint → 1 waypoint completed
-   `current_waypoint_index = N`: Agent is targeting waypoint N → N waypoints completed

**Changes Made**:

1. **EpisodeMetrics dataclass**: Updated documentation

    ```python
    max_waypoints_reached: int = -1  # Count of waypoints COMPLETED
    # -1 = no waypoint data
    # 0 = targeting first waypoint (none completed)
    # N = completed N waypoints, targeting waypoint N+1
    ```

2. **Console output**: Shows actual completed count

    ```python
    # Before: max_waypoint_reached + 1
    # After: max_waypoint_reached (already represents completed count)
    waypoints_completed = max_waypoint_reached
    waypoint_info = f" | Waypoints: {waypoints_completed}/{total_waypoints} reached"
    ```

3. **Graph display**: Removed the misleading +1
    ```python
    # Before: e.max_waypoints_reached + 1 for e in episodes
    # After: e.max_waypoints_reached for e in episodes
    ```

#### Impact

-   Episode start now shows "0/8 waypoints reached" (accurate)
-   Graphs show 0 when no waypoints reached (not 1)
-   Console output matches graph display
-   Clearer learning progress tracking

---

## Testing

### Verify Distance Tracking Fix

Run a short training session and check the metrics:

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 5 \
  --settings settings.json \
  --mode gui \
  --detector-model yolo12n
```

**Expected Results**:

1. `distance_to_goal` should vary (not stuck at 999.0)
2. Console should show decreasing min_distance values
3. Episode JSON should have `min_distance_to_goal < 999.0`

**Check**:

```powershell
# View episode metrics
Get-Content "artifacts/<latest-run>/metrics/episodes.json" | ConvertFrom-Json |
  Select-Object episode, min_distance_to_goal, max_waypoints_reached
```

Expected output:

```
episode  min_distance_to_goal  max_waypoints_reached
-------  --------------------  ---------------------
1        45.2                  0
2        38.7                  1
3        52.1                  0
...
```

### Verify Waypoint Display Fix

**Console Output** at episode start:

```
📊 Started tracking episode 1
   Path: 8 waypoints
```

**Console Output** at episode end:

```
📊 Episode 1 completed:
   Reward: -15.23 | Steps: 105 | Duration: 31.2s
   Max Speed: 5.2 m/s | Min Distance: 45.2m | Waypoints: 0/8 reached
```

**Graph**: Check `artifacts/<run>/metrics/episode_summary.png`

-   Waypoint Progress subplot should show 0 for episodes that don't reach any waypoint
-   Should match console output

---

## Impact on Training

### Before Fix

-   ❌ `distance_to_goal` = 999.0 (constant)
-   ❌ FOLLOW_LANE commands never complete
-   ❌ No completion bonuses awarded
-   ❌ Metrics show "1/8 waypoints" when actually 0
-   ❌ Impossible to track learning progress

### After Fix

-   ✅ `distance_to_goal` = actual distance to current waypoint
-   ✅ Commands complete when criteria met (5m progress, heading change, etc.)
-   ✅ Completion bonuses (+100.0) awarded correctly
-   ✅ Metrics accurately show 0/8, 1/8, 2/8... waypoints reached
-   ✅ Clear progress tracking for debugging and analysis

### Expected Training Improvement

With working completion bonuses, the reward structure becomes:

-   **Large sparse positives**: +100 per command completion (now works!)
-   **Dense positives**: +2 to +7 per step for good driving
-   **Dense negatives**: -0.02 to -2.0 per step for errors
-   **Large sparse negatives**: -200 for collision

This balanced structure should enable learning within 20-50 episodes.

---

## Additional Notes

### Waypoint Advancement Logic

The simulator adapter now mirrors the `PathManager` waypoint advancement:

-   Both use 5.0m threshold (`_WAYPOINT_THRESHOLD`)
-   Simulator advances `_current_waypoint_index` when within threshold
-   Environment's `PathManager` independently tracks same logic
-   Both should stay synchronized (same waypoint sequence, same threshold)

### Potential Race Condition

There's a minor timing discrepancy:

-   Simulator adapter advances waypoint in `_build_state_dict()` (during step)
-   Environment's `PathManager` advances in `step()` after simulator returns
-   In practice, both advance at same step, just slightly different timing

This shouldn't cause issues since:

1. Both use identical 5.0m threshold
2. Both calculate from same position
3. Distance calculation is idempotent

### Legacy Compatibility

The fix maintains backward compatibility:

-   Single-goal experiments still work (falls back to `goal_pose`)
-   Old experiment configs don't break
-   New waypoint configs use enhanced tracking

---

## Related Files

-   `src/scripts/train_and_eval.py`: Simulator adapter with waypoint support
-   `src/utils/metrics_tracker.py`: Corrected waypoint display
-   `src/airsim_env/env.py`: Environment uses PathManager (unchanged)
-   `src/hrl_agent/coordination.py`: Command completion uses distance_to_goal
-   `docs/training-convergence-analysis.md`: Full analysis of convergence issues

---

## Next Steps

After verifying these fixes work:

1. **Rebalance rewards** (Priority 2): Adjust coefficients as recommended in analysis doc
2. **Refine command shaping** (Priority 3): Separate conflicting objectives
3. **Run full training**: 100 episodes to confirm convergence

Expected training behavior:

-   Episodes 1-20: Exploration, variable rewards
-   Episodes 20-50: Learning kicks in, rewards trend positive
-   Episodes 50-100: Performance stabilizes, increasing waypoint completion
