# Fix: Waypoint Progress Plot Not Showing

## Problem

The episode summary plot was still showing "Closest to Goal" instead of "Waypoint Progress" even with waypoint-based navigation.

## Root Cause

**Detection Logic Error:**

```python
# WRONG: This fails when agent is at waypoint index 0!
has_waypoints = any(e.max_waypoints_reached > 0 for e in episode_metrics_snapshot)
```

**Why This Failed:**

-   Waypoint indices are 0-based (first waypoint = index 0)
-   `max_waypoints_reached` was initialized to 0
-   Agent at first waypoint → `max_waypoints_reached = 0`
-   Check `> 0` returns False
-   Plot defaults to "Closest to Goal" (legacy mode)

**Example:**

```python
Episode 1: Agent reaches waypoint 0 → max_waypoints_reached = 0
Episode 2: Agent reaches waypoint 0 → max_waypoints_reached = 0
Check: any([0 > 0, 0 > 0]) = False → Show "Closest to Goal" ❌
```

## Solution

### 1. Use Sentinel Value (-1)

Changed default to indicate "no waypoint data":

```python
# NEW: -1 means no waypoint data, 0+ means waypoint tracking active
max_waypoints_reached: int = -1
```

### 2. Fix Detection Logic

```python
# CORRECT: Check if waypoint data exists (>= 0)
has_waypoints = any(e.max_waypoints_reached >= 0 for e in episode_metrics_snapshot)
```

**Now:**

```python
Episode 1: Agent at waypoint 0 → max_waypoints_reached = 0
Episode 2: Agent at waypoint 0 → max_waypoints_reached = 0
Check: any([0 >= 0, 0 >= 0]) = True → Show "Waypoint Progress" ✅
```

### 3. Update Calculation Logic

```python
# Track max waypoint index reached
waypoint_indices = [
    s.current_waypoint_index
    for s in self.current_episode_steps
    if s.current_waypoint_index is not None
]
max_waypoint_reached = max(waypoint_indices) if waypoint_indices else -1  # -1 = no data
```

## Result

**Episode Summary Plot (4th subplot):**

-   **With waypoint data** (`max_waypoints_reached >= 0`): Shows "Waypoint Progress"
-   **Without waypoint data** (`max_waypoints_reached == -1`): Shows "Closest to Goal"

**Visualization:**

-   Purple bars showing waypoints reached (0, 1, 2, ...)
-   Y-axis: Number of waypoints (0-indexed, displayed as 1-indexed for humans)
-   Clear progress indicator for circuit/linear paths

## Edge Cases Handled

### Case 1: Agent Never Moves

-   No waypoint data captured
-   `max_waypoints_reached = -1`
-   Falls back to "Closest to Goal"

### Case 2: Agent Reaches First Waypoint Only

-   `max_waypoints_reached = 0`
-   Detection: `0 >= 0` = True ✅
-   Shows "Waypoint Progress" with value 1 (human-readable)

### Case 3: Agent Progresses Through Multiple Waypoints

-   `max_waypoints_reached = 0, 2, 5, ...`
-   Detection: All `>= 0` ✅
-   Shows increasing waypoint progress

### Case 4: Legacy Config (No Waypoints)

-   No `current_waypoint_index` in telemetry
-   `waypoint_indices = []`
-   `max_waypoints_reached = -1`
-   Falls back to "Closest to Goal" ✅

## Testing

✅ All 152 tests pass
✅ Detection logic fixed for 0-indexed waypoints
✅ Sentinel value (-1) properly distinguishes "no data" from "at first waypoint"
✅ Backward compatible with legacy configs

## Files Changed

-   `src/utils/metrics_tracker.py`:
    -   Changed `max_waypoints_reached` default from `0` to `-1`
    -   Updated calculation to use `-1` for missing data
    -   Fixed detection logic from `> 0` to `>= 0`

## Next Steps

Run training again:

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 10
```

The 4th subplot should now show **"Waypoint Progress"** with purple bars! 🎉
