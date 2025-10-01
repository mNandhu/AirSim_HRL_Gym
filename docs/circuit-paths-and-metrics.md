# Circuit Paths and Metrics Improvements

## Issues Identified

### Issue 1: "Closest to Goal" Misleading for Circuits

**Problem:**

-   The "Closest to Goal" metric tracked minimum distance to the **final goal** waypoint
-   For circuit/loop paths (e.g., race track), this is meaningless
-   Example: Starting at (0,0), final waypoint at (0.4, 6.2) → shows ~1000m at far end of circuit
-   Agent appears to fail even when successfully following the path

**Why This Happened:**

-   "Distance to goal" means distance to final destination (last waypoint)
-   For circuits, you're usually far from the "finish line" while correctly following the track
-   Metric was designed for point-to-point navigation, not circuits

### Issue 2: Circuit Path Handling

**Problem:**

-   Current PathManager has no concept of "loop" or "circuit"
-   Last waypoint triggers episode termination
-   For circuits, last waypoint should lead back to first waypoint
-   Circuit paths need special handling

**Current Behavior:**

```yaml
waypoints:
  - x: 0.0, y: 0.0      # Start
  - x: 50.0, y: 0.0     # Waypoint 1
  - x: 50.0, y: 50.0    # Waypoint 2
  - x: 0.0, y: 50.0     # Waypoint 3
  - x: 0.4, y: 6.2      # Last waypoint (NOT back at start)
```

Problem: Episode ends at last waypoint, doesn't truly loop.

## Solutions Implemented

### 1. Better Episode Metrics

**Added `max_waypoints_reached`** to `EpisodeMetrics`:

```python
@dataclass
class EpisodeMetrics:
    ...
    max_waypoints_reached: int = 0  # Track furthest waypoint reached
```

**Benefits:**

-   Shows actual progress through the path
-   Meaningful for both point-to-point and circuit paths
-   Easy to interpret: "Reached waypoint 5 of 8"

### 2. Updated Episode Summary Plot

**Changed 4th subplot** from "Closest to Goal" to "Waypoint Progress":

-   **With waypoint data**: Shows number of waypoints reached per episode
-   **Without waypoint data** (legacy): Falls back to "Closest to Goal"

**Visual Result:**

-   Purple bars showing waypoint progress
-   Clear indication of how far agent progressed through path
-   Useful for both linear paths and circuits

### 3. Enhanced Episode Completion Message

**Before:**

```
📊 Episode 1 completed:
   Reward: 285.32 | Steps: 200 | Duration: 45.2s
   Max Speed: 6.5 m/s | Min Distance: 1000.0m
```

**After:**

```
📊 Episode 1 completed:
   Reward: 285.32 | Steps: 200 | Duration: 45.2s
   Max Speed: 6.5 m/s | Min Distance: 1000.0m | Waypoints: 5/8
```

Shows: "Reached 5 out of 8 waypoints" - much more meaningful!

## Current Limitations

### Circuit Paths Not Fully Supported

**What Works:**

-   You can create a path that loops back near the start
-   Agent will follow the path correctly
-   Metrics will show waypoint progress

**What Doesn't Work:**

-   Episode terminates when reaching last waypoint
-   No automatic loop back to first waypoint
-   Can't do true "infinite lap" training

**Workaround:**
Add the start position as the last waypoint:

```yaml
waypoints:
  - x: 0.0, y: 0.0      # Start / Finish
  - x: 50.0, y: 0.0
  - x: 50.0, y: 50.0
  - x: 0.0, y: 50.0
  - x: 0.0, y: 0.0      # Back to start (explicit loop closure)
```

This creates a proper circuit where reaching the last waypoint = completing one lap.

## Future Enhancements

### Proposed: Circuit Mode

Add explicit circuit/loop support to PathManager:

```python
class PathManager:
    def __init__(
        self,
        waypoints: Sequence[Waypoint],
        waypoint_threshold: float = 5.0,
        circuit_mode: bool = False  # NEW
    ):
        self._circuit_mode = circuit_mode
        ...

    def update(self, current_position) -> bool:
        """Update with circuit looping."""
        if distance <= threshold:
            self._current_index += 1

            # Circuit mode: loop back to start
            if self._circuit_mode and self._current_index >= len(self._waypoints):
                self._current_index = 0  # Loop!
                self._lap_count += 1
                return True

            return True
        return False
```

**Benefits:**

-   True multi-lap training
-   Lap counting
-   Episode terminates after N laps
-   Better for race track scenarios

### Proposed: Lap-based Metrics

For circuit mode:

-   Track laps completed
-   Best lap time
-   Lap-to-lap reward improvement
-   Sector times (if waypoints mark sectors)

## Recommendations

### For Linear Paths (Point-to-Point)

-   Use current waypoint-based navigation as-is
-   "Waypoints Reached" metric shows progress
-   Episode ends at final waypoint (as intended)

### For Circuit Paths (Current Implementation)

1. **Add start as last waypoint** to close the loop:

    ```yaml
    waypoints:
      - x: 0.0, y: 0.0  # Start
      - x: 10.0, y: 5.0
      - ...
      - x: 0.0, y: 0.0  # Finish (loop closure)
    ```

2. **Set appropriate horizon**:

    - One lap might be 500-1000 steps
    - Set horizon to allow completion
    - Monitor "Waypoints Reached" metric

3. **Interpret metrics correctly**:
    - "Distance to Goal" = distance to finish line (not useful mid-circuit)
    - "Waypoints Reached" = actual progress indicator
    - "Max Speed" and rewards still meaningful

### For True Multi-Lap Training (Future)

Wait for circuit_mode implementation, or:

-   Run multiple episodes (each = one lap attempt)
-   Aggregate metrics across episodes
-   Monitor waypoint progress to see lap completion rate

## Testing

✅ All 152 tests pass
✅ Episode summary now shows waypoint progress when available
✅ Console output includes waypoints reached
✅ Backward compatible with legacy single-goal configs

## Files Changed

-   `src/utils/metrics_tracker.py`:
    -   Added `max_waypoints_reached` to `EpisodeMetrics`
    -   Updated `_finish_episode` to track waypoint progress
    -   Modified `_plot_episode_summary` to show waypoint progress
    -   Enhanced console output with waypoint count

## Impact

**For Linear Paths:**

-   Better progress tracking
-   Clear indication of how far agent traveled
-   No change in behavior

**For Circuit Paths:**

-   "Closest to Goal" no longer misleading (replaced with waypoint progress)
-   Clear visualization of lap progress
-   Still requires loop closure in waypoint list

**Backward Compatibility:**

-   Legacy configs without waypoints: Shows "Closest to Goal" as before
-   No breaking changes
