# Distance Metrics for Waypoint Navigation

## Summary

With waypoint-based navigation, we need **two separate distance metrics** to avoid confusion:

1. **`distance_to_goal`** - Distance to the **final goal** (last waypoint)

    - Never changes meaning
    - Provides overall progress view
    - Used for "Distance to Goal" graph

2. **`distance_to_current_waypoint`** - Distance to the **current target waypoint**
    - Shows step-by-step progress
    - Decreases as agent approaches each waypoint
    - Used for "Waypoint Progress" graph

## Why Two Metrics?

### The Problem with Single Metric

Initially, we changed `distance_to_goal` to reflect the current waypoint distance. This caused a **misleading visualization**:

```
Start: distance_to_goal = 10m (to waypoint 1)
Approaches waypoint 1: 5m, 3m, 1m
Reaches waypoint 1: Suddenly jumps to 15m (to waypoint 2) ⚠️
```

This makes it look like the agent is **moving away** from the goal when it's actually making progress!

### The Solution: Separate Metrics

**`distance_to_goal`** (Final Goal):

-   Start: 50m (distance to final waypoint at 50,25)
-   Decreases monotonically as agent progresses
-   Clear indicator of overall completion

**`distance_to_current_waypoint`** (Current Target):

-   Start: 10m (to first waypoint at 10,0)
-   Approaches: 5m, 3m, 1m
-   Reaches waypoint: Jumps to ~10m (next waypoint)
-   Shows path-following behavior

## Implementation

### Telemetry Fields Added

`AirSimEnv` now adds three new fields to telemetry:

```python
telemetry["distance_to_current_waypoint"] = path_manager.get_distance_to_current_waypoint(position)
telemetry["current_waypoint_index"] = path_manager.current_waypoint_index
telemetry["total_waypoints"] = path_manager.total_waypoints
```

### Metrics Tracker Updates

**StepMetrics dataclass**:

```python
distance_to_goal: float = 0.0  # Distance to final goal (unchanged)
distance_to_current_waypoint: Optional[float] = None  # New: distance to current waypoint
current_waypoint_index: Optional[int] = None  # New: which waypoint we're targeting
total_waypoints: Optional[int] = None  # New: total waypoint count
```

### Performance Metrics Plot

The performance metrics plot now has **4 subplots** when waypoint data is available:

1. **Vehicle Speed** - Speed over time
2. **Distance to Goal** - Distance to final goal (smooth decrease)
3. **Waypoint Progress** - Distance to current waypoint (step-like pattern with vertical lines showing waypoint transitions)
4. **Reward Component Breakdown** - Stacked reward components

## Visual Interpretation

### Distance to Goal Graph

-   **Smooth downward trend** as agent progresses through waypoints
-   Reflects overall mission progress
-   Never confusing or jumping around

### Waypoint Progress Graph

-   **Sawtooth pattern**: decreases as approaching waypoint, jumps up when advancing to next
-   **Vertical green lines**: mark waypoint transitions
-   **WP labels**: show which waypoint index was reached
-   Clearly shows path-following behavior

## Backward Compatibility

**Legacy configs** (goal_pose only):

-   `distance_to_goal`: Distance to goal (as before)
-   `distance_to_current_waypoint`: Same as distance_to_goal (2-waypoint path)
-   No visual confusion since there's only one target

**New configs** (waypoints):

-   Both metrics available
-   4-subplot performance metrics plot
-   Clear separation of concerns

## Reward Calculation

**VehicleState still uses current waypoint distance** for reward calculation:

-   `progress_velocity` reward based on movement toward current waypoint
-   Aligns incentives with path-following behavior
-   No change from waypoint navigation implementation

## Testing

✅ **All 152 tests pass**
✅ **Test verifies both metrics work correctly**
✅ **Distance to goal remains constant (final goal)**
✅ **Distance to current waypoint decreases with progress**

## Files Changed

-   `src/airsim_env/env.py`: Adds `distance_to_current_waypoint`, `current_waypoint_index`, `total_waypoints` to telemetry
-   `src/utils/metrics_tracker.py`:
    -   Updated `StepMetrics` dataclass with waypoint fields
    -   Enhanced `_plot_performance_metrics` to show waypoint progress
    -   Automatically detects waypoint data and adds 4th subplot
-   `tests/integration/test_distance_to_goal_fix.py`: Updated to verify both metrics
-   `docs/distance-to-goal-fix.md`: This documentation

## Usage

No code changes needed! The system automatically:

-   Detects waypoint-based navigation
-   Adds waypoint metrics to telemetry
-   Shows appropriate plots based on available data

Your next training run will show:

-   **Distance to Goal**: Smooth progress toward final destination
-   **Waypoint Progress**: Step-by-step advancement through the path
