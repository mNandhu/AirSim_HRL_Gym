# Waypoint Visualization in Trajectory Plots

## Update Summary

The trajectory plotting system has been enhanced to visualize the complete waypoint path alongside the vehicle's actual trajectory.

## What Changed

### Before

Trajectory plots showed:

-   Blue trajectory line (actual path taken)
-   Green start marker
-   Orange latest position marker
-   Red star for single goal (if using legacy `goal_pose`)

### After

Trajectory plots now show:

-   Blue trajectory line (actual path taken)
-   Green start marker
-   Orange latest position marker
-   **Red waypoints** - All waypoints along the intended path
-   **Red dashed line** - Connecting waypoints to show the intended route
-   **Red star** - Final goal waypoint (larger marker)

## Implementation Details

### MetricsTracker Updates (`src/utils/metrics_tracker.py`)

1. **Added waypoints storage**:

    ```python
    self.current_waypoints: List[Tuple[float, float]] = []
    ```

2. **Updated `start_episode` signature**:

    ```python
    def start_episode(self, episode: int, waypoints: Optional[List[Tuple[float, float]]] = None)
    ```

3. **Enhanced `_plot_trajectory_map`**:
    - Plots waypoints as red circles with dark red edges
    - Connects waypoints with a red dashed line showing the intended path
    - Marks the final waypoint with a larger star marker
    - Includes waypoints in extent calculation for proper plot sizing

### Training Integration (`src/scripts/train_and_eval.py`)

Added `_extract_waypoints_from_experiment()` helper function that:

-   Extracts waypoints from experiment config if using new waypoint-based navigation
-   Falls back to creating 2-waypoint path (start → goal) for legacy `goal_pose` configs
-   Passes waypoints to metrics tracker on episode start

### Orchestrator Support (`src/hrl_agent/orchestrator.py`)

Added `_extract_waypoints_from_env()` method for parallel execution scenarios to extract waypoints from any environment instance.

## Visual Result

Trajectory plots now clearly show:

1. **Intended path** (red dashed line connecting waypoints)
2. **Actual trajectory** (blue solid line)
3. **Waypoint targets** (red circles) - The sequential goals the agent should reach
4. **Final goal** (red star) - The ultimate destination

This makes it easy to see:

-   Whether the agent is following the intended path
-   How far the agent deviates from waypoints
-   If the agent is taking shortcuts or going off-course
-   Progress through the waypoint sequence

## Example Output

For a curved path with waypoints at:

-   (0, 0) - Start
-   (10, 0)
-   (20, 10)
-   (30, 20) - Goal

The plot will show all four waypoints connected by a dashed red line, with the vehicle's actual trajectory overlaid in blue. This immediately reveals whether the vehicle followed the intended curved path or attempted to take a straighter route.

## Backward Compatibility

-   **Legacy configs** with `goal_pose` automatically create a 2-waypoint path
-   Plots without waypoints fall back to showing just the single target marker
-   Existing training runs and plots remain unaffected

## Testing

All 151 tests pass, including integration with the updated trajectory plotting system.

## Usage

No changes needed to existing training commands. Waypoints are automatically extracted from your experiment config:

```bash
# With waypoint-based config
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100

# With legacy goal_pose config (shows 2-waypoint path)
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training.yaml \
  --episodes 100
```

Plots are saved to: `artifacts/{run_id}/metrics/trajectory.png`
