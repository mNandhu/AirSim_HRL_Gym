# Waypoint-Based Navigation System

## Problem Statement

After over 100 episodes of training, we diagnosed a fundamental architectural issue preventing the agent from learning effectively:

### The Reward Conflict

The agent was receiving **contradictory reward signals**:

1. **`command_shaping` reward**: Encouraged following the road (FOLLOW_LANE command)
2. **`progress_velocity` reward**: Rewarded moving towards a distant final goal in a straight line

When the road curved away from the goal, the agent was **punished for following the road** while simultaneously being told to follow it. This created an unsolvable learning problem where no stable policy could emerge.

**Visual Evidence**: Trajectory plots from Episode 110 clearly showed the agent attempting to cut across terrain to reach the goal rather than following the curved road, despite the FOLLOW_LANE command being active.

## Solution: Path-Following Architecture

We refactored the system from **goal-reaching** to **path-following** navigation:

### Core Components

#### 1. PathManager (`src/utils/path_manager.py`)

A new utility class that manages waypoint-based navigation:

-   **Ordered waypoints**: Defines the correct route the agent should follow
-   **Current target tracking**: Returns the next waypoint the agent should move towards
-   **Automatic advancement**: When a waypoint is reached (within threshold), automatically advances to the next
-   **Progress tracking**: Monitors overall path completion

```python
# Example usage
waypoints = [(0, 0), (10, 0), (10, 10), (20, 10)]
path_manager = PathManager(waypoints, waypoint_threshold=5.0)

# During episode
current_position = (3.0, 1.0)
target_waypoint = path_manager.current_waypoint
vector_to_target = path_manager.get_vector_to_current_waypoint(current_position)
distance = path_manager.get_distance_to_current_waypoint(current_position)

# Automatically advance when close
waypoint_reached = path_manager.update(current_position)
```

#### 2. Updated ExperimentDefinition Schema

**New field**: `waypoints` (list of poses defining the path)
**Legacy field**: `goal_pose` (single goal, deprecated but still supported)

```yaml
# New waypoint-based config
waypoints:
    - x: 0.0
      y: 0.0
      z: -2.0
    - x: 10.0
      y: 0.0
      z: -2.0
    - x: 20.0
      y: 10.0
      z: -2.0
```

#### 3. AirSimEnv Integration

The environment now:

-   Creates a `PathManager` from experiment config
-   Resets the path manager on episode reset
-   Updates the path manager each step with current position
-   Passes `vector_to_next_waypoint` to reward calculator (points to **current path waypoint**, not distant final goal)

**Critical change in `_vehicle_state_from_telemetry`:**

```python
# OLD (caused conflict):
vector_to_waypoint = compute_vector_to_final_goal(position, final_goal)

# NEW (aligned rewards):
vector_to_waypoint = path_manager.get_vector_to_current_waypoint(position)
```

#### 4. Termination Condition Update

**Old**: Episode terminates when `distance_to_goal < threshold`
**New**: Episode terminates when `path_manager.all_waypoints_reached`

This allows the agent to complete multi-waypoint paths naturally.

## Impact on Reward Signals

### Before (Conflicting)

```
Position: On curved road
Road direction: North
Final goal: East (distant)

vector_to_next_waypoint: Points East (off road)
progress_velocity reward: High (moving towards goal)
command_shaping (FOLLOW_LANE): Low (not following road)

→ Agent punished for following road correctly!
```

### After (Aligned)

```
Position: On curved road
Road direction: North
Next waypoint: North (10m ahead on road)

vector_to_next_waypoint: Points North (along road)
progress_velocity reward: High (moving towards waypoint)
command_shaping (FOLLOW_LANE): High (following road)

→ Agent rewarded for correct road-following behavior!
```

## Configuration Guide

### New Waypoint-Based Config (Recommended)

```yaml
scene: Neighbourhood
vehicle: DefaultSedan

start_pose:
    x: 0.0
    y: 0.0
    z: -2.0
    yaw: 0.0

waypoints:
    - x: 10.0
      y: 0.0
      z: -2.0
    - x: 20.0
      y: 5.0
      z: -2.0
    - x: 30.0
      y: 15.0
      z: -2.0

horizon: 1000
seeds:
    python: 123
    numpy: 124
    torch: 125
    airsim: 126
```

### Legacy Config (Backward Compatible)

```yaml
scene: Neighbourhood
vehicle: DefaultSedan

start_pose:
    x: 0.0
    y: 0.0
    z: -2.0
    yaw: 0.0

goal_pose:
    x: 50.0
    y: 0.0
    z: -2.0
    yaw: 0.0

horizon: 500
```

**Note**: Legacy configs create a simple 2-waypoint path: [start_pose, goal_pose]

## Testing

### Unit Tests

-   `tests/unit/test_path_manager.py`: Comprehensive tests for PathManager
    -   Waypoint creation and conversion
    -   Distance and vector calculations
    -   Automatic waypoint advancement
    -   Progress tracking
    -   Edge cases (single waypoint, all reached, etc.)

### Integration Tests

-   Updated `tests/integration/test_termination_conditions.py`
    -   Tests waypoint-based termination
    -   Verifies backward compatibility with goal_pose

### Test Coverage

All tests pass (151 passed, 0 failed) including:

-   24 PathManager unit tests
-   2 updated integration tests
-   All existing environment/reward tests

## Migration Guide

### For New Training Runs

Use `configs/experiments/training_waypoints.yaml` which includes a multi-waypoint curved path.

### For Existing Configs

Legacy configs with `goal_pose` continue to work but create a straight-line path. To leverage the full benefit of aligned rewards:

1. Identify the route in your scene
2. Define waypoints along that route (spacing: 10-20m recommended)
3. Replace `goal_pose:` with `waypoints:`
4. Test with a short episode to verify path coverage

## Expected Learning Improvements

With aligned reward signals, we expect:

1. **Stable policy convergence**: No more contradictory gradients
2. **Better road-following**: Agent rewarded for correct behavior
3. **Natural curve handling**: No incentive to cut corners
4. **Improved sample efficiency**: Clearer learning signal

## Files Changed

-   `src/utils/path_manager.py` (new): PathManager implementation
-   `src/config/experiment.py`: Added waypoints field, made goal_pose optional
-   `src/airsim_env/env.py`: Integrated PathManager, updated reward calculation
-   `src/scripts/run_experiment.py`: Support waypoints in simulator adapter
-   `configs/experiments/training_waypoints.yaml` (new): Example waypoint config
-   `tests/unit/test_path_manager.py` (new): PathManager unit tests
-   `tests/integration/test_termination_conditions.py`: Updated for waypoint-based termination

## References

-   **Original Issue**: Contradictory reward signals observed in Episode 110 trajectory analysis
-   **Reward Contract**: `docs/reward-contract.md` (ensure `progress_velocity` documentation matches new behavior)
-   **Architecture**: Path-following now fundamental to the environment design
