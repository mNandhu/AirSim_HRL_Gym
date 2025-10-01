# Training Convergence Analysis - Non-Convergence Root Cause

**Training Run**: `20251001T090519Z_train_af3bbd74-f2f3-4eb2-9429-cea0f004710e`  
**Date**: October 1, 2025  
**Episodes**: 100  
**Configuration**: `training_waypoints.yaml` (8 waypoints, waypoint-based navigation)

## Executive Summary

The training exhibits severe non-convergence with predominantly negative rewards (see `episode_summary.png`). Analysis reveals **one critical bug** that breaks the reward system and **three design issues** that make positive rewards too sparse.

### Key Findings

1. 🔴 **CRITICAL BUG**: Waypoint progress tracking is completely broken - distance_to_goal stuck at 999.0
2. ⚠️ **Completion bonuses never awarded** - 100-episode training had ZERO completion bonuses due to bug #1
3. ⚠️ **Time penalty too aggressive** - constant -0.02 drain per step (~-2.0 to -5.0 per episode)
4. ⚠️ **Reward imbalance** - penalties dominate over progress rewards, especially at low speeds

---

## Critical Bug #1: Distance to Goal Always 999.0

### Location

`src/scripts/train_and_eval.py:179` in `AirSimSimulatorAdapter._build_state_dict()`

### The Bug

```python
# Calculate distance to goal (if experiment provided)
distance_to_goal = 999.0
goal_pose = getattr(experiment, "goal_pose", None) or self._goal_pose
position = car_state.kinematics_estimated.position
pos_xy = (float(position.x_val), float(position.y_val))
goal_xy: tuple[float, float] | None = None
if goal_pose is not None:
    goal_xy = (float(goal_pose.x), float(goal_pose.y))
    distance_to_goal = math.sqrt(
        (position.x_val - goal_pose.x) ** 2 + (position.y_val - goal_pose.y) ** 2
    )
```

**Problem**: When using waypoint-based navigation (`training_waypoints.yaml`), there is NO `goal_pose` - only a list of `waypoints`. The code defaults to `distance_to_goal = 999.0` and never updates it.

### Evidence from Metrics

From `artifacts/.../metrics/episodes.json`:

```json
{
  "episode": 1,
  "min_distance_to_goal": 999.0,  // ← Never changes!
  ...
}
```

All 100 episodes show `min_distance_to_goal: 999.0`, confirming the value is never updated.

### Cascading Impact

#### 1. Command Completion Detection Broken

In `src/hrl_agent/coordination.py:166-169`:

```python
elif command == "FOLLOW_LANE":
    # Completion relative to the starting distance of this command instance
    if start_distance is not None and distance is not None:
        completed = (start_distance - distance) >= 5.0
```

-   `start_distance` = 999.0 (captured when command starts)
-   `distance` = 999.0 (never changes)
-   `start_distance - distance` = 0.0 (always!)
-   **FOLLOW_LANE commands NEVER complete**

#### 2. No Completion Bonuses Ever Awarded

The `completion_bonus` (+100.0) is the largest positive reward component, but it's **never triggered** across all 100 episodes because FOLLOW_LANE (the primary command) can never complete.

#### 3. Metrics Show No Progress

The metric `min_distance_to_goal` is useless for tracking learning progress, making it impossible to detect whether the agent is improving at navigation.

### Why the Environment Fix Doesn't Help

The environment layer (`src/airsim_env/env.py:283-285`) DOES calculate correct waypoint distances:

```python
if current_position:
    distance_to_goal = self._path_manager.get_distance_to_current_waypoint(current_position)
```

**BUT** this happens in `_vehicle_state_from_telemetry()`, which:

1. Only affects reward calculation (VehicleState)
2. Does NOT update the telemetry dict that coordination sees
3. Coordination reads the original telemetry with `distance_to_goal: 999.0`

So rewards use correct distances, but **command completion still uses 999.0**.

---

## Issue #2: Reward Imbalance - Penalties Dominate

### Current Reward Structure

| Component                | Value/Range  | Frequency      | Total Impact (typical)   |
| ------------------------ | ------------ | -------------- | ------------------------ |
| **Positive Rewards**     |
| `completion_bonus`       | +100.0       | Never (bug #1) | **0.0**                  |
| `progress_velocity`      | 0 to ~7.2    | Every step     | +2.0 to +6.0 per step    |
| `heading_alignment`      | -1.0 to +1.0 | Turn commands  | ±0.5 per step            |
| **Negative Penalties**   |
| `time_penalty`           | -0.02        | **Every step** | -2.0 to -5.0 per episode |
| `lane_deviation_penalty` | 0 to -8.0+   | Every step     | -1.0 to -3.0 per step    |
| `idle_penalty`           | -0.5         | When stopped   | -1.0 to -5.0 per episode |
| `collision_penalty`      | -200.0       | On collision   | -200.0 (episode killer)  |

### The Math Doesn't Add Up

**Best case scenario** (perfect driving at moderate speed):

```
progress_velocity:     +5.0  (speed 4.2 m/s * alignment 1.0 * coef 1.2)
lane_deviation:        -0.1  (very small deviation 0.22² * 2.0)
time_penalty:          -0.02
─────────────────────────────
Net per step:          +4.88 ✓ Good
```

**Realistic scenario** (decent driving, some deviation):

```
progress_velocity:     +3.6  (speed 3.0 m/s * alignment 1.0 * coef 1.2)
lane_deviation:        -0.5  (moderate deviation 0.5² * 2.0)
time_penalty:          -0.02
─────────────────────────────
Net per step:          +3.08 ✓ Okay
```

**Common scenario** (lower speed, larger deviation):

```
progress_velocity:     +2.4  (speed 2.0 m/s * alignment 1.0 * coef 1.2)
lane_deviation:        -2.0  (deviation 1.0² * 2.0)
time_penalty:          -0.02
─────────────────────────────
Net per step:          +0.38 ⚠️ Barely positive
```

**Poor scenario** (slow or misaligned):

```
progress_velocity:     +0.96 (speed 2.0 m/s * alignment 0.4 * coef 1.2)
lane_deviation:        -2.0  (deviation 1.0² * 2.0)
time_penalty:          -0.02
─────────────────────────────
Net per step:          -1.06 ❌ NEGATIVE
```

### Problem Analysis

1. **Time penalty accumulates**: Over 100-200 steps, `-0.02 * 150 = -3.0` total
2. **Lane deviation is quadratic**: Small errors (0.5m) → -0.5, but 1.0m → -2.0 (4x worse)
3. **Progress reward is linear**: Doubling deviation from 0.5 to 1.0 quadruples penalty, but doubling speed only doubles reward
4. **No completion bonuses**: Without the +100.0 bonuses, there's no large positive signal to offset accumulated penalties

---

## Issue #3: Command-Specific Reward Conflicts

### FOLLOW_LANE Command

Current shaping: `progress_velocity + lane_deviation_penalty`

**Conflict**: Both rewards pull in different directions:

-   Progress wants: "Move forward fast toward next waypoint"
-   Lane deviation wants: "Stay perfectly centered on the lane"

When the road curves or the waypoint is off to the side, these conflict. The agent might need to deviate from lane center to approach the waypoint, but gets punished for doing so.

**Evidence**: Episodes 20-100 show highly variable rewards, suggesting the agent can't find a stable policy.

### Turn Commands

Current shaping: `progress_velocity + heading_alignment`

**Better alignment**: These work together more naturally - both want the agent to turn toward the waypoint. This explains why turn completions might be slightly more consistent (though still no bonuses due to bug #1).

---

## Issue #4: Sparse High-Value Rewards

### The Completion Bonus Problem

The reward design relies heavily on sparse +100.0 completion bonuses to offset accumulated penalties. But:

1. **Bug #1 prevents them** - FOLLOW_LANE never completes
2. **Even without bug**: Bonuses are sparse by nature

    - Need to drive ~5 meters to complete FOLLOW_LANE
    - Takes ~20-30 steps at moderate speed
    - Dense penalties accumulate: 25 steps \* -0.02 = -0.5 time + ~-10.0 lane deviations
    - Need +100 bonus to make up for 30 steps of accumulated negatives

3. **Learning problem**: Sparse rewards make credit assignment hard
    - Agent doesn't know which actions led to completion
    - Dense penalties give immediate feedback for every mistake
    - But dense rewards for good behavior are weak

---

## Observed Training Behavior

### From episode_summary.png

**Episodes 1-10**: Slightly positive rewards (mostly +5 to +50)

-   Agent exploring, occasionally gets lucky with good alignment
-   Some episodes reach moderate speeds (max 6-7 m/s)

**Episodes 10-20**: Sharp decline to negative

-   Agent starts learning that movement → penalties
-   Possibly discovering that staying still avoids lane deviation penalties
-   Rewards drop to -50 to -100 range

**Episodes 20-100**: Consistently negative, high variance

-   Stuck in local minimum of poor behavior
-   Can't escape because:
    -   No completion bonuses to guide toward better regions
    -   Dense penalties immediately punish exploration
    -   Time penalty punishes both action and inaction

**Waypoint Progress**: All episodes show `waypoints_reached: 1/8`

-   Agent barely moves or doesn't reach even the first waypoint
-   Confirms navigation is not working

---

## Root Cause Summary

### Critical

1. **Distance tracking broken** → No completion bonuses → No large positive rewards → Can't offset penalties

### Design Issues

2. **Time penalty too aggressive** → Constant drain makes net rewards negative
3. **Lane deviation quadratic** → Disproportionately punishes moderate errors
4. **Progress coefficient too low** → Good behavior (moving forward) not rewarded enough
5. **Conflicting objectives** → FOLLOW_LANE command has competing goals

---

## Recommended Fixes

> **✅ UPDATE**: Priority 1 fix has been implemented.  
> See `docs/fix-distance-tracking-and-metrics.md` for details.

### Priority 1: Fix Distance Tracking (CRITICAL) ✅ COMPLETED

**File**: `src/scripts/train_and_eval.py:179`

Add support for waypoint-based distance calculation:

```python
def _build_state_dict(self, car_state, experiment=None) -> dict[str, Any]:
    # ... existing code ...

    # Calculate distance to goal/waypoint
    distance_to_goal = 999.0
    goal_xy: tuple[float, float] | None = None

    # NEW: Check for waypoint-based navigation first
    if hasattr(experiment, 'waypoints') and experiment.waypoints:
        # Use current waypoint from path_manager if available
        # For now, use first waypoint as fallback
        first_waypoint = experiment.waypoints[0]
        goal_xy = (float(first_waypoint.x), float(first_waypoint.y))
        distance_to_goal = math.sqrt(
            (position.x_val - first_waypoint.x) ** 2 +
            (position.y_val - first_waypoint.y) ** 2
        )
    elif hasattr(experiment, 'goal_pose') and experiment.goal_pose is not None:
        # Legacy single-goal mode
        goal_pose = experiment.goal_pose
        goal_xy = (float(goal_pose.x), float(goal_pose.y))
        distance_to_goal = math.sqrt(
            (position.x_val - goal_pose.x) ** 2 +
            (position.y_val - goal_pose.y) ** 2
        )
```

**Better fix**: Pass `path_manager` to simulator adapter so it can query current waypoint dynamically.

### Priority 2: Rebalance Rewards

**File**: `src/airsim_env/reward.py`

Recommended changes:

```python
@dataclass(frozen=True)
class RewardConfig:
    # Increase progress reward to dominate over penalties
    progress_velocity_coef: float = 2.5  # Was 1.2 → +108% increase

    # Reduce lane deviation penalty (make it less harsh)
    lane_deviation_penalty_coef: float = 1.0  # Was 2.0 → -50% reduction

    # Reduce or remove time penalty
    time_penalty: float = -0.005  # Was -0.02 → -75% reduction

    # Keep completion bonus high
    completion_bonus: float = 100.0  # Unchanged (but now will work!)
```

**Rationale**:

-   Higher progress coefficient: Good behavior (moving forward) gets ~2x reward
-   Lower lane deviation: Tolerates necessary deviations for path-following
-   Lower time penalty: Less constant drain, allows exploration
-   Net effect: Good driving gives +5-15 per step vs -0.5 to -2.0 for errors

### Priority 3: Better Command Shaping

**File**: `src/airsim_env/reward.py:99-102`

For FOLLOW_LANE, remove lane deviation penalty:

```python
if active_command == "FOLLOW_LANE":
    # Only reward progress, don't penalize deviation
    # (Path might require deviation from lane center)
    command_shaping_reward = progress
elif "TURN" in active_command:
    command_shaping_reward = heading_alignment + progress
```

Add lane deviation as a separate, always-active penalty with reduced weight:

```python
# Apply lane deviation penalty universally but gently
lane_penalty = self._lane_deviation_penalty(current_state) * 0.5  # Half weight

components = {
    "command_shaping": command_shaping_reward,
    "lane_centering": lane_penalty,  # Separate component
    "collision_penalty": collision,
    "completion_bonus": completion_bonus,
    "idle_penalty": idle_penalty,
    "time_penalty": self._config.time_penalty,
}
```

---

## Testing Plan

After fixes:

1. **Verify distance tracking**:

    ```bash
    # Check that min_distance_to_goal decreases over time
    uv run python src/scripts/train_and_eval.py train --config configs/experiments/training_waypoints.yaml --episodes 5
    # Inspect: artifacts/.../metrics/episodes.json
    # Expected: min_distance_to_goal should be < 999.0 and vary
    ```

2. **Verify completion bonuses**:

    ```bash
    # Check episode logs for "completion_bonus: 100.0" entries
    # Expected: Should see bonuses when commands complete
    ```

3. **Verify reward balance**:

    ```bash
    # Train for 20 episodes, check episode_summary.png
    # Expected: Rewards should trend toward positive as agent learns
    ```

4. **Verify waypoint progress**:
    ```bash
    # Check metrics for "waypoints_reached: N/8"
    # Expected: Should increase over episodes (2, 3, 4, ... waypoints)
    ```

---

## Additional Notes

### Why Episode 1 Was Positive

Looking at episode 1 data: `cumulative_reward: 46.47`

This likely happened because:

1. Random initialization placed the agent in a favorable orientation
2. Initial movement happened to align well with waypoint
3. High `progress_velocity` rewards in early steps before deviation accumulated
4. Once the agent turned or deviated, penalties took over

### Waypoint Configuration Quality

The 8-waypoint path in `training_waypoints.yaml` forms a circuit:

-   Start: (0, 0) → End: (0, 6.2)
-   Total path length: ~400+ meters
-   This is a GOOD test - long enough to be challenging, but finite

The non-convergence is NOT due to poor waypoint design; it's due to the bugs and reward imbalance.

---

## Conclusion

The training fails to converge due to:

1. **Critical bug**: Distance tracking returns 999.0 → no completion bonuses
2. **Reward imbalance**: Penalties too strong, progress rewards too weak
3. **Missing large positive signals**: Completion bonuses never awarded

**Fix priority order**:

1. Fix distance tracking (enables completion bonuses) ← MUST DO
2. Rebalance reward coefficients (makes good driving worthwhile) ← SHOULD DO
3. Refine command shaping (resolves conflicts) ← NICE TO HAVE

With these fixes, the agent should start learning meaningful navigation behavior within 20-50 episodes.
