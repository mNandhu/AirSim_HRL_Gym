# Episode Termination Analysis: Why Episodes End Without Reaching Final Goal

**Date**: October 5, 2025  
**Question**: If collision rate is 9.6%, what causes the other 90.4% of episodes to terminate?

---

## Answer: Truncation at Horizon Limit

### Episode Termination Types

In the 198-episode training run, there are **TWO ways episodes can end**:

1. **Terminated** (`terminated=True`): Episode ends due to a terminal condition
2. **Truncated** (`truncated=True`): Episode ends due to time/step limit

---

## The Termination Logic

From `src/airsim_env/env.py`:

```python
# Check if episode should end
terminated = self._check_terminated(telemetry)
truncated = self._check_truncated()

done_flags = {
    "terminated": terminated,
    "truncated": truncated,
    "collision": bool(telemetry.get("collision", False)),
    "goal_reached": terminated and not telemetry.get("collision", False),
}
```

### `_check_terminated()`: Natural Episode Endings

```python
def _check_terminated(self, telemetry: Mapping[str, Any]) -> bool:
    # 1. Collision
    if telemetry.get("collision", False):
        return True

    # 2. Off-road (< 20% lane coverage)
    lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
    if lane_ratio < 0.2:
        return True

    # 3. All waypoints reached
    return self._path_manager.all_waypoints_reached
```

**Returns `True` when**:

-   ❌ Vehicle collides with obstacle
-   ❌ Vehicle goes severely off-road (<20% lane coverage)
-   ✅ Vehicle reaches **ALL 8 waypoints**

### `_check_truncated()`: Time Limit

```python
def _check_truncated(self) -> bool:
    return self._step_index + 1 >= self._experiment.horizon
```

**Returns `True` when**:

-   ⏰ Episode reaches 1000 steps (horizon limit)

---

## What "goal_reached=True" Actually Means

### The Confusing Logic

```python
"goal_reached": terminated and not telemetry.get("collision", False)
```

This means:

-   `goal_reached = True` when episode **terminates** without collision
-   This includes BOTH:
    -   ✅ **Reaching all 8 waypoints** (success!)
    -   ⏰ **Hitting time limit** (truncation, not actually success!)

**This is misleading!**

### Example: Episode 190 (marked as "goal_reached=True")

```json
{
  "steps": 127,
  "position": [51.58, -2.14],
  "goal": [117.1, -1.9],
  "distance_to_goal": 65.5 meters,
  "collision": false,
  "goal_reached": true  // ← But 65m away from next waypoint!
}
```

**What happened**:

-   Agent reached first waypoint (within 5m threshold)
-   PathManager advanced to waypoint 2
-   Episode continued toward waypoint 2
-   Got to 65m away (still pretty far)
-   Episode ended at step 127
-   Why? Probably **truncated at horizon or some other condition**

---

## Episode Termination Breakdown

### Categories

From the 198 episodes:

| Condition                            | Count | Percentage | What It Means            |
| ------------------------------------ | ----- | ---------- | ------------------------ |
| `collision=True, goal_reached=False` | 19    | 9.6%       | Crashed into something   |
| `collision=False, goal_reached=True` | 179   | 90.4%      | **Terminated "cleanly"** |

### What "goal_reached=True" Actually Represents

The 179 "successful" episodes likely fall into these categories:

**A. Truncated at Horizon** (Most Common)

-   Episode hits 1000-step limit
-   Agent is somewhere on the path
-   Episode ends with `truncated=True, goal_reached=True`
-   **NOT truly successful** - didn't reach all waypoints

**B. Off-Road Termination** (If lane segmentation worked)

-   Lane coverage drops below 20%
-   Episode ends with `terminated=True, goal_reached=True`
-   **NOT truly successful** - went off-road

**C. Actually Reached All Waypoints** (Rare/Never)

-   `all_waypoints_reached=True`
-   Episode ends with `terminated=True, goal_reached=True`
-   **Truly successful** - completed the path

---

## Why Episodes Are Short (96 Steps Average)

### Expected vs Actual

**If reaching all waypoints**:

-   Path length: ~450 meters (8 waypoints around loop)
-   At 5 m/s: ~90 seconds = 360 steps (at 0.25s per step)
-   **Expected: 360+ steps**

**Actual**:

-   Average: 96 steps
-   Max: 172 steps
-   **This is way too short!**

### Possible Causes

1. **Early Termination Due to Off-Road** (Most Likely)

    - Lane coverage drops < 0.2
    - Episode ends after ~50-100 steps
    - Logged as "goal_reached=True" (misleading!)
    - **But lane data is NULL**, so this shouldn't trigger...

2. **Some Other Termination Condition** (Unknown)

    - Maybe another check we haven't found?
    - Or a crash/exception being caught?

3. **Agent Not Progressing** (Unlikely)
    - Agent stops moving
    - Idle penalty accumulates
    - But no explicit idle termination condition

---

## The Mystery: Why Do Episodes End?

### What We Know

-   Episodes are SHORT (96 steps average, max 172)
-   Not hitting 1000-step horizon
-   Not all collisions (only 9.6%)
-   Labeled as "goal_reached=True" but agent nowhere near final waypoint

### What We Don't Know

**If lane_mask_coverage_ratio is NULL**:

-   Off-road termination shouldn't trigger (defaults to 1.0)
-   But episodes are still ending early
-   **Something else must be terminating them!**

### Hypothesis

Looking at the code more carefully:

```python
return self._path_manager.all_waypoints_reached
```

This returns `True` when:

```python
def all_waypoints_reached(self) -> bool:
    return self._current_index >= len(self._waypoints)
```

**Wait!** This triggers when `_current_index` moves PAST the last waypoint!

### How Waypoints Advance

```python
def update(self, current_position) -> bool:
    distance = self.get_distance_to_current_waypoint(current_position)

    if distance <= self._waypoint_threshold:  # 5 meters
        self._current_index += 1  # Advance to next waypoint
        return True
```

**Aha! The issue**:

1. Agent reaches waypoint 1 (within 5m)
2. PathManager increments to waypoint 2
3. Agent now targeting waypoint 2
4. **But the environment is checking `all_waypoints_reached`**
5. If somehow `_current_index` jumps to 8, episode ends!

**This shouldn't happen unless**:

-   Agent reaches ALL 8 waypoints (unlikely in 96 steps)
-   OR there's a bug in waypoint advancement

---

## The Real Answer: Likely Truncation

### Most Probable Scenario

The 179 "successful" episodes are probably **truncated** at the horizon, but the logging is confusing:

```python
"goal_reached": terminated and not collision
```

**Should be**:

```python
"goal_reached": self._path_manager.all_waypoints_reached and not collision
```

### Why Episodes Are Short

If not hitting 1000-step horizon, then:

1. **Off-road termination IS working** (even though lane data NULL)

    - Somehow the default 1.0 is being overridden
    - Or there's a separate off-road check

2. **Agent gets stuck/stops**

    - Speed drops below threshold
    - No explicit termination, but something stops the episode

3. **Exception/Error**
    - Silent failure catching exceptions
    - Episode ends prematurely

---

## Recommendation: Add Better Logging

To understand what's really happening, add explicit termination reason logging:

```python
def _check_terminated(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
    if telemetry.get("collision", False):
        return True, "collision"

    lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
    if lane_ratio < 0.2:
        return True, f"off_road (coverage={lane_ratio:.2f})"

    if self._path_manager.all_waypoints_reached:
        return True, "all_waypoints_reached"

    return False, "continuing"

# In step():
terminated, reason = self._check_terminated(telemetry)
print(f"Episode ending: {reason}")  # Log the reason!

done_flags = {
    "terminated": terminated,
    "truncated": truncated,
    "termination_reason": reason,  # Add this!
    # ...
}
```

This will reveal exactly why episodes are ending early.

---

## Summary

**Your Question**: If collision rate is 9.6%, what terminates the other 90.4%?

**Answer**: The 90.4% marked as "goal_reached=True" are **NOT actually reaching all waypoints**. They are most likely:

1. **Truncated at horizon** (hitting 1000-step limit) - Unlikely, episodes are too short
2. **Off-road termination** (< 20% lane coverage) - Likely, even though lane data shows NULL
3. **Unknown termination condition** - Needs investigation

**The Confusion**: `goal_reached` flag is misleading. It means "terminated without collision", not "reached all waypoints".

**To Fix**:

1. Add explicit termination reason logging
2. Change `goal_reached` to check `all_waypoints_reached` explicitly
3. Investigate why episodes are so short (96 steps average)

**The Real Issue**: Something is terminating episodes early (~96 steps), but we don't know what because logging only shows "goal_reached=True" which is ambiguous.
