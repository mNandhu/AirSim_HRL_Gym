# Reward System Fix for Single-Agent Training

**Date**: October 2, 2025  
**Issue**: Rewards too sparse, all episodes getting ~-50 (collision penalty only)  
**Status**: ✅ Fixed

---

## Problem Analysis

After examining training artifacts from `20251002T045035Z_train_sac_8714c381-2081-4f28-95c6-56d178efceb3`:

### Reward Breakdown (Episode 1, 58 steps):

```json
{
  "command_shaping": 0.0,           // ❌ NO positive rewards!
  "collision_penalty": -50.0,       // Only at collision
  "completion_bonus": 0.0,          // No waypoints reached
  "idle_penalty": 0.0,
  "time_penalty": -0.005 * 58 = -0.29
}
Total: -50.29
```

### Key Issues:

1. **`command_shaping` always 0.0**: Single-agent has no `active_command`, so progress/heading rewards never applied
2. **Collision penalty dominates**: -50 overwhelms any small per-step rewards
3. **No waypoint incentive**: Agent never rewarded for making progress toward waypoints
4. **All episodes end in collision**: No learning signal about good navigation

---

## Root Cause

The reward calculator was designed for hierarchical RL with command-specific shaping:

```python
# OLD CODE (broken for single-agent)
command_shaping_reward = 0.0
if active_command == "FOLLOW_LANE":
    command_shaping_reward = lane_deviation_penalty + progress
elif "TURN" in active_command:
    command_shaping_reward = heading_alignment + progress
# If active_command is "" or None: command_shaping = 0.0 ❌
```

In single-agent mode, `active_command` is always `""` or `None`, so progress rewards never applied!

---

## Solution

### 1. Reduced Collision Penalty (80% reduction)

```python
collision_penalty: float = 10.0  # Was 50.0
```

**Rationale**:

-   -50 per episode was dominating all other signals
-   Agent needs to see reward differences based on navigation quality
-   -10 is still significant but allows learning from per-step rewards

### 2. Added Waypoint Progress Bonus

```python
waypoint_progress_bonus: float = 50.0  # NEW
```

**Rationale**:

-   Provides strong positive signal for making progress
-   +50 for each waypoint offsets collision penalty
-   Encourages exploration toward waypoints

### 3. Fixed Command Shaping for Single-Agent

```python
# NEW CODE (works for single-agent)
if not active_command or active_command == "":
    # Single-agent mode: combine progress and heading alignment
    command_shaping_reward = progress + heading_alignment
elif active_command == "FOLLOW_LANE":
    command_shaping_reward = lane_deviation_penalty + progress
elif "TURN" in active_command:
    command_shaping_reward = heading_alignment + progress
```

**Rationale**:

-   Provides default navigation rewards when no command context
-   Rewards speed toward waypoint (`progress`)
-   Rewards heading alignment toward waypoint (`heading_alignment`)
-   Same formula as hierarchical TURN commands (makes sense for waypoint navigation)

---

## Expected Reward Patterns After Fix

### Before Collision (per step):

```
Speed: 5 m/s toward waypoint (alignment: 0.8)
progress = 5.0 * 0.8 * 2.0 = 8.0
heading_alignment = 0.8 * 1.0 = 0.8
command_shaping = 8.0 + 0.8 = 8.8
time_penalty = -0.005
Total: +8.8 - 0.005 = +8.795 per step ✓
```

### On Reaching Waypoint:

```
waypoint_progress_bonus = +50.0
```

### On Collision:

```
collision_penalty = -10.0
```

### Episode Example (50 steps, reach 1 waypoint, then collide):

```
Per-step rewards: +8.795 * 50 = +439.75
Waypoint bonus: +50.0
Collision penalty: -10.0
Total: +479.75 ✓ (vs -50.29 before)
```

---

## Files Modified

1. **`src/airsim_env/reward.py`**

    - Reduced `collision_penalty` from 50.0 to 10.0
    - Added `waypoint_progress_bonus: 50.0`
    - Added `waypoint_reached` parameter to `compute()`
    - Fixed command shaping to provide default rewards for single-agent

2. **`src/airsim_env/env.py`**

    - Track `waypoint_reached` flag in step loop
    - Pass `waypoint_reached` to `compute_reward()`

3. **`tests/integration/test_single_agent_rewards.py`**
    - Updated test to expect non-zero `command_shaping`
    - Added assertion for `waypoint_progress_bonus` component

---

## Reward Component Summary

| Component                   | Value          | When Applied                        |
| --------------------------- | -------------- | ----------------------------------- |
| **command_shaping**         | +0 to +20/step | Per-step (progress + heading)       |
| **collision_penalty**       | -10            | On collision (was -50)              |
| **waypoint_progress_bonus** | +50            | When waypoint reached (NEW)         |
| **completion_bonus**        | +100           | When all waypoints reached          |
| **idle_penalty**            | -0.5           | When stopped with progress possible |
| **time_penalty**            | -0.005         | Every step                          |

---

## Testing

### Unit/Integration Tests

All tests pass:

```bash
uv run pytest tests/unit/test_single_agent_env.py \
             tests/integration/test_single_agent_rewards.py \
             tests/unit/test_reward_components.py \
             --no-cov -q
# Result: 15 passed
```

### Expected Training Improvements

1. **Early episodes**: Small positive rewards (e.g., +5 to +50) even without reaching waypoints
2. **Mid training**: Larger rewards (+100 to +300) as agent reaches 1-2 waypoints before collision
3. **Late training**: Very large rewards (+400+) reaching multiple waypoints, potentially completing path
4. **Learning signal**: Clear gradient showing better navigation = higher reward

---

## Backward Compatibility

✅ **Hierarchical RL still works**

-   Command-specific shaping preserved for `FOLLOW_LANE`, `TURN_LEFT`, etc.
-   New waypoint bonus applies to hierarchical too (optional boost)
-   Reduced collision penalty benefits both modes

✅ **No breaking changes to API**

-   `waypoint_reached` defaults to `False` if not provided
-   Existing code continues to work

---

## Next Steps for Training

With these changes, you should see:

1. **Immediate feedback**: Run 1 episode and check logs - should see positive rewards per step
2. **Learning curve**: Rewards should trend upward over episodes (not flat at -50)
3. **Exploration**: Agent should move toward waypoints (gets +50 bonus)
4. **Collision handling**: -10 penalty still discourages crashes but doesn't dominate

**Command to re-train**:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 50 \
  --mode headless \
  --detector-model yolo12n
```

Monitor the `training_log.jsonl` - you should now see varied rewards (positive and negative) instead of all ~-50.

---

**Status**: ✅ **Fixed and tested** - Rewards now provide meaningful learning signal for single-agent navigation
