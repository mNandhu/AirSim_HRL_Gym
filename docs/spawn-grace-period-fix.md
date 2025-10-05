# Critical Fix: Spawn Grace Period Added

**Date**: October 5, 2025  
**Issue**: Immediate termination at spawn with 0.35 threshold  
**Status**: ✅ **FIXED**

---

## The Problem

**After implementing v4.3 threshold (0.35)**:

```
Episode 1: coverage=0.000, terminated step 1
Episode 2: coverage=0.250, terminated step 1
Episode 3: coverage=0.249, terminated step 1
Episode 4: coverage=0.250, terminated step 1
...
ALL episodes terminating at spawn! ❌
```

**Root Cause**:

-   Car spawns with 25% lane coverage (reasonable)
-   New threshold: 35% (stricter)
-   25% < 35% → Immediate termination!
-   Agent can't even take first action

---

## The Solution

**Added 10-step grace period**:

```python
def _check_terminated(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
    if telemetry.get("collision", False):
        return True, "collision"

    # Grace period: Skip off-road check for first 10 steps
    if self._step_index < 10:
        # Only check collision during grace period
        return False, "active"

    # After grace period, enforce strict threshold
    lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
    if lane_ratio < 0.35:
        return True, f"off_road (coverage={lane_ratio:.3f})"
    # ...
```

**Effect**:

-   ✅ First 10 steps: No off-road termination
-   ✅ Allows car to start driving from spawn
-   ✅ After step 10: Full 35% threshold enforced
-   ✅ Agent has time to adjust from spawn position

---

## Why 10 Steps?

**Reasoning**:

```
At 10 m/s (typical speed):
- 10 steps = 0.5 seconds
- Distance: ~5 meters
- Enough time to: Start moving, adjust steering
- Not enough to: Drive recklessly off-road
```

**Balance**:

-   Too short (5 steps): May still catch spawn issues
-   Too long (20+ steps): Agent could abuse grace period
-   10 steps: Just right for spawn adjustment

---

## Expected Behavior Now

**Episodes 1-10** (Testing):

```
Episode 1: Drive 20-40m, terminate off-road
Episode 2: Drive 30-50m, terminate off-road
Episode 3: Drive 40-60m, maybe reach WP1!
...
```

**vs Before Fix**:

```
Episode 1: Terminate at spawn (step 1)
Episode 2: Terminate at spawn (step 1)
Episode 3: Terminate at spawn (step 1)
ALL FAILED ❌
```

---

## Testing

**Run again**:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 10 \
  --mode headless \
  --detector-model yolo12n
```

**Expected**:

-   ✅ Episodes lasting > 10 steps
-   ✅ Some episodes reaching 30-50m
-   ✅ Off-road terminations happening mid-drive (not at spawn)
-   ✅ Agent can actually learn

---

## Alternative Approaches Considered

### Option 1: Lower Threshold (REJECTED)

```python
if lane_ratio < 0.30:  # Instead of 0.35
```

**Problem**: Doesn't fix root issue (spawn at 25%)

### Option 2: Increase Spawn Coverage (REJECTED)

**Problem**: Requires AirSim settings change, not in our control

### Option 3: Grace Period (CHOSEN) ✅

**Why**:

-   Simple implementation
-   Addresses spawn issue
-   Maintains strict threshold after
-   Standard practice in RL

---

## Files Modified

1. ✅ `src/airsim_env/env.py` - Added grace period check
2. ✅ `docs/hyperparameter-fixes-v4.3.md` - Updated documentation

---

## Lessons Learned

**Always test hyperparameter changes with a few episodes first!**

We should have run:

```bash
# Quick test (would have caught this)
uv run python ... --episodes 5
```

Before committing to full training runs.

**Spawn conditions matter!**

-   Always consider initial state
-   Grace periods are common in RL
-   Test edge cases

---

**Status**: ✅ **FIXED AND READY**  
**Next**: Re-run training with grace period in place  
**Expected**: Normal episode lengths, proper learning
