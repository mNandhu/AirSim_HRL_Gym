# Known Issues & Problems (v4.1)

**Date**: October 5, 2025  
**Status**: 🔴 **CRITICAL ISSUES BLOCKING PROGRESS**

This document consolidates all identified problems from training runs and analysis, organized by severity and certainty of cause.

---

## 🚨 CRITICAL ISSUES (Blocking Training)

### Issue #1: Lane Segmentation Not Calculating ⚠️ BLOCKER

**Status**: 🔴 **BROKEN SINCE v4.1** - High confidence in cause  
**Severity**: CRITICAL - Blocks lane deviation penalty  
**Training Impact**: Agent not penalized for off-road behavior

#### Symptoms

```json
// Every single step in all episodes:
{
    "lane_mask_coverage_ratio": null // ← Should be 0.0-1.0
}
```

#### Evidence

-   ✅ Training run 1 (1000 eps, no fix): NULL
-   ✅ Training run 2 (198 eps, with "fix"): STILL NULL
-   ✅ Settings.json has `ImageType: 5` (segmentation enabled)
-   ✅ Code has `_calculate_lane_coverage()` method
-   ❌ Method not actually calculating or returning data

#### Root Cause Analysis

**Primary Hypothesis** (90% confidence):

```python
# In airsim_util.py:
def _calculate_lane_coverage(self) -> float:
    if self._perception_pipeline is None:  # ← Most likely this
        return 1.0
    # ...
```

**Likely cause**: `perception_pipeline` parameter not being passed correctly to `AirSimSimulatorAdapter`

**Secondary Hypothesis** (70% confidence):

-   Segmentation capture failing silently (exception caught)
-   Road segment IDs `[0, 1]` wrong for Neighbourhood map
-   `simGetImages()` not returning segmentation data

#### Why It Matters

```python
# Without lane data:
lane_deviation_penalty = -1.0 * (0.0 ** 2) = 0.0  # No penalty!

# With lane data (e.g., 2m off center):
lane_deviation_penalty = -1.0 * (2.0 ** 2) = -4.0  # Actual penalty
```

#### Action Items

1. **Add debug logging** (IMMEDIATE):

    ```python
    def _calculate_lane_coverage(self) -> float:
        print(f"DEBUG: perception_pipeline = {self._perception_pipeline}")
        print(f"DEBUG: _enable_rgb = {self._enable_rgb}")

        if self._perception_pipeline is None:
            print("ERROR: No perception pipeline!")
            return 1.0

        raw_image = self._get_camera_image()
        print(f"DEBUG: raw_image type = {type(raw_image)}")

        perception_data = self._perception_pipeline.build_observation_inputs(raw_image)
        print(f"DEBUG: perception_data keys = {perception_data.keys()}")

        seg_mask = perception_data.get("segmentation_mask")
        print(f"DEBUG: seg_mask shape = {seg_mask.shape if seg_mask is not None else None}")

        if seg_mask is not None:
            print(f"DEBUG: unique segment IDs = {np.unique(seg_mask)}")
    ```

2. **Verify perception pipeline passed** (IMMEDIATE):

    ```python
    # In train_single_agent.py, line ~265:
    simulator = AirSimSimulatorAdapter(
        client,
        horizon=experiment.horizon,
        enable_rgb=enable_rgb,
        perception_pipeline=perception,  # ← Is this actually working?
    )
    ```

3. **Test on single episode** (IMMEDIATE):

    ```bash
    uv run python src/scripts/train_single_agent.py \
      --config configs/experiments/training_waypoints.yaml \
      --episodes 1 \
      --mode gui  # Watch what happens
    ```

4. **Check AirSim segment IDs** (IF step 1-3 don't work):
    - Capture segmentation manually
    - Print unique values: `np.unique(seg_mask)`
    - Adjust road pixel detection in `_calculate_lane_coverage()`

#### Expected Resolution Timeline

-   Debug logging: 10 minutes
-   Root cause identification: 30 minutes
-   Fix implementation: 1 hour
-   Validation: 10 episodes (~30 minutes)

**Total: ~2-3 hours**

---

### Issue #2: Erratic Steering Behavior (Zigzagging)

**Status**: 🟡 **CONFIRMED** - High confidence in cause  
**Severity**: HIGH - Reduces driving efficiency, increases lateral drift  
**Training Impact**: Inefficient paths, wobbling, potential off-road

#### Symptoms

```
Step  Steering  Speed
  7    +0.99    0.22
  8    -0.64    0.55   ← 1.63 unit swing!
  9    +0.47    0.86   ← 1.11 unit swing!
 10    -0.25    1.27
 11    -0.45    1.59
```

**Pattern**: Wild oscillations between -1.0 and +1.0

#### Evidence from Training

-   Episode 198 steering changes: -0.64 → +0.47 → -0.25 → -0.45
-   Trajectory shows lateral drift: Y goes from 0 → 1.8m (should stay at 0)
-   Action distribution plot: Uniform across [-1, 1] (not concentrated at 0)

#### Root Causes (Multiple Contributing Factors)

**Cause 1: No Smoothness Constraint** (100% confidence)

```python
# Current reward:
reward = progress + heading + lane_deviation

# Missing:
# steering_smoothness_penalty = -coef * |steering[t] - steering[t-1]|
```

-   Agent can change steering instantly
-   No penalty for jerky movements
-   Both smooth and zigzag paths get same reward (if reaching waypoint)

**Cause 2: High Exploration Entropy** (100% confidence)

```python
Entropy at episode 198: 0.713  # Should be < 0.3 for convergence
```

-   SAC adds Gaussian noise to actions (exploration)
-   High entropy = large noise magnitude
-   Agent still exploring randomly, not exploiting learned policy

**Cause 3: Insufficient Training** (90% confidence)

```python
Episodes: 198
Steps: ~19,000
Needed: 50,000-100,000 steps for SAC convergence
```

#### Why It Matters

-   **Inefficiency**: Zigzag path longer than straight line
-   **Lateral drift**: 1.8m off centerline on straight road
-   **Instability**: Cannot maintain stable steering
-   **Poor generalization**: Won't handle curves well

#### Action Items

1. **Add steering smoothness penalty** (HIGH PRIORITY):

    ```python
    # In src/airsim_env/reward.py:

    class RewardConfig:
        action_smoothness_coef: float = 0.5

    class RewardCalculator:
        def __init__(self, config: RewardConfig):
            self._config = config
            self._previous_steering = 0.0  # NEW

        def compute(self, ...):
            # ... existing code ...

            # NEW: Steering smoothness penalty
            current_steering = action.get("steering", 0.0)
            steering_change = abs(current_steering - self._previous_steering)
            smoothness_penalty = -self._config.action_smoothness_coef * steering_change
            self._previous_steering = current_steering

            components["action_smoothness"] = smoothness_penalty
            # ...
    ```

2. **Update reward contract** (MEDIUM PRIORITY):

    - Document new smoothness penalty
    - Add to per-step penalties table
    - Update version to 4.2

3. **Train longer** (LOW PRIORITY):
    - Need 500+ episodes
    - Wait until other issues fixed first

#### Tuning Guidance

```python
# Start conservative:
action_smoothness_coef = 0.5

# If still zigzagging:
action_smoothness_coef = 1.0  # Double the penalty

# If too conservative (won't turn):
action_smoothness_coef = 0.2  # Reduce penalty

# Monitor:
# - Average steering change per step (target < 0.3)
# - Trajectory smoothness (visual inspection)
# - Lateral drift (should stay near centerline)
```

#### Expected Impact

-   **First 50 episodes**: Steering changes reduce by ~30%
-   **Episodes 50-200**: Smooth trajectories emerge
-   **Episodes 200+**: Stable steering around centerline

---

## 🟠 HIGH PRIORITY ISSUES

### Issue #3: Episodes Terminate Early (Unknown Cause)

**Status**: 🔴 **UNKNOWN CAUSE** - Low confidence, needs investigation  
**Severity**: HIGH - Prevents learning multi-waypoint navigation  
**Training Impact**: Agent never progresses beyond first waypoint

#### Symptoms

```python
Average episode length: 96 steps (max 172)
Expected for full path: 360+ steps
Success rate: 90.4% reach "goal"
But "goal" = only first waypoint!
```

#### Evidence

-   Episode 190: Ended at (51.5, -2.1), next waypoint at (117.1, -1.9) - 65m away!
-   Episode 198: Ended at (41.3, 1.8), goal at (51.5, -0.4) - 10m away
-   Max episode: 172 steps (should be up to 1000)
-   No episodes reaching waypoints 2-8

#### Possible Causes (Ranked by Likelihood)

**Hypothesis 1: Off-Road Termination Working Despite NULL Data** (60% probability)

```python
# In env.py:
lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
if lane_ratio < 0.2:
    return True  # Terminate
```

**Evidence FOR**:

-   Episodes end around 96 steps (consistent with going off-road)
-   User observed "lane coverage barely above 20%" in plot
-   Trajectory shows 1.8m lateral drift (possibly off-road)

**Evidence AGAINST**:

-   Lane data is NULL in logs
-   Default value is 1.0 (should never trigger < 0.2)
-   Plot might be showing misleading default

**Hypothesis 2: PathManager Bug** (30% probability)

```python
# Waypoint advancement bug?
self._current_index += 1  # Increments incorrectly?
if self._current_index >= len(self._waypoints):  # 8 waypoints
    return True  # All reached
```

**Evidence FOR**:

-   Episodes marked as "goal_reached=True"
-   Termination logic checks `all_waypoints_reached`

**Evidence AGAINST**:

-   Agent only at waypoint 1-2, not waypoint 8
-   Would need to skip 6 waypoints (very unlikely)

**Hypothesis 3: Silent Exception/Error** (10% probability)

-   Exception caught somewhere
-   Episode ends prematurely
-   No error logged

**Evidence FOR**:

-   No other explanation fits
-   Short episode length unexplained

**Evidence AGAINST**:

-   No error messages in logs
-   Consistent behavior (not random crashes)

#### Why It Matters

-   **Curriculum problem**: Agent "solves" first waypoint, never learns rest
-   **Limited training**: Only learning straight-line navigation
-   **No turn learning**: Never encounters 90° turn at waypoint 3

#### Action Items

1. **Add termination reason logging** (IMMEDIATE):

    ```python
    # In src/airsim_env/env.py:

    def _check_terminated(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
        if telemetry.get("collision", False):
            return True, "collision"

        lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
        if lane_ratio < 0.2:
            return True, f"off_road (lane_coverage={lane_ratio:.3f})"

        if self._path_manager.all_waypoints_reached:
            wp_count = self._path_manager.current_waypoint_index
            return True, f"success (completed {wp_count} waypoints)"

        return False, "continuing"

    # In step():
    terminated, reason = self._check_terminated(telemetry)

    done_flags = {
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": reason,  # ADD THIS
        # ...
    }
    ```

2. **Log termination reasons** (IMMEDIATE):

    ```python
    # In metrics tracker or training log:
    if terminated or truncated:
        print(f"Episode {episode} ended: {reason}")
        # Add to episode summary
    ```

3. **Fix goal_reached flag** (IMMEDIATE):

    ```python
    # Current (WRONG):
    "goal_reached": terminated and not collision

    # Fixed:
    "goal_reached": self._path_manager.all_waypoints_reached and not collision
    ```

4. **Test hypothesis 1** (AFTER step 1-3):

    - If termination reason shows "off_road"
    - Even though lane data is NULL
    - Then there's a bug in default value handling

5. **Increase waypoint threshold temporarily** (IF no other cause found):

    ```python
    # Current:
    waypoint_threshold = 5.0  # meters

    # Temporary:
    waypoint_threshold = 10.0  # Make it easier to "reach"
    ```

    This would test if waypoint detection is the issue

#### Expected Resolution Timeline

-   Add logging: 30 minutes
-   Test run (10 episodes): 30 minutes
-   Identify root cause: Variable (could be immediate or hours)
-   Fix: 1-2 hours

**Total: 2-4 hours** (depends on what logging reveals)

---

### Issue #4: Misleading "goal_reached" Flag

**Status**: 🟡 **CONFIRMED** - 100% confidence in cause  
**Severity**: MEDIUM - Causes confusion in analysis  
**Training Impact**: Misleading success metrics

#### Problem

```python
# Current implementation:
"goal_reached": terminated and not collision

# Means:
goal_reached = True  →  "Episode ended without collision"
# NOT: "Reached all waypoints"
```

#### Evidence

```json
{
    "episode": 190,
    "goal_reached": true,
    "position": [51.58, -2.14],
    "goal_xy": [117.1, -1.9],
    "distance_to_goal": 65.5 // ← Still 65m away!
}
```

#### Why It's Misleading

-   90.4% "success rate" actually means "didn't crash"
-   Not "completed the path"
-   Inflates perceived performance

#### Action Items

```python
# Fix in src/airsim_env/env.py:

done_flags = {
    "terminated": terminated,
    "truncated": truncated,
    "collision": collision,
    "goal_reached": self._path_manager.all_waypoints_reached,  # FIX THIS
    "termination_reason": reason,  # From Issue #3
}
```

---

## 🟡 MEDIUM PRIORITY ISSUES

### Issue #5: Off-Road Threshold Too Lenient

**Status**: 🟢 **KNOWN** - User feedback  
**Severity**: MEDIUM - Allows too much off-road driving  
**Training Impact**: Agent not strictly staying on road

#### Current Setting

```python
if lane_ratio < 0.2:  # 80% off-road before termination
    return True
```

#### User Observation

> "Lane coverage barely above 20%"

#### Recommended Change

```python
# Option 1: More strict
if lane_ratio < 0.4:  # 60% off-road

# Option 2: Very strict
if lane_ratio < 0.5:  # 50% off-road

# Start with Option 1, adjust based on results
```

#### Action Items

1. Change threshold to 0.4
2. Monitor early termination rate
3. Adjust if too many false positives

---

### Issue #6: Insufficient Training Duration

**Status**: 🟢 **KNOWN** - Clear cause  
**Severity**: MEDIUM - Agent hasn't converged  
**Training Impact**: Policy not stable, still exploring

#### Evidence

```python
Episodes: 198
Steps: ~19,000
Entropy: 0.713  # Should be < 0.3
```

#### Why It Matters

-   SAC needs 50K-100K steps to converge
-   High entropy = random exploration, not exploitation
-   Policy not stable yet

#### Action Items

-   Train for 500-1000 episodes
-   Monitor entropy decay
-   Target: entropy < 0.3

---

## 🟢 LOW PRIORITY / NICE-TO-HAVE

### Issue #7: Sharp 90-Degree Turn at Waypoint 3

**Status**: 🟡 **DESIGN LIMITATION** - Known difficulty  
**Severity**: LOW - Expected challenge for RL agent  
**Training Impact**: May need curriculum learning

#### Problem

-   Waypoint 2: (117.1, -1.9)
-   Waypoint 3: (128.5, 22.8)
-   Distance: 11m horizontal, 24m vertical
-   Turn angle: ~90 degrees
-   At 10 m/s: Physically difficult

#### Current Agent Performance

-   Never reaches waypoint 3
-   Stops at waypoint 1-2
-   But Issue #3 prevents reaching WP3 anyway

#### Possible Solutions (Future)

1. **Add intermediate waypoints**:

    ```yaml
    waypoints:
      - x: 117.1, y: -1.9  # WP2
      - x: 122.0, y: 5.0   # NEW: Ease into turn
      - x: 127.0, y: 15.0  # NEW: Mid-turn
      - x: 128.5, y: 22.8  # WP3
    ```

2. **Increase waypoint threshold for turns**:

    ```python
    if is_turn_waypoint:
        threshold = 10.0  # More lenient
    else:
        threshold = 5.0   # Normal
    ```

3. **Add turn-aware speed reward**:
    ```python
    turn_sharpness = compute_turn_angle()
    target_speed = max_speed * (1.0 - 0.7 * turn_sharpness)
    # Rewards slowing down for turns
    ```

#### Action Items

-   **DEFER** until Issues #1-#3 fixed
-   Agent needs to reach WP2 consistently first

---

## 📊 Issue Priority Matrix

| Issue                 | Severity | Confidence | Blocks Training | Priority |
| --------------------- | -------- | ---------- | --------------- | -------- |
| #1 Lane Segmentation  | CRITICAL | High       | ✅ Yes          | 🔴 P0    |
| #2 Erratic Steering   | HIGH     | High       | ❌ No           | 🟠 P1    |
| #3 Early Termination  | HIGH     | Low        | ✅ Yes          | 🟠 P1    |
| #4 goal_reached Flag  | MEDIUM   | High       | ❌ No           | 🟡 P2    |
| #5 Off-Road Threshold | MEDIUM   | High       | ❌ No           | 🟡 P2    |
| #6 Training Duration  | MEDIUM   | High       | ❌ No           | 🟡 P2    |
| #7 Sharp Turns        | LOW      | High       | ❌ No           | 🟢 P3    |

---

## 🎯 Recommended Action Plan

### Phase 1: Critical Blockers (1-2 days)

**Day 1 Morning (3 hours)**:

1. ✅ Add debug logging to lane segmentation
2. ✅ Test single episode with logging
3. ✅ Identify root cause of NULL lane data
4. ✅ Fix lane segmentation calculation
5. ✅ Validate with 10 episodes

**Day 1 Afternoon (2 hours)**: 6. ✅ Add termination reason logging 7. ✅ Fix goal_reached flag 8. ✅ Test single episode 9. ✅ Identify why episodes end early

**Day 2 (4 hours)**: 10. ✅ Implement steering smoothness penalty 11. ✅ Increase off-road threshold (0.2 → 0.4) 12. ✅ Update reward contract documentation 13. ✅ Run 50-episode validation

### Phase 2: Extended Training (3-5 days)

**Training Run 1 (Episodes 1-200)**:

-   Validate all fixes working
-   Monitor termination reasons
-   Check lane coverage data flowing
-   Observe steering smoothness

**Training Run 2 (Episodes 200-500)**:

-   Agent should reach WP2 consistently
-   Entropy dropping toward 0.3
-   Smooth trajectories

**Training Run 3 (Episodes 500-1000)**:

-   Agent attempting multi-waypoint navigation
-   May need turn assistance (Issue #7)

### Phase 3: Advanced Improvements (Future)

-   Curriculum learning for turns
-   More sophisticated lane following
-   Multi-waypoint completion

---

## 🔧 Quick Reference: Fixing Critical Issues

### Fix #1: Lane Segmentation Debug

```bash
# 1. Add debug logging to _calculate_lane_coverage()
# 2. Run test:
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 1 \
  --mode gui

# 3. Check terminal output for DEBUG messages
# 4. Identify if perception_pipeline is None
```

### Fix #2: Termination Logging

```python
# src/airsim_env/env.py - Line ~333
def _check_terminated(self, telemetry) -> tuple[bool, str]:
    if telemetry.get("collision", False):
        return True, "collision"

    lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
    if lane_ratio < 0.2:
        return True, f"off_road_{lane_ratio:.2f}"

    if self._path_manager.all_waypoints_reached:
        return True, f"success_wp{self._path_manager.current_waypoint_index}"

    return False, "active"
```

### Fix #3: Steering Smoothness

```python
# src/airsim_env/reward.py
# Add to RewardConfig:
action_smoothness_coef: float = 0.5

# Add to RewardCalculator.__init__:
self._previous_steering = 0.0

# Add to compute():
current_steering = action.get("steering", 0.0)
smoothness_penalty = -self._config.action_smoothness_coef * abs(
    current_steering - self._previous_steering
)
self._previous_steering = current_steering
components["action_smoothness"] = smoothness_penalty
```

---

## 📝 Testing Checklist

After implementing fixes, verify:

-   [ ] Lane coverage is NOT NULL (should be 0.0-1.0)
-   [ ] Termination reason logged for each episode
-   [ ] goal_reached only True when all waypoints completed
-   [ ] Steering changes reduce (average < 0.3 per step)
-   [ ] Off-road terminations happen (when coverage < 0.4)
-   [ ] Episodes can reach beyond first waypoint
-   [ ] TensorBoard shows entropy decreasing
-   [ ] Trajectory plots show smooth paths

---

## 🎓 Lessons Learned

1. **Always validate telemetry data before training**

    - NULL data = broken feature
    - Should have caught in episode 1

2. **Log everything during development**

    - Termination reasons
    - Action statistics
    - Intermediate values

3. **Don't trust "success" metrics without validation**

    - goal_reached was misleading
    - Always check what flags actually mean

4. **Physics constraints matter**

    - Sharp turns need speed management
    - Smoothness should be in reward

5. **RL needs lots of training**
    - 200 episodes not enough for SAC
    - Need 500-1000 for convergence

---

**Document Status**: Living document, update as issues discovered/resolved  
**Last Updated**: October 5, 2025  
**Next Review**: After Phase 1 fixes implemented
