# Hyperparameter Fixes: Lane-Keeping Enforcement (v4.3)

**Date**: October 5, 2025  
**Version**: 4.2 → 4.3  
**Purpose**: Fix poor lane-keeping behavior (agent driving at road edge)

---

## Problem Summary

**After 50 episodes of training**, agent learned WRONG strategy:

-   Drive at 19-20% lane coverage (minimum threshold)
-   Go off-road at ~45m mark
-   Never reach WP1 (at 51m)
-   0 waypoints completed in 50 episodes

**Root causes**:

1. Off-road threshold too lenient (20%)
2. Steering smoothness penalty too weak (0.5 coef)
3. Agent optimizing for speed, ignoring lane quality

---

## Changes Implemented

### Change #1: Stricter Off-Road Threshold

**File**: `src/airsim_env/env.py` (line ~352)

**Before**:

```python
# Terminate if severely off-road (< 20% lane coverage)
lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
if lane_ratio < 0.2:  # Less than 20% on road
    return True, f"off_road (coverage={lane_ratio:.3f})"
```

**After**:

```python
# Terminate if severely off-road (< 35% lane coverage)
# Changed from 0.20 to 0.35 to enforce better lane-keeping
# Grace period: Skip check for first 10 steps to allow spawn adjustment
if self._step_index < 10:
    # Only check collision during grace period
    return False, "active"

lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
if lane_ratio < 0.35:  # Less than 35% on road
    return True, f"off_road (coverage={lane_ratio:.3f})"
```

**Effect**:

-   Agent CANNOT coast at 19% anymore (after grace period)
-   Must maintain 35%+ coverage (more safely on road)
-   **Grace period**: First 10 steps skip off-road check (prevents spawn termination)
-   Forces learning of proper lane-keeping
-   Old strategy: "Stay at minimum" → New requirement: "Stay safely on road"

### Change #2: Stronger Smoothness Penalty

**File**: `src/airsim_env/reward.py` (line ~58)

**Before**:

```python
action_smoothness_coef: float = 0.5  # Penalty for rapid steering changes
```

**After**:

```python
action_smoothness_coef: float = 1.0  # Doubled to reduce zigzag more aggressively
```

**Effect**:

-   Penalty doubled for same steering change
-   Small change (0.05): -0.025 → -0.05
-   Large change (0.8): -0.40 → -0.80
-   Stronger incentive for smooth control

### Change #3: Documentation Updates

**File**: `docs/reward-contract.md`

-   ✅ Version bumped: 4.2 → 4.3
-   ✅ Added v4.3 changes section
-   ✅ Updated off-road threshold documentation
-   ✅ Updated smoothness penalty examples
-   ✅ Updated reward calculation examples

---

## Expected Impact

### Immediate Effects (Episodes 51-60)

**Lane Coverage**:

```
Before: 19-20% average
After:  35-50% expected (forced by new threshold)
Change: +15-30 percentage points
```

**Steering Smoothness**:

```
Before: 0.362 avg change (v4.2 with 0.5 coef)
After:  0.25-0.30 expected (v4.3 with 1.0 coef)
Change: -17-31% smoother
```

**Episode Terminations**:

```
Before: 73% off-road, mostly at 45m
After:  Expect MORE early terminations initially
Reason: Agent's old strategy now illegal
```

**Distance Progress**:

```
Before: Best 5.06m (terminate at 45m, WP1 at 51m)
After:  May initially REGRESS as agent relearns
Then:   Should reach 60-70m once adapted
Goal:   First WP1 completions (51m+)
```

### Learning Curve (Episodes 51-100)

**Phase 1 (Episodes 51-60): Adaptation**

-   Many early terminations (old strategy fails)
-   Agent forced to explore new strategies
-   Reward may drop temporarily
-   Coverage will increase (forced compliance)

**Phase 2 (Episodes 61-80): Learning**

-   Agent discovers: "Stay more on road = survive longer"
-   Steering becomes smoother (stronger penalty)
-   Episodes reaching 60-70m
-   First WP1 approaches (<10m distance)

**Phase 3 (Episodes 81-100): Breakthrough**

-   Consistent lane-keeping (35-50% coverage)
-   Smooth steering (< 0.30 avg change)
-   **First WP1 completions** (goal!)
-   Episodes lasting 120+ steps

---

## Monitoring Metrics

### Key Indicators of Success

**1. Lane Coverage** (Primary):

```bash
# Check average coverage in episodes
grep "lane_mask_coverage_ratio" episode_N_steps.json | average

Target: > 35% (minimum threshold)
Goal:   > 45% (safely on road)
```

**2. Episode Reach** (Primary):

```bash
# Check max distance driven
max distance_traveled in episodes

Target: > 60m (past WP1 at 51m)
Goal:   > 51m with WP1 completion
```

**3. Steering Smoothness** (Secondary):

```python
# Average steering change per step
avg(|steering[t] - steering[t-1]|)

Target: < 0.30
Goal:   < 0.25
```

**4. Waypoint Completions** (Primary Goal):

```bash
# Count waypoints reached
grep "goal_reached.*true" training_log.jsonl | count

Target: > 0 (any completion!)
Goal:   5-10% completion rate by episode 100
```

### Warning Signs

**If after 30 episodes (episode 80)**:

-   ❌ Coverage still < 30% average
-   ❌ No episodes reaching 60m
-   ❌ Steering change > 0.35

**Then consider**:

1. Increase lane penalty coef (1.0 → 3.0)
2. Add direct coverage bonus
3. Reduce progress reward (2.0 → 1.5)

---

## Reward Structure Changes

### Per-Step Reward Comparison

**v4.2 (Before)**:

```python
progress = +8.0
heading = +0.8
lane_deviation = -1.0
smoothness (smooth) = -0.5 * 0.05 = -0.025
smoothness (erratic) = -0.5 * 0.8 = -0.40
time = -0.005
---
Total (smooth): +7.77
Total (erratic): +7.40
```

**v4.3 (After)**:

```python
progress = +8.0
heading = +0.8
lane_deviation = -1.0
smoothness (smooth) = -1.0 * 0.05 = -0.05   (doubled)
smoothness (erratic) = -1.0 * 0.8 = -0.80   (doubled)
time = -0.005
---
Total (smooth): +7.75 (-0.02)
Total (erratic): +7.00 (-0.40, stronger penalty)
```

**Difference**: Erratic steering now costs 2x more per step

### Episode Reward Impact

**50-step episode, smooth driving**:

```
v4.2: +438.5 + 50 = +488.5
v4.3: +437.5 + 50 = +487.5
Change: -1.0 (negligible)
```

**50-step episode, erratic steering (0.8 avg change)**:

```
v4.2: +419.75 + 50 = +469.75
v4.3: +399.75 + 50 = +449.75
Change: -20.0 (significant penalty!)
```

**Message to agent**: "Smooth steering is now MUCH more important"

---

## Testing Plan

### Phase 1: Immediate Validation (10 Episodes)

**Run**:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 10 \
  --mode headless \
  --detector-model yolo12n \
  --resume models/single_agent/[latest]/sac_episode_0030.zip
```

**Check**:

1. Off-road terminations happening at 35% (not 20%)
2. More frequent early terminations (expected)
3. Stronger smoothness penalties in logs
4. Agent forced to adapt strategy

### Phase 2: Learning Validation (50 Episodes)

**Run**:

```bash
# Continue from Phase 1
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 50 \
  --mode headless \
  --detector-model yolo12n \
  --resume [phase 1 checkpoint]
```

**Check**:

1. Lane coverage increasing (target: 35-50%)
2. Episodes reaching 60m+ (past WP1)
3. Steering smoothness < 0.30
4. **First WP1 completions** (1-5 episodes)

### Phase 3: Consistency Validation (50 Episodes)

**Run**: Continue to episode 150

**Check**:

1. Consistent WP1 reaching (>25% rate)
2. Attempting WP1→WP2 straight section
3. Some episodes reaching WP2 (117m)
4. Foundation for turn learning

---

## Rollback Plan

**If changes are too aggressive** (after 30 episodes):

### Scenario 1: Too Many Early Terminations

**Symptom**: > 90% episodes < 20 steps

**Fix**: Slightly relax threshold

```python
if lane_ratio < 0.30:  # Instead of 0.35
```

### Scenario 2: Agent Can't Learn

**Symptom**: No improvement after 50 episodes, coverage still 25%

**Fix**: Reduce threshold OR add coverage reward

```python
# Option A: More lenient
if lane_ratio < 0.28:

# Option B: Direct reward
lane_coverage_bonus = (lane_ratio - 0.35) * 1.0
```

### Scenario 3: Steering Too Conservative

**Symptom**: Agent driving very slow, refusing to turn

**Fix**: Reduce smoothness penalty

```python
action_smoothness_coef: 0.75  # Between 0.5 and 1.0
```

---

## Success Criteria

### Short-Term (Episodes 51-100)

-   [ ] Lane coverage average > 35%
-   [ ] At least 1 episode reaches > 60m
-   [ ] Steering smoothness < 0.30
-   [ ] At least 1 WP1 completion

**If all 4 met**: ✅ Changes successful!

### Medium-Term (Episodes 101-150)

-   [ ] Lane coverage average > 45%
-   [ ] WP1 completion rate > 25%
-   [ ] Some episodes reaching WP2 (117m)
-   [ ] Steering smoothness < 0.25

### Long-Term (Episodes 151-200)

-   [ ] Consistent WP1-WP2 navigation
-   [ ] Attempting WP2→WP3 turn
-   [ ] Multi-waypoint progress
-   [ ] Foundation for full path completion

---

## Files Modified

1. ✅ `src/airsim_env/env.py` - Off-road threshold (0.20 → 0.35)
2. ✅ `src/airsim_env/reward.py` - Smoothness penalty (0.5 → 1.0)
3. ✅ `docs/reward-contract.md` - Version 4.3 updates

---

## Related Documents

-   `docs/known-issues-v4.1.md` - Updated with real root causes
-   `docs/training-reality-check.md` - Analysis of misunderstanding
-   `docs/training-run-analysis-50ep.md` - 50-episode baseline
-   `docs/issue2-resolution-steering-smoothness.md` - Original smoothness fix

---

**Changes Complete**: October 5, 2025  
**Status**: ✅ **READY FOR TRAINING**  
**Next**: Run 50 episodes with new hyperparameters  
**Expected**: First waypoint completions within 50 episodes!
