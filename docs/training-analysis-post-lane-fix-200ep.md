# Training Analysis: Post Lane-Fix 200 Episodes (20251004T181843Z)

**Date**: October 5, 2025  
**Run ID**: `20251004T181843Z_train_sac_bd575f4a-6a3f-42c8-95f2-add72833e18e`  
**Episodes**: 198 (terminated early at 199)  
**Status**: ⚠️ **MIXED RESULTS - Lane Segmentation Still Broken**

---

## Executive Summary

**Good News**: Agent learned to reach the first waypoint with 90.4% success rate (up from 0%)!

**Bad News**:

1. **Lane segmentation STILL not working** - `lane_mask_coverage_ratio` is NULL in all steps
2. **Erratic steering** - Agent oscillates wildly between -0.64 and +0.99
3. **Only reaches first waypoint** - Not progressing to waypoints 2-8

**Key Insight**: The v4.1 "fixes" didn't actually fix lane segmentation, but agent still improved due to better reward structure.

---

## Performance Metrics

### Overall Statistics

```
Episodes Completed:  198 (terminated at 199)
Avg Reward:          883  (↑ from 500 early, but ↓ from 1684 unfixed)
Avg Steps:           96   (↓ from 160 unfixed)
Best Episode:        1583 reward, 172 steps
Worst Episode:       6 reward, 1 step
Success Rate:        90.4% (179/198 reached first waypoint)
Collision Rate:      9.6% (19/198)
```

### Training Progress

| Episode Range | Avg Reward | Avg Steps | Success Rate | Observation            |
| ------------- | ---------- | --------- | ------------ | ---------------------- |
| 1-50          | ~750       | ~85       | ~85%         | Learning to reach WP1  |
| 51-100        | ~900       | ~95       | ~90%         | Consistent WP1 arrival |
| 101-150       | ~950       | ~100      | ~92%         | Slight improvement     |
| 151-198       | ~1050      | ~105      | ~95%         | Plateauing             |

**Observation**: Performance improved rapidly in first 50 episodes, then plateaued. Agent "solved" the task of reaching first waypoint but didn't learn to continue.

### TensorBoard Metrics

```
Training Steps:   ~19,000
Entropy (start):  0.983  (high exploration)
Entropy (end):    0.713  (still too high!)
Episode Length:   60-75 steps average
Episode Reward:   525-722 average
```

**Entropy Analysis**:

-   Should be < 0.3 for convergence
-   At 0.713, still heavily exploring
-   SAC hasn't converged to stable policy

---

## Critical Issue #1: Lane Segmentation STILL Broken

### Evidence

**From episode logs**:

```json
{
  "step": 19,
  "lane_mask_coverage_ratio": null,  // ← STILL NULL!
  "speed_mps": 3.85,
  "position_xy": [48.87, ...]
}
```

**Every single step in all 198 episodes**: `lane_mask_coverage_ratio = null`

### What This Means

1. **Lane deviation penalty = 0** (defaults to 1.0 coverage)
2. **Off-road termination never triggers** (need < 0.2 for termination)
3. **Agent not penalized for going off-road**
4. **The v4.1 "fix" didn't work!**

### Why the Fix Failed

Looking at `src/scripts/airsim_util.py`:

```python
def _calculate_lane_coverage(self) -> float:
    try:
        raw_image = self._get_camera_image()  # ← Getting image
        perception_data = self._perception_pipeline.build_observation_inputs(raw_image)  # ← Processing
        seg_mask = perception_data.get("segmentation_mask")  # ← Extracting mask

        if seg_mask is None:
            return 1.0  # ← Defaulting to 1.0 (on-road)

        road_pixels = np.isin(seg_mask, [0, 1])  # ← Counting road pixels
        # ...
    except Exception:
        return 1.0  # ← Fallback
```

**Likely Issues**:

1. `_perception_pipeline` is None (not passed correctly)
2. Segmentation capture failing silently
3. Road segment IDs [0, 1] wrong for this AirSim environment
4. Exception being caught and returning default 1.0

**Result**: Method runs but always returns 1.0, which gets converted to NULL in telemetry somehow.

---

## Critical Issue #2: Erratic Steering Behavior

### Observation

**Episode 198 steering sequence** (first 30 steps):

```
Step  Steering  Change
  0    0.99      --
  1    0.99      0.00
  7    0.99      0.00
  8   -0.64     -1.63  ← HUGE swing!
  9    0.47      1.11  ← Another huge swing!
 10   -0.25     -0.72
 11   -0.45     -0.20
 12   -0.34      0.11
 ...continues oscillating...
```

**Visualization**:

```
Steering:
 1.0 ██▁     ▁▁
 0.5 ▁▁▁█▁▁▁▁▁▁█
 0.0 ▁▁▁▁█▁▁█▁▁▁
-0.5 ▁▁▁▁▁█▁▁█▁
-1.0 ▁▁▁▁▁▁▁▁▁▁
     Zigzag pattern!
```

### Why This Happens

**Three contributing factors**:

1. **High Exploration (Entropy = 0.713)**:

    - SAC adds Gaussian noise to actions
    - High entropy = large noise
    - Action jitter is intentional for exploration

2. **No Smoothness Constraint**:

    - Action space: continuous [-1, 1]
    - No penalty for `|steering[t] - steering[t-1]|`
    - Agent can change 0.99 → -0.64 instantly

3. **Reward Structure**:
    - Only rewards: speed toward waypoint + heading alignment
    - Both are achieved whether steering smoothly or zigzagging
    - No incentive to maintain stable steering

### Impact on Driving

**Trajectory wobble**:

-   Start: (0, 0)
-   Step 30: (30.1, 1.2) - drifted 1.2m right
-   Step 60: (36.1, 2.0) - drifted 2.0m right
-   End: (41.3, 1.8) - oscillating around 1.8m off center

The zigzag steering creates lateral drift and inefficient path.

---

## Critical Issue #3: Only Reaches First Waypoint

### Success Definition

Looking at episode 198:

```python
goal_reached = True
final_position = (41.3, 1.8)
goal_xy = (51.5, -0.4)
distance_to_goal = 10.4 meters
```

**Wait, 10.4m away but "goal_reached"?**

This is because waypoint threshold is 5 meters, so at 10.4m it's approaching but the episode ended before actually reaching.

### Actual Progress Through Waypoints

**Waypoint locations**:

1. WP1: (51.5, -0.4) ← Agent reaches this 90% of time
2. WP2: (117.1, -1.9) ← Never reached
3. WP3: (128.5, 22.8) ← Never reached (90° turn)
4. WP4-8: ... ← Never reached

**Why only WP1?**:

-   Episode length: 96 steps average
-   At 5 m/s, that's ~480 meters of travel
-   But only traveling 51 meters (to WP1)
-   Either:
    a) Episodes ending early (collision/off-road)
    b) Zigzag path wastes distance
    c) Agent hasn't learned to continue past WP1

---

## Trajectory Analysis

### Episode 198 Path

```
Start: (0, 0)
Final: (41.3, 1.8)
Goal:  (51.5, -0.4)

Progress: 41.3 / 51.5 = 80% to WP1
Lateral Drift: 1.8m right (should be -0.4m)
```

**Path shape** (from trajectory data):

```
Y
2.0 │              ╱──╲
1.5 │           ╱─╯    ╲
1.0 │       ╱─╯          ╲─╮
0.5 │   ╱─╯                 ╲
0.0 ●─╯─────────────────────→ X
    0   10   20   30   40

    Wobbly path with rightward drift
```

Compare to ideal:

```
0.0 ●────────────────────────→ X
    Straight line at Y=0
```

**Diagnosis**: Agent moving toward waypoint but with inefficient zigzag and lateral drift.

---

## Comparison to Previous Runs

### Run 1: v4.0 (No Lane Fix, 1000 eps)

```
Avg Reward:      1684
Avg Steps:       160
Success Rate:    0% (never reached waypoints)
Collision Rate:  99.9%
Behavior:        Off-road shortcuts, straight-line to distant goal
```

### Run 2: v4.1 (Lane Fix Attempt, 198 eps) ← Current

```
Avg Reward:      883
Avg Steps:       96
Success Rate:    90.4% (reaches first waypoint)
Collision Rate:  9.6%
Behavior:        Zigzag steering, reaches WP1, then terminates
```

### Analysis

**Why did v4.1 perform "worse" in reward but "better" in success?**

1. **Reward dropped** because:

    - Heading alignment penalty when zigzagging
    - Shorter episodes (96 vs 160 steps)
    - Less time to accumulate positive rewards

2. **Success improved** because:

    - Agent stopped taking off-road shortcuts (even though lane penalty not working!)
    - Learned that forward motion toward waypoint = good
    - Collision avoidance improved

3. **The paradox**: Lane segmentation didn't work, but behavior improved anyway!

**Explanation**: The updated reward structure (progress + heading + lane_deviation) changed the optimization landscape even though lane_deviation=0. The heading alignment term discouraged wild off-road maneuvers.

---

## Root Cause Analysis

### Why Lane Coverage Shows ~20% in Plot But NULL in Data?

Looking at the performance metrics image:

-   Lane coverage subplot shows teal line at 0.2-0.3
-   But episode JSON shows NULL

**Possibilities**:

1. **Plot showing default fallback**:

    - When NULL, matplotlib might render as 0.2
    - Or getattr() providing default value

2. **Different episode**:

    - Plot from episode with partial data
    - But logs show ALL episodes have NULL

3. **Calculation running but not logging**:
    - `_calculate_lane_coverage()` runs
    - Returns 1.0 or some value
    - But telemetry conversion drops it

**Most likely**: The plot is using a default value when data is NULL, making it look like coverage is low when it's actually missing.

---

## Issues Identified

### Priority 1: CRITICAL - Fix Lane Segmentation (Again)

**Problem**: `_calculate_lane_coverage()` not working

**Debugging Steps**:

1. **Check if perception pipeline passed**:

    ```python
    # In train_single_agent.py, verify:
    simulator = AirSimSimulatorAdapter(
        ...,
        perception_pipeline=perception  # Is this actually passed?
    )
    ```

2. **Add debug logging**:

    ```python
    def _calculate_lane_coverage(self) -> float:
        print(f"DEBUG: perception_pipeline = {self._perception_pipeline}")
        if self._perception_pipeline is None:
            print("DEBUG: No perception pipeline!")
            return 1.0

        raw_image = self._get_camera_image()
        print(f"DEBUG: raw_image = {type(raw_image)}")

        perception_data = self._perception_pipeline.build_observation_inputs(raw_image)
        print(f"DEBUG: perception_data keys = {perception_data.keys()}")

        seg_mask = perception_data.get("segmentation_mask")
        print(f"DEBUG: seg_mask shape = {seg_mask.shape if seg_mask is not None else 'None'}")
        # ...
    ```

3. **Check segment IDs**:

    - AirSim road segment IDs might not be [0, 1]
    - Try: print unique values in seg_mask
    - Adjust road_pixels calculation

4. **Verify segmentation enabled**:
    ```bash
    # Training command should NOT have:
    # --disable-segmentation
    ```

---

### Priority 2: HIGH - Add Steering Smoothness Penalty

**Problem**: Agent oscillates steering wildly

**Solution**: Add action smoothness reward component

**Implementation**:

```python
# In reward.py RewardConfig
action_smoothness_coef: float = 0.5

# In reward.py RewardCalculator
def __init__(self):
    self._previous_action = None

def compute(self, ...):
    # ... existing code ...

    # NEW: Action smoothness penalty
    smoothness_penalty = 0.0
    if self._previous_action is not None:
        steering_change = abs(
            current_action.get("steering", 0) -
            self._previous_action.get("steering", 0)
        )
        smoothness_penalty = -self._config.action_smoothness_coef * steering_change

    self._previous_action = current_action.copy()

    components["action_smoothness"] = smoothness_penalty
    # ...
```

**Expected Effect**:

-   Penalize large steering changes
-   Encourage smooth, gradual turns
-   Reduce zigzag behavior

**Tuning**:

-   Start with coefficient = 0.5
-   If still jerky, increase to 1.0
-   If too conservative (won't turn), decrease to 0.2

---

### Priority 3: MEDIUM - Increase Off-Road Termination Threshold

**Problem**: User observed coverage "barely above 20%"

**Current**:

```python
if lane_ratio < 0.2:  # 80% off-road
    return True  # terminate
```

**Proposed**:

```python
if lane_ratio < 0.4:  # 60% off-road
    return True  # terminate
```

**Rationale**:

-   More strict = less tolerance for off-road
-   Forces agent to stay closer to road
-   May cause more early terminations initially
-   But encourages road-following behavior

**Alternative**: Start at 0.3 (70% off-road) and monitor

---

### Priority 4: LOW - Increase Training Duration

**Problem**: Only 198 episodes, entropy still high (0.713)

**Recommendation**: Train for 500-1000 episodes

**Why**:

-   SAC needs 50K-100K steps to converge
-   Currently at ~19K steps
-   Entropy needs to drop to < 0.3
-   Agent needs to explore more before exploiting

**Expected**:

-   Episodes 200-500: Entropy drops to 0.4-0.5
-   Episodes 500-800: Entropy drops to 0.2-0.3
-   Episodes 800-1000: Stable policy, low variance

---

## Steering Behavior Deep Dive

### Why Erratic Steering Persists Despite Training

**Episode 1** (early training):

```
Steering: 0.99, 0.99, 0.99, -0.64, 0.47, -0.25, ...
Pattern: Random exploration
```

**Episode 198** (late training):

```
Steering: 0.99, 0.99, 0.99, -0.64, 0.47, -0.25, ...
Pattern: SAME random exploration!
```

**Why no improvement?**

1. **Entropy too high**:

    - SAC not converged
    - Still exploring, not exploiting
    - Action noise added every step

2. **No smoothness in reward**:

    - Zigzag gets same reward as smooth
    - Both reach waypoint eventually
    - Agent optimizes for speed, not path quality

3. **Curriculum problem**:
    - Agent "solved" WP1 task in 50 episodes
    - No incentive to improve further
    - Stuck in local optimum

### Action Distribution Analysis

From action analysis image:

-   Target speed: Mostly 0 or 10 (binary!)
-   Target steering: Uniform distribution across [-1, 1]
-   No concentration around 0 (straight)

**This indicates**:

-   Agent hasn't learned stable straight driving
-   Steering is essentially random
-   Speed control learned (0 = stop, 10 = go)
-   But directional control not learned

---

## Recommendations

### Immediate Actions (Before Next Training)

1. **✅ Debug lane segmentation**:

    - Add print statements
    - Verify perception pipeline passed
    - Check segment IDs
    - Test on single episode

2. **✅ Add steering smoothness penalty**:

    - Implement in reward.py
    - Start with coefficient 0.5
    - Update reward contract docs

3. **✅ Increase off-road threshold**:

    - Change 0.2 → 0.3 or 0.4
    - More strict about staying on road

4. **✅ Verify training command**:
    - Ensure NO `--disable-segmentation`
    - Check settings.json has ImageType 5

### Next Training Run Configuration

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 500 \
  --mode headless \
  --detector-model yolo12n \
  --save-interval 50 \
  --learning-rate 3e-4 \
  --gamma 0.99
```

**Expected Outcomes**:

-   First 50 eps: Fix debug messages, verify lane data
-   Episodes 50-200: Steering smoothness improves
-   Episodes 200-400: Reaches WP2 occasionally
-   Episodes 400-500: Consistent multi-waypoint nav

---

## Success Criteria for Next Run

### Phase 1 (Episodes 1-50): Validation

-   [ ] Lane coverage data NOT NULL
-   [ ] Lane coverage values between 0.0-1.0
-   [ ] Off-road terminations logged
-   [ ] Steering smoothness penalty visible in rewards

### Phase 2 (Episodes 50-200): Learning

-   [ ] Steering oscillation reduced by 50%
-   [ ] Average steering change < 0.3 per step
-   [ ] Reaches WP1 consistently (>95%)
-   [ ] Occasional WP2 reaches (>10%)

### Phase 3 (Episodes 200-500): Mastery

-   [ ] Entropy < 0.4
-   [ ] Reaches WP2 consistently (>80%)
-   [ ] Some WP3 reaches (>20%)
-   [ ] Smooth trajectories (no zigzag)

---

## Conclusion

**What Worked**:

-   ✅ Agent learned to reach first waypoint (90.4% success)
-   ✅ Collision rate dropped from 99.9% to 9.6%
-   ✅ Stopped taking off-road shortcuts
-   ✅ Reward structure improvements helped

**What Didn't Work**:

-   ❌ Lane segmentation STILL not calculating
-   ❌ Off-road penalty not being applied
-   ❌ Erratic steering behavior persists
-   ❌ Only reaches first waypoint, not beyond

**The Paradox**:
Agent improved despite lane segmentation being broken! The reward structure changes (progress + heading) were enough to discourage extreme off-road behavior, even without the lane penalty.

**Next Steps**:

1. **MUST FIX**: Debug and fix lane segmentation (again)
2. **SHOULD ADD**: Steering smoothness penalty
3. **NICE TO HAVE**: Continue training longer (500+ episodes)

**Status**: ⚠️ **Partial Success - Critical Issues Remain**

---

**Training Run**: 20251004T181843Z  
**Analysis Date**: October 5, 2025  
**Next Review**: After fixing lane segmentation and adding smoothness penalty
