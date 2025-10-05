# Training Analysis: 1000-Episode Run (20251002T160200Z)

**Date**: October 4, 2025  
**Run ID**: `20251002T160200Z_train_sac_7a007ed7-cd19-432f-a62b-bf206307afa2`  
**Episodes**: 1000  
**Status**: ⚠️ **CRITICAL ISSUES IDENTIFIED**

---

## Executive Summary

The agent **learned to go off-road** because there is **NO penalty for leaving the road**. Lane segmentation data is not being provided by the simulator, so `distance_from_lane_center` is always NULL, resulting in zero lane deviation penalty. The agent optimized for "reach waypoint fast" by taking shortcuts through grass and buildings.

**Key Metrics:**

-   Average Reward: 1,684 (↑ from ~500 to ~2,200 over training)
-   Average Steps: 160 / 1000 (16% of horizon)
-   Collision Rate: **99.9%** (999/1000 episodes)
-   Success Rate: **0%** (never reached all waypoints)
-   Best Reward: 2,621

---

## Critical Issues Discovered

### 1. ❌ Lane Deviation NOT Calculated (BLOCKER)

**Evidence:**

```json
{
    "lane_mask_coverage_ratio": null, // ← Should be 0.0-1.0
    "distance_from_lane_center": null // ← Should be distance in meters
}
```

**Root Cause:**

-   Segmentation adapter is NOT providing lane mask data
-   Without lane data, `distance_from_lane_center = 0` always
-   Lane deviation penalty = 0 always

**Impact:**

-   Agent has NO incentive to stay on road
-   Takes direct shortcuts between waypoints (off-road)
-   Collides with obstacles/terrain

---

### 2. ❌ Lane Penalty Not Used in Single-Agent Mode

**Current Reward** (single-agent):

```python
if not active_command or active_command == "":
    command_shaping_reward = progress + heading_alignment
    # ↑ NO lane_deviation_penalty!
```

**Hierarchical FOLLOW_LANE** (for comparison):

```python
elif active_command == "FOLLOW_LANE":
    command_shaping_reward = lane_deviation_penalty + progress
    # ↑ Includes lane penalty
```

**Issue:**

-   Lane deviation penalty only applies to hierarchical FOLLOW_LANE command
-   Single-agent doesn't use lane deviation at all
-   Even if segmentation worked, wouldn't help!

**Why This Matters:**
The reward function literally doesn't care if the agent is on or off the road in single-agent mode.

---

### 3. ❌ Agent Takes Shortcuts (Learned Behavior)

**Trajectory Analysis (Episode 1000):**

| Waypoint | Target Position   | Agent Position  | Distance | On Road?  |
| -------- | ----------------- | --------------- | -------- | --------- |
| Start    | (0, 0)            | (0, 0)          | 0m       | ✓ Yes     |
| WP1      | (51.5, -0.4)      | ~(51, -0.5)     | ~0.1m    | ✓ Yes     |
| WP2      | (117.1, -1.9)     | ~(117, -2.0)    | ~0.1m    | ✓ Yes     |
| **WP3**  | **(128.5, 22.8)** | **(134, -5.4)** | **~30m** | **❌ NO** |

**What Happened:**

1. Agent drives straight from (0,0) to (51, -0.5) ✓
2. Continues straight to (117, -2) ✓
3. **Should turn 90° right** to reach (128, 23)
4. **Instead overshoots to (134, -5)** - way off road!
5. Collides with terrain/obstacle

**Why:**

-   Reward says: "Go fast toward waypoint"
-   Turning 90° is SLOW (must decelerate)
-   Straight line is FAST (max speed)
-   Agent chose fast straight line over slow turn
-   No penalty for going off-road → learned to cut corners

---

### 4. ⚠️ 90-Degree Turn Too Difficult

**Turn Analysis:**

```
From: (117, -2) heading East
To:   (128, 23) heading North
Turn: ~90 degrees right
```

**Physics:**

-   At 10 m/s, cannot make 90° turn without slowing
-   Must decelerate to ~3-5 m/s for sharp turns
-   Takes 5-10 steps to complete turn

**Reward Conflict:**

-   `progress_velocity = speed * alignment * 2.0`
-   Turning requires LOW speed (progress reward drops)
-   Going straight keeps HIGH speed (progress reward high)
-   Agent learns: "Avoid turns, go straight"

---

## Performance Analysis

### Overall Statistics

```
Total Episodes:     1000
Avg Reward:         1684  (↑ from ~500 early to ~2200 late)
Avg Steps:          160   (16% of 1000-step horizon)
Std Dev Reward:     ~600  (high variance)
Best Episode:       2621 reward
Worst Episode:      -9 reward
Collision Rate:     99.9% (999/1000)
Success Rate:       0.0%  (never reached all 8 waypoints)
```

### Training Progress

| Episode Range | Avg Reward | Avg Steps | Observation             |
| ------------- | ---------- | --------- | ----------------------- |
| 1-100         | ~500       | ~60       | Random exploration      |
| 100-300       | ~900       | ~90       | Learning forward motion |
| 300-600       | ~1400      | ~140      | Optimizing speed        |
| 600-900       | ~1800      | ~175      | Refining trajectory     |
| 900-1000      | ~2200      | ~200      | **Maximum off-road**    |

**Interpretation:**

-   Agent IS learning (reward ↑ 4.4x)
-   Agent IS improving (steps ↑ 3.3x)
-   But learning the WRONG thing (off-road shortcuts)

### Entropy Decay

```
Step 1,000:   ent_coef = 0.976  (high exploration)
Step 160,000: ent_coef = 0.586  (moderate exploration)
```

**Analysis:**

-   Entropy decreased as expected (0.976 → 0.586)
-   Still relatively high (should be <0.3 for convergence)
-   Agent still exploring, not fully converged

---

## Root Cause Analysis

### Why Lane Segmentation is NULL

**Possible Causes:**

1. **Segmentation Disabled in Training Command**

    ```bash
    --disable-segmentation  # ← Was this flag used?
    ```

2. **Segmentation Adapter Not Initialized**

    - Check `perception.segmentation.SegmentationAdapter`
    - Verify `airsim_client.simGetImages()` returns segmentation

3. **AirSim Settings Missing Segmentation**

    ```json
    {
        "CaptureSettings": [
            { "ImageType": 5 } // ← Segmentation type
        ]
    }
    ```

4. **Telemetry Not Extracting Lane Data**
    - Check `AirSimSimulatorAdapter.step()`
    - Verify lane mask → lane_mask_coverage_ratio conversion

### Why Lane Penalty Not in Single-Agent

**Design Decision:**

-   Single-agent reward was designed as: `progress + heading_alignment`
-   Lane deviation was reserved for hierarchical FOLLOW_LANE command
-   Assumption: Waypoints would keep agent on road (WRONG!)

**The Problem:**

-   Waypoints define where to GO, not where to DRIVE
-   Agent finds shortest path (straight line)
-   Shortest path often leaves road

---

## Critical Path Fixes Required

### Priority 1: Enable Lane Segmentation (BLOCKER)

**Action Items:**

1. ✅ Verify segmentation NOT disabled: Remove `--disable-segmentation` flag
2. ✅ Check AirSim settings include segmentation capture (ImageType 5)
3. ✅ Verify `PerceptionPipeline` extracts lane mask
4. ✅ Verify `AirSimSimulatorAdapter` calculates `lane_mask_coverage_ratio`
5. ✅ Test: Print `lane_mask_coverage_ratio` in episode logs (should be 0.0-1.0, not NULL)

**Validation:**

```python
# Should see in telemetry:
{
  "lane_mask_coverage_ratio": 0.85,  // ← Not NULL!
  "distance_from_lane_center": 0.3   // ← Not NULL!
}
```

---

### Priority 2: Add Lane Deviation to Single-Agent Reward

**Current Code:**

```python
# src/airsim_env/reward.py (lines ~108-114)
if not active_command or active_command == "":
    command_shaping_reward = progress + heading_alignment
elif active_command == "FOLLOW_LANE":
    command_shaping_reward = lane_deviation_penalty + progress
```

**Proposed Fix Option A: Add Lane Penalty to Single-Agent**

```python
if not active_command or active_command == "":
    # Single-agent: combine all navigation rewards
    command_shaping_reward = progress + heading_alignment + lane_deviation_penalty
elif active_command == "FOLLOW_LANE":
    command_shaping_reward = lane_deviation_penalty + progress
```

**Proposed Fix Option B: Hybrid Mode Based on Lane Data Availability**

```python
if not active_command or active_command == "":
    # Use lane penalty if lane data available, otherwise just navigation
    if state.distance_from_lane_center > 0.01:  # Lane data available
        command_shaping_reward = progress + heading_alignment + lane_deviation_penalty
    else:  # No lane data (off-road or unavailable)
        command_shaping_reward = progress + heading_alignment
```

**Recommended: Option A** (Always include lane penalty in single-agent)

---

### Priority 3: Add Off-Road Termination

**Proposal:**

```python
# In env.py _check_terminated()
def _check_terminated(self, telemetry: Mapping[str, Any]) -> bool:
    if telemetry.get("collision", False):
        return True

    # NEW: Terminate if severely off-road
    lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
    if lane_ratio < 0.2:  # <20% on road → terminate
        return True

    return self._path_manager.all_waypoints_reached
```

**Benefits:**

-   Prevents agent from wandering off-road forever
-   Provides clear negative signal (episode ends)
-   Shortens useless exploration episodes

**Tuning:**

-   Threshold 0.2 = 80% off-road
-   Could start stricter (0.5 = 50% off-road)
-   Monitor false positives (legitimate passing?)

---

### Priority 4: Add Off-Road Penalty (In Addition to Termination)

**Proposal:**

```python
# In reward.py RewardConfig
off_road_penalty_coef: float = 5.0  # Per-step penalty for being off-road

# In reward.py compute()
def compute(self, ...):
    # ... existing code ...

    # NEW: Off-road penalty
    off_road_penalty = 0.0
    lane_ratio = telemetry.get("lane_mask_coverage_ratio", 1.0)
    if lane_ratio < 0.5:  # Mostly off-road
        off_road_ratio = 1.0 - lane_ratio  # 0.5 → 0.5, 0.0 → 1.0
        off_road_penalty = -self._config.off_road_penalty_coef * off_road_ratio

    components["off_road_penalty"] = off_road_penalty
```

**Effect:**

-   Per-step penalty proportional to off-road amount
-   At 100% off-road: -5.0 per step
-   At 50% off-road: -2.5 per step
-   At >50% on-road: 0 penalty (lane deviation handles it)

---

## Secondary Recommendations

### 5. Simplify Path (Reduce Turn Difficulty)

**Current Path Issue:**

-   Waypoint 2 → 3: 90° right turn in ~11 meters
-   Too sharp for 10 m/s speeds
-   Requires emergency braking

**Proposal A: Add Intermediate Waypoints**

```yaml
waypoints:
    - { x: 117.1, y: -1.9 } # Current WP2
    - { x: 124.0, y: 5.0 } # NEW: Eases into turn
    - { x: 128.5, y: 15.0 } # NEW: Mid-turn
    - { x: 128.5, y: 22.8 } # Current WP3
```

**Proposal B: Increase Waypoint Threshold**

-   Current: 5 meters radius
-   Proposed: 10 meters for turns, 5 meters for straights
-   Allows agent to "cut corners" slightly

**Proposal C: Start with Simpler Tracks**

-   Oval (no sharp turns)
-   Figure-8 (gentle curves)
-   Then graduate to urban (90° turns)

---

### 6. Add Turn-Aware Speed Reward

**Proposal:**

```python
def _progress_velocity_with_turn_penalty(self, state):
    # Current: speed * alignment * 2.0
    # Problem: Encourages max speed even in turns

    # Proposed: Reduce reward for sharp turns
    turn_sharpness = 1.0 - abs(alignment)  # 0=straight, 1=90° turn
    speed_target = max_speed * (1.0 - 0.7 * turn_sharpness)

    # Reward approaching target speed, not just max speed
    speed_error = abs(state.speed_mps - speed_target)
    reward = (max_speed - speed_error) * alignment * 2.0
    return max(0, reward)
```

**Effect:**

-   Straight roads: Target 10 m/s (max speed)
-   90° turns: Target 3 m/s (reduced speed)
-   Rewards matching target, penalizes over/underspeed

---

## Immediate Action Plan

### Phase 1: Diagnose Segmentation (Day 1)

**Tasks:**

1. Check training command for `--disable-segmentation`
2. Inspect `settings.json` for segmentation capture
3. Add debug prints in `PerceptionPipeline` to verify lane mask capture
4. Add debug prints in `AirSimSimulatorAdapter` to verify `lane_mask_coverage_ratio` calculation
5. Run 1 episode with verbose logging, confirm lane data NOT NULL

**Success Criteria:**

-   `lane_mask_coverage_ratio` in range [0.0, 1.0], not NULL
-   `distance_from_lane_center` in range [0.0, 10.0], not NULL

---

### Phase 2: Fix Reward Function (Day 2)

**Tasks:**

1. Modify `reward.py`: Add `lane_deviation_penalty` to single-agent shaping
2. Modify `reward.py`: Add `off_road_penalty` component
3. Modify `env.py`: Add off-road termination (lane_ratio < 0.2)
4. Update `reward-contract.md` to document changes
5. Run unit tests to verify reward calculation

**Success Criteria:**

-   Tests pass
-   Reward contract updated
-   Off-road behavior triggers termination

---

### Phase 3: Retrain and Validate (Day 3-5)

**Training:**

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 200 \
  --mode headless \
  --detector-model yolo12n
  # Note: Do NOT use --disable-segmentation
```

**Monitoring:**

1. Episode 1-10: Verify `lane_mask_coverage_ratio` NOT NULL
2. Episode 50: Check trajectory stays mostly on road
3. Episode 100: Verify off-road terminations happening
4. Episode 200: Check success rate >0%

**Success Criteria:**

-   Lane data flowing correctly
-   Agent penalized for off-road
-   Trajectories follow road (not shortcuts)
-   Some episodes reach waypoint 3+

---

### Phase 4: Path Tuning (If Needed - Day 6-7)

**If agent still struggles with 90° turns:**

1. Add intermediate waypoints (Option A above)
2. Or increase waypoint threshold radius
3. Or implement turn-aware speed reward

**Monitor:**

-   Collision rate should decrease
-   Average steps should increase (longer survival)
-   Eventually: Success rate >5%

---

## Expected Outcomes After Fixes

### Short-Term (Episodes 1-100)

-   Lane segmentation data available
-   Off-road behavior immediately penalized
-   Agent learns to stay on road (even if slow)
-   Reward may temporarily drop (learning new constraint)

### Medium-Term (Episodes 100-500)

-   Agent follows road consistently
-   Reaches waypoint 2-3 regularly
-   Still struggles with 90° turn
-   Collision rate ~80% (down from 99.9%)

### Long-Term (Episodes 500-1000)

-   Agent navigates first 3 waypoints
-   Success rate 5-10% (reaching WP3)
-   May need path simplification for full completion
-   Average steps 400-600 (vs current 160)

---

## Lessons Learned

1. **Always verify telemetry data availability before training**

    - NULL data = feature not working
    - Could have caught this in episode 1

2. **Reward design must consider ALL constraints**

    - "Reach waypoint" ≠ "Drive on road to waypoint"
    - Need explicit road-keeping reward

3. **Single-agent needs same constraints as hierarchical**

    - Lane deviation shouldn't be exclusive to FOLLOW_LANE
    - Consistency across modes

4. **Sharp turns need special handling**

    - 90° turns at 10 m/s = impossible
    - Physics-aware reward or path design

5. **Test on simple tracks first**
    - Validate rewards on straight roads
    - Then test on curves
    - Finally try urban environments

---

## Conclusion

The agent **successfully learned** to optimize the given reward function, achieving 4.4x reward improvement. However, it learned to **take off-road shortcuts** because:

1. ❌ Lane segmentation data was NULL (not being calculated)
2. ❌ Lane deviation penalty not included in single-agent mode
3. ❌ No off-road termination or penalty
4. ⚠️ 90-degree turns too difficult for current reward structure

**The agent isn't broken - the reward function is incomplete.**

With segmentation enabled and lane penalties added, the agent should learn proper road-following behavior. The 90-degree turn may still need path tuning (intermediate waypoints) or turn-aware speed rewards.

**Next Steps**: Follow the 4-phase action plan above, starting with segmentation diagnosis.

---

**Status**: ⚠️ **CRITICAL** - Training successful but learned wrong behavior  
**Priority**: Fix segmentation and reward function before continuing training  
**ETA**: 3-7 days to resolve and retrain with corrections
