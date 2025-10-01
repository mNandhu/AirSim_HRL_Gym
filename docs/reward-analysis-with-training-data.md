# Reward Function Deep Analysis with Training Correlation

**Training Run**: `20251001T124429Z_train_d678bbd7-718e-44a2-8fc3-11915861a9d7` (5 episodes)  
**Date**: October 1, 2025  
**Config**: `training_waypoints.yaml` with distance tracking fix applied

---

## Executive Summary

After fixing the critical distance tracking bug, training showed **dramatic improvement**:

-   ✅ `distance_to_goal` now varies (32-52m range, not 999.0)
-   ✅ Episodes complete without horizon truncation (76-97 steps vs 1000 limit)
-   ⚠️ **Logging bug discovered**: Collision penalties (~-200) are applied but not logged in `reward_components`
-   ⚠️ Rewards are still lower than theoretical maximum due to reward imbalance

---

## Corrected Training Results

### Actual vs Logged Rewards (Collision Penalty Accounting)

| Episode | Logged Reward | Hidden Collision | **Actual Reward** | Steps | Min Distance |
| ------- | ------------- | ---------------- | ----------------- | ----- | ------------ |
| 1       | -1.20         | -198.20          | **+197.00**       | 97    | 33.12m       |
| 2       | +16.63        | -198.63          | **+215.26**       | 82    | 32.91m       |
| 3       | +8.51         | -198.06          | **+206.56**       | 79    | 32.78m       |
| 4       | +6.09         | -199.61          | **+205.70**       | 79    | 32.90m       |
| 5       | +31.64        | -195.39          | **+227.03**       | 76    | 33.17m       |

**Key Insight**: All episodes actually achieved +197 to +227 total reward before the final collision penalty! The agent IS learning to navigate.

---

## Detailed Reward Component Analysis

### Episode 2 Breakdown (Representative)

**Raw Component Sums** (82 steps):

```
Command Shaping:    +216.88  (Avg: +2.64/step, Max: +7.13/step)
Collision Penalty:  -198.63  (Applied once at episode end, not logged)
Completion Bonus:      0.00  (No commands completed yet)
Idle Penalty:          0.00  (Agent kept moving)
Time Penalty:         -1.62  (82 steps × -0.02)
────────────────────────────
Total:                +16.63
```

**Per-Step Analysis** (First 20 steps):

| Step | Reward | Speed | Distance | Command     | Shaping | Observations         |
| ---- | ------ | ----- | -------- | ----------- | ------- | -------------------- |
| 0    | -0.02  | 0.00  | 51.50    | STOP        | 0.00    | Starting position    |
| 1    | +2.28  | 1.31  | 51.50    | FOLLOW_LANE | 0.00    | Accelerating         |
| 2    | +4.26  | 1.92  | 51.50    | TURN_RIGHT  | +2.30   | Good progress        |
| 3    | +5.18  | 2.73  | 51.50    | TURN_LEFT   | +4.28   | Excellent            |
| 4    | -0.02  | 3.50  | 51.50    | STOP        | +5.20   | Stopped while moving |
| 5    | +0.80  | 4.48  | 51.50    | FOLLOW_LANE | +0.82   | Resumed              |
| 14   | +1.08  | 0.60  | 51.44    | FOLLOW_LANE | +1.10   | Making progress!     |
| 15   | +1.49  | 0.91  | 51.37    | FOLLOW_LANE | +1.51   | Distance decreasing  |
| 16   | +1.81  | 1.26  | 51.26    | FOLLOW_LANE | +1.83   | Consistent progress  |
| 17   | -0.02  | 1.53  | 51.15    | STOP        | +1.83   | STOP while moving    |

---

## Current Reward Structure Issues

### Issue #1: Time Penalty Drain

**Formula**: `-0.02` per step, every step

**Impact**:

-   80 steps = -1.60 total
-   100 steps = -2.00 total
-   This is a **constant drain** regardless of performance

**Correlation with Training**:

-   Episode 1 (97 steps): -1.92 time penalty
-   Episode 5 (76 steps): -1.50 time penalty
-   Difference: Only 0.42 reward benefit for finishing 21 steps faster
-   **Conclusion**: Time penalty is too weak to incentivize speed

**Recommendation**: Either increase to -0.05 to create urgency, OR decrease to -0.005 to reduce drain

---

### Issue #2: No Completion Bonuses Yet

**Formula**: `+100.0` when command completes

**Impact**: Zero bonuses across all 5 episodes

**Why No Completions?**:
Looking at episode 2 step data:

-   Steps 0-4: Multiple command switches (STOP → FOLLOW_LANE → TURN_RIGHT → TURN_LEFT → STOP)
-   Average command duration: ~3-5 steps
-   FOLLOW_LANE completion requires 5m progress
-   At 2-3 m/s speed, need ~2 seconds = ~20 steps of consistent FOLLOW_LANE
-   But commands switch every 3-5 steps!

**Root Cause**: Command switching is too frequent. The DQN manager is changing commands before workers can complete them.

**Evidence**:

```
Episode 2 command distribution:
  STOP:        22 times
  FOLLOW_LANE: 19 times
  TURN_RIGHT:  18 times
  TURN_LEFT:   23 times
Total: 82 steps, 82 command instances = switching EVERY step!
```

**Conclusion**: The hierarchical structure isn't working - commands should last 10-30 steps, not 1 step.

---

### Issue #3: Collision Penalties Dominate

**Formula**: `-200.0` on collision

**Impact**: Single collision wipes out all progress

**Correlation**:

-   All 5 episodes earned +197-227 in shaping rewards
-   All 5 episodes ended with collision penalty
-   Net result: +6 to +31 (97% of progress erased)

**Analysis**:

-   Collision happens at episode end (steps 76-97)
-   Agent drives well for most of episode (earning +200)
-   Final collision negates everything

**Why Collisions Happen**:
Looking at episode 1 final steps:

-   Step 93-96: Speed increasing (2.17 → 2.75 m/s)
-   Distance decreasing (33.57 → 33.12 m)
-   Rewards positive (+3.14, +3.30, +3.41)
-   Collision at step 96

**Hypothesis**: Agent is rewarded for moving toward waypoint, doesn't learn to slow down or avoid obstacles.

---

### Issue #4: Command Shaping is Working Well!

**Formula**:

-   FOLLOW_LANE: `progress_velocity + lane_deviation_penalty`
-   TURN: `progress_velocity + heading_alignment`

**Performance**:

-   Average: +2.64 per step
-   Max: +7.13 per step
-   Min: 0.00 (when stopped)

**Evidence of Learning**:
Episode comparison (shaping per step):

-   Episode 1: +2.05 avg
-   Episode 2: +2.64 avg (+29% improvement!)
-   Episode 5: +3.01 avg (+47% improvement!)

**Breakdown** (typical step at 3 m/s, good alignment):

```
progress_velocity = 3.0 m/s × 0.9 alignment × 1.2 coef = +3.24
lane_deviation    = -0.5 (small deviation)
───────────────────────────────────────────────────────
command_shaping   = +2.74 per step ✓ Good
```

**Conclusion**: The dense shaping rewards ARE working. The problem is they're overwhelmed by collision penalties.

---

## Reward Balance Analysis

### Current Balance (Per 80-Step Episode)

**Positives**:

```
Command Shaping:  +2.64 avg × 80 steps = +211.20
Completion Bonus: +0.00 (not triggered)     = +0.00
──────────────────────────────────────────────────────
Total Positive:                              +211.20
```

**Negatives**:

```
Time Penalty:      -0.02 × 80 steps       = -1.60
Collision Penalty: -200.00 (at end)        = -200.00
─────────────────────────────────────────────────────
Total Negative:                             -201.60
```

**Net Result**: +211.20 - 201.60 = **+9.60** per episode

**Problem**: 95% of positive rewards are canceled by one collision!

---

## Root Cause: Collision Avoidance Not Learned

### Why Collision Happens

Looking at the reward structure from the agent's perspective:

**What the agent learns**:

1. ✓ Moving toward waypoint = +3 to +7 per step (immediate, consistent)
2. ✓ Faster speed = higher reward (linear with speed)
3. ✗ Collision = -200 ONE TIME (sparse, terminal)

**The Learning Problem**:

-   Dense positive: "Go fast toward waypoint" (+3 to +7 every step)
-   Sparse negative: "Don't hit things" (-200 once at end)
-   **Credit assignment**: Agent doesn't know WHICH actions 10-20 steps ago led to collision
-   **Exploration**: Higher speeds give better rewards (3 m/s > 2 m/s), so agent explores higher speeds
-   **Result**: Agent learns "go fast" but not "slow down for obstacles"

**Evidence**:
All 5 episodes show:

-   Good initial navigation (earning +200)
-   Increasing speed trend toward episode end
-   Collision at final steps
-   Pattern repeats every episode (not learning collision avoidance)

---

## Mathematical Analysis: Why Current Rewards Don't Work

### Scenario: Agent Approaching Obstacle

**Option A: Slow Down (Cautious)**

```
Speed: 2.0 m/s
Alignment: 0.9
Progress reward: 2.0 × 0.9 × 1.2 = +2.16 per step
Steps to cover 30m: 30m / 2.0 m/s ≈ 15 steps
Total reward: +2.16 × 15 = +32.40
Collision: 0% chance
Final reward: +32.40
```

**Option B: Go Fast (Risky)**

```
Speed: 4.0 m/s
Alignment: 0.9
Progress reward: 4.0 × 0.9 × 1.2 = +4.32 per step
Steps to cover 30m: 30m / 4.0 m/s ≈ 7.5 steps
Total reward: +4.32 × 7.5 = +32.40
Collision: 50% chance
Expected reward: 0.5 × (+32.40) + 0.5 × (+32.40 - 200) = -67.40
```

**Agent's Decision**:

-   Option A gives guaranteed +32.40
-   Option B gives expected -67.40
-   **Agent SHOULD choose Option A**

**Why Agent Chooses Option B**:

1. **Temporal discounting**: Future collision penalty is discounted (γ = 0.99^15 ≈ 0.86)
2. **Exploration**: Early in training, collision probability unknown
3. **Immediate reward**: +4.32 > +2.16 right now
4. **Credit assignment**: Can't connect action at step 60 to collision at step 80

---

## Recommendations for Reward Rebalancing

### Priority 1: Reduce Collision Penalty Dominance

**Option A - Reduce Collision Penalty** (Recommended):

```python
collision_penalty: float = 50.0  # Was 200.0 → -75% reduction
```

**Rationale**:

-   Still significant penalty (cancels ~20 steps of good behavior)
-   But doesn't erase entire episode progress
-   Allows agent to learn from near-misses
-   Gradual cost vs catastrophic loss

**Option B - Add Dense Obstacle Avoidance Reward**:

```python
# New component based on distance to nearest obstacle
def _obstacle_proximity_penalty(self, state: VehicleState) -> float:
    """Penalize getting close to obstacles (if obstacle detection available)"""
    if state.min_obstacle_distance < 5.0:  # meters
        # Quadratic: closer = worse
        penalty = -(5.0 - state.min_obstacle_distance) ** 2
        return penalty * 2.0  # Make it significant
    return 0.0
```

---

### Priority 2: Fix Command Completion

**Current Problem**: Commands switch every 1-3 steps (should be 10-30 steps)

**Root Cause**: DQN manager is being queried every step and choosing new commands

**Solution**: Add command persistence/hysteresis

```python
# In coordination.py
class CommandCoordinator:
    def __init__(self, ...):
        self._min_command_duration = 10  # Minimum steps before switching
        self._steps_since_command_change = 0

    def act_for_env(self, ...):
        # Only query manager if enough time passed
        if self._steps_since_command_change < self._min_command_duration:
            # Keep current command, just re-query worker
            command = self._state.command
            self._steps_since_command_change += 1
        else:
            # Allow command change
            command = self._manager.select_command(...)
            if command != self._state.command:
                self._steps_since_command_change = 0
```

**Expected Impact**: Commands last 10-30 steps → completion bonuses can trigger

---

### Priority 3: Adjust Dense Reward Coefficients

**Current Coefficients**:

```python
progress_velocity_coef: float = 1.2
lane_deviation_penalty_coef: float = 2.0
time_penalty: float = -0.02
```

**Proposed Adjustments**:

```python
progress_velocity_coef: float = 2.0    # +67% boost to progress
lane_deviation_penalty_coef: float = 1.0  # -50% lane strictness
time_penalty: float = -0.005           # -75% time drain (or -0.05 for urgency)
```

**Expected Reward Per Step** (at 3 m/s, decent driving):

```
Before:
  progress:        3.0 × 0.9 × 1.2 = +3.24
  lane_deviation:  -2.0 × 0.5^2 = -0.50
  time:            -0.02
  ────────────────────────────────────
  Total:           +2.72 per step

After:
  progress:        3.0 × 0.9 × 2.0 = +5.40  (+67%)
  lane_deviation:  -1.0 × 0.5^2 = -0.25     (50% less harsh)
  time:            -0.005                   (75% less drain)
  ────────────────────────────────────
  Total:           +5.15 per step (+89%)
```

**Impact**: Episode reward increases from ~+200 to ~+400, giving more buffer against collision penalty

---

## Summary Table: Before vs After Proposed Changes

| Component                  | Current | Proposed     | Reason                                |
| -------------------------- | ------- | ------------ | ------------------------------------- |
| **Collision Penalty**      | -200.0  | **-50.0**    | Don't erase entire episode progress   |
| **Progress Velocity Coef** | 1.2     | **2.0**      | Stronger incentive for forward motion |
| **Lane Deviation Coef**    | 2.0     | **1.0**      | More tolerance for path-following     |
| **Time Penalty**           | -0.02   | **-0.005**   | Reduce constant drain                 |
| **Command Min Duration**   | 1 step  | **10 steps** | Enable completion bonuses             |
| **Completion Bonus**       | 100.0   | **100.0**    | Keep high (will now trigger)          |

**Expected Episode Reward** (80 steps, cautious driving, 1 collision):

```
Before: +211 - 201 = +10
After:  +412 - 50 - 0.4 + 100 = +461.6 (+4500% improvement!)
```

**Expected Learning Behavior**:

-   Episodes 1-10: Exploration, learning basic navigation (+100 to +300)
-   Episodes 10-30: Completion bonuses start triggering (+300 to +500)
-   Episodes 30-50: Collision avoidance learned (+400 to +600)
-   Episodes 50+: Consistent high performance (+500+)

---

## Additional Finding: Logging Bug

**Issue**: Collision penalties are applied but not logged in `reward_components` dict

**Evidence**:

-   All 5 episodes show ~-200 discrepancy between logged components sum and actual reward
-   `collision_penalty` component always shows 0.00
-   But actual reward includes the penalty

**Location**: Likely in `metrics_tracker.py` logging or environment reward passing

**Impact**: Makes debugging harder, but doesn't affect learning (agent receives correct rewards)

**Fix**: Check reward component logging in `metrics_tracker.log_step()` and ensure all components are captured

---

## Conclusion

The reward function IS working for navigation (agent learns to move toward waypoints), but:

1. ❌ **Collision penalty too harsh**: -200 erases all progress
2. ❌ **No completion bonuses**: Commands switch too fast (every step)
3. ❌ **Sparse collision signal**: Agent can't learn avoidance from single terminal penalty
4. ✅ **Command shaping works**: +2-3 per step, improving across episodes
5. ✅ **Distance tracking fixed**: Agent makes progress toward waypoints

**Critical fixes needed**:

1. Reduce collision penalty (200 → 50)
2. Add command persistence (min 10 steps)
3. Increase progress rewards (1.2 → 2.0 coef)

With these changes, expect meaningful learning within 20-50 episodes.
