# Priority 2 Implementation: Reward Rebalancing & Command Persistence

**Date**: October 1, 2025  
**Status**: ✅ IMPLEMENTED - Ready for Testing

## Changes Implemented

### 1. Reward Configuration Rebalancing (`src/airsim_env/reward.py`)

#### Collision Penalty Reduced

```python
# Before
collision_penalty: float = 200.0

# After
collision_penalty: float = 50.0  # -75% reduction
```

**Rationale**: Single collision was erasing 92% of episode progress. New penalty still significant but allows learning from mistakes.

#### Progress Velocity Boosted

```python
# Before
progress_velocity_coef: float = 1.2

# After
progress_velocity_coef: float = 2.0  # +67% increase
```

**Rationale**: Stronger incentive for forward motion toward waypoints. Good navigation now earns +5.4 per step vs +3.24.

#### Lane Deviation Relaxed

```python
# Before
lane_deviation_penalty_coef: float = 2.0

# After
lane_deviation_penalty_coef: float = 1.0  # -50% reduction
```

**Rationale**: Less harsh penalty allows more flexibility for path-following and natural driving behavior.

#### Time Penalty Reduced

```python
# Before
time_penalty: float = -0.02  # -2.0 per 100 steps

# After
time_penalty: float = -0.005  # -0.5 per 100 steps (-75% reduction)
```

**Rationale**: Reduced constant drain allows more exploration and reduces pressure on episode length.

---

### 2. Command Persistence Added (`src/hrl_agent/coordination.py`)

#### New Feature: Minimum Command Duration

**Added to `CoordinatorState`**:

```python
@dataclass
class CoordinatorState:
    # ... existing fields ...
    steps_in_current_command: int = 0  # NEW
```

**Added to `CommandCoordinator.__init__`**:

```python
def __init__(self, ..., *, min_command_duration: int = 10):
    # ...
    self._min_command_duration = min_command_duration  # NEW
```

**Modified `act_for_env` method**:

```python
def act_for_env(self, env_id: str, observation: Any, *, deterministic: bool = True):
    prev = self._per_env_state.get(env_id)

    # NEW: Command persistence logic
    if prev is None or prev.steps_in_current_command >= self._min_command_duration:
        # Allow command selection/change
        command = self._manager.select_command(command_input, deterministic=deterministic)
    else:
        # Keep current command, increment step counter
        command = prev.command if prev.command is not None else "FOLLOW_LANE"

    # ... rest of method ...

    # Track steps in current command
    if prev is None or prev.command != command:
        state = CoordinatorState(..., steps_in_current_command=0)  # Reset on change
    else:
        state = CoordinatorState(..., steps_in_current_command=prev.steps_in_current_command + 1)  # Increment
```

**Behavior**:

-   Commands now last minimum 10 steps before manager can switch them
-   Manager is only queried every 10+ steps
-   Worker continues executing same command until minimum duration reached
-   Enables completion bonuses to trigger (commands can now complete before switching)

---

### 3. Documentation Updates

#### Updated `docs/reward-contract.md`

-   Version 2.0 → Version 3.0
-   Added changelog of all coefficient changes
-   Updated formulas and rationale
-   Documented command persistence feature

#### Created `docs/reward-analysis-with-training-data.md`

-   Deep analysis of 5-episode training run
-   Correlation between reward components and actual behavior
-   Mathematical proof of reward imbalance
-   Recommendations for tuning (all implemented)

---

## Expected Impact

### Before Changes (Based on 5-Episode Test)

**Typical 80-Step Episode**:

```
Command Shaping:    +217  (avg +2.72/step)
Time Penalty:       -1.6
Collision Penalty:  -200  (92% of progress lost!)
Completion Bonus:   +0    (never triggered)
─────────────────────────
Total:              +15   (net barely positive)
```

**Command Switching**: Every 1-3 steps (chaotic, no completions)

### After Changes (Predicted)

**Typical 80-Step Episode**:

```
Command Shaping:    +412  (avg +5.15/step, +90% improvement)
Time Penalty:       -0.4  (75% less drain)
Collision Penalty:  -50   (12% of progress lost, -75% penalty)
Completion Bonus:   +200  (2 commands complete @ +100 each)
─────────────────────────
Total:              +562  (+3600% improvement!)
```

**Command Switching**: Every 10-30 steps (stable, allows completions)

---

## Testing Instructions

### Quick Test (5 Episodes)

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 5 \
  --save-interval 10 \
  --settings settings.json \
  --mode headless \
  --detector-model yolo12n \
  --metrics-update-interval 5
```

### What to Check

1. **Episode Rewards Trend Positive**

    ```bash
    Get-Content "artifacts/<latest>/metrics/episodes.json" | ConvertFrom-Json |
      Select-Object episode, cumulative_reward, max_speed, min_distance_to_goal
    ```

    Expected: Rewards in range +100 to +400 (not -50 to +30)

2. **Command Persistence Working**

    ```bash
    Get-Content "artifacts/<latest>/metrics/episodes/episode_1_steps.json" |
      ConvertFrom-Json | Select-Object step, command | Group-Object command
    ```

    Expected:

    - Each command appears 10-30 times consecutively
    - Not 80+ different command instances per episode

3. **Completion Bonuses Triggering**

    ```bash
    Get-Content "artifacts/<latest>/metrics/episodes/episode_1_steps.json" |
      ConvertFrom-Json | Where-Object { $_.reward_components.completion_bonus -ne 0 }
    ```

    Expected: At least 1-2 steps show +100 completion bonus

4. **Collision Impact Reduced**

    - If collision occurs, check that episode reward is still positive
    - Episode with collision: +200 shaping - 50 collision = +150 (vs previous: +200 - 200 = 0)

5. **Visual Inspection**
    - Check `artifacts/<latest>/metrics/episode_summary.png`
    - Episode Rewards subplot should show upward trend or stable positive values
    - Waypoint Progress subplot should show increasing waypoints reached

---

## Full Training Test (50-100 Episodes)

After validating the 5-episode test:

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100 \
  --save-interval 10 \
  --settings settings.json \
  --mode headless \
  --detector-model yolo12n \
  --metrics-update-interval 25
```

### Expected Learning Curve

**Episodes 1-20**:

-   Initial exploration
-   Rewards: +100 to +300
-   Some collisions, but not catastrophic
-   0-2 waypoints reached

**Episodes 20-40**:

-   Command completions start triggering regularly
-   Rewards: +300 to +500
-   Collision rate decreasing
-   1-3 waypoints reached

**Episodes 40-60**:

-   Collision avoidance learned
-   Rewards: +400 to +600
-   Stable command execution
-   2-4 waypoints reached

**Episodes 60-100**:

-   Consistent high performance
-   Rewards: +500+
-   Rare collisions
-   3-5+ waypoints reached

---

## Rollback Instructions

If changes cause issues, revert with:

```bash
# Restore original reward coefficients
git checkout HEAD -- src/airsim_env/reward.py

# Restore original coordination (no command persistence)
git checkout HEAD -- src/hrl_agent/coordination.py

# Restore original documentation
git checkout HEAD -- docs/reward-contract.md
```

Or manually revert in `src/airsim_env/reward.py`:

```python
collision_penalty: float = 200.0  # Restore
progress_velocity_coef: float = 1.2  # Restore
lane_deviation_penalty_coef: float = 2.0  # Restore
time_penalty: float = -0.02  # Restore
```

---

## Tuning Guidance

If training still doesn't converge after these changes:

### Further Reduce Collision Penalty

```python
collision_penalty: float = 30.0  # or even 20.0
```

### Increase Command Duration

```python
min_command_duration: int = 15  # or 20 for very stable commands
```

### Boost Progress Rewards More

```python
progress_velocity_coef: float = 3.0  # Even stronger incentive
```

### Add Obstacle Avoidance Shaping (if available)

If you have distance-to-obstacle telemetry:

```python
def _obstacle_proximity_penalty(self, state: VehicleState) -> float:
    """Dense penalty for getting close to obstacles"""
    if hasattr(state, 'min_obstacle_distance') and state.min_obstacle_distance < 5.0:
        penalty = -(5.0 - state.min_obstacle_distance) ** 2
        return penalty * 2.0
    return 0.0
```

---

## Known Issues

1. **Logging Bug**: Collision penalties are applied but not logged in `reward_components` dict

    - Impact: Debugging harder, but learning unaffected
    - Fix location: Check `metrics_tracker.log_step()` and environment reward passing

2. **Command Completion for STOP**: Uses hysteresis (5 consecutive steps below 0.2 m/s)
    - May need adjustment based on typical stopping behavior

---

## Success Criteria

Training is successful if:

-   ✅ Episode rewards trend positive over 50 episodes
-   ✅ Completion bonuses trigger (visible in step logs)
-   ✅ Commands persist 10+ steps (not switching every step)
-   ✅ Waypoint progress increases across episodes
-   ✅ Collision rate decreases over time
-   ✅ Agent learns stable navigation patterns

If all criteria met: **Reward rebalancing successful!** 🎉

---

## Next Steps After Validation

1. Run full 100-episode training
2. Analyze learning curves and metrics
3. If still not converging, consider:

    - Curriculum learning (start with easier scenarios)
    - Hyperparameter tuning (learning rates, buffer sizes)
    - Network architecture changes
    - Additional dense rewards (obstacle avoidance, smooth steering)

4. Document final configurations and performance
5. Update reward contract with empirically validated coefficients
