# Reward Contract (Version 4.2)

**Last Updated**: 2025-10-05

This document outlines the reward shaping components for the AirSim HRL agent. The total reward is the sum of these components. The function adapts its shaping based on the active high-level command, with full support for single-agent (non-hierarchical) training.

## Changes in Version 4.2 (Steering Smoothness)

-   **Action smoothness penalty added**: Penalizes rapid steering changes to reduce zigzag behavior
-   **Coefficient**: `action_smoothness_coef = 0.5` (tunable)
-   **Formula**: `-0.5 * |steering[t] - steering[t-1]|`
-   **Effect**: Encourages smooth, gradual turns instead of erratic oscillations

### Historical Changes (Version 4.1 - Road-Following Fix)

-   **Lane deviation added to single-agent**: Now includes `lane_deviation_penalty` in command shaping
-   **Off-road termination**: Episodes end when `lane_mask_coverage_ratio < 0.2` (80% off-road)
-   **Lane segmentation enabled**: Proper calculation of road coverage from segmentation masks
-   **Single-agent formula**: Changed from `progress + heading` to `progress + heading + lane_deviation`

### Historical Changes (Version 4.0)

-   **Collision penalty reduced**: 50.0 → 10.0 (-80% to prevent signal domination)
-   **Waypoint progress bonus added**: +50.0 per waypoint (new sparse reward)
-   **Single-agent support**: Command shaping now provides default rewards when no command is active
-   **Default navigation mode**: Uses `progress_velocity + heading_alignment` for waypoint-following

### Historical Changes (Version 3.0)

-   Collision penalty reduced: 200.0 → 50.0
-   Progress velocity boosted: 1.2 → 2.0 (+67%)
-   Lane deviation relaxed: 2.0 → 1.0 (-50%)
-   Time penalty reduced: -0.02 → -0.005 (-75%)
-   Command persistence added: Minimum 10 steps per command

## 1. Major Penalties & Bonuses (Event-Driven)

These rewards are typically large and sparse, defining catastrophic failures or major successes.

| Component                     | Value/Formula | Trigger Condition                | Notes                                                                                                                                                            |
| :---------------------------- | :------------ | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`collision_penalty`**       | `-10.0`       | `VehicleState.collision == True` | Reduced from -50 to allow learning from mistakes without dominating the reward signal. Still significant enough to discourage collisions.                        |
| **`waypoint_progress_bonus`** | `+50.0`       | `waypoint_reached == True`       | **NEW in v4.0**: Large bonus awarded when reaching a waypoint. Encourages exploration and provides clear milestone feedback. Critical for single-agent training. |
| **`completion_bonus`**        | `+100.0`      | `command_completed == True`      | Large sparse bonus for successfully completing a high-level command (hierarchical) or reaching all waypoints (single-agent). The ultimate success signal.        |

## 2. Dense Shaping Rewards (Per-Step Guidance)

These rewards are calculated on every time step to provide continuous feedback to the agent, guiding it toward correct behavior. The primary shaping reward adapts based on the active command.

### 2.1. Command-Specific Shaping (`command_shaping`)

This is the main guidance signal, composed of other dense rewards depending on the task.

| Active Command                                | Formula                                                                     | Purpose                                                             |
| :-------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------ |
| **Single-Agent / No Command**                 | `progress_velocity` + `heading_alignment_reward` + `lane_deviation_penalty` | **v4.1**: Full navigation for waypoint-following WITH road-keeping. |
| **`FOLLOW_LANE`** (Hierarchical)              | `progress_velocity` + `lane_deviation_penalty`                              | Drive efficiently along the road center.                            |
| **`TURN_LEFT` / `TURN_RIGHT`** (Hierarchical) | `progress_velocity` + `heading_alignment_reward`                            | Smoothly turn towards the next waypoint without overshooting.       |
| **`STOP` / Other**                            | `0.0`                                                                       | No specific guidance when stationary or for unrecognized commands.  |

### 2.2. Core Dense Components

These are the building blocks for the command-specific shaping rewards.

| Component                      | Formula                              | Sign    | Notes                                                                                                                                                                                                                |
| :----------------------------- | :----------------------------------- | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`progress_velocity`**        | `speed_mps * max(0, cos(θ)) * 2.0`   | **+**   | Rewards forward velocity scaled by alignment with the next waypoint. `θ` is the angle between the car's forward vector and the waypoint vector. Increased coefficient (2.0) provides stronger gradient for learning. |
| **`lane_deviation_penalty`**   | `-1.0 * (distance_from_center ** 2)` | **-**   | A squared penalty for deviating from the lane centerline. Reduced coefficient (1.0) allows more flexibility for path-following while still encouraging lane centering.                                               |
| **`heading_alignment_reward`** | `cos(θ) * 1.0`                       | **+/-** | Rewards aligning the car's heading with the next waypoint. It is positive when turning towards the waypoint and negative when turning away, providing critical guidance for turns.                                   |

## 3. General Penalties (Per-Step)

These penalties apply in all situations to discourage universally undesirable behavior.

| Component               | Value/Formula                            | Trigger Condition                                 | Notes                                                                                                                                             |
| :---------------------- | :--------------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`idle_penalty`**      | `-0.5`                                   | `speed_mps < 0.1` AND `progress_possible == True` | A small penalty to discourage the agent from stopping when it should be moving (e.g., on a clear, straight road).                                 |
| **`action_smoothness`** | `-0.5 * \|steering[t] - steering[t-1]\|` | Always                                            | **NEW in v4.2**: Penalizes rapid steering changes to encourage smooth driving. Reduces zigzag behavior and lateral drift. Coefficient is tunable. |
| **`time_penalty`**      | `-0.005` per step                        | Always                                            | Reduced per-timestep penalty to minimize constant drain while still incentivizing efficient completion.                                           |

---

## 4. Reward Structure Summary

### Per-Step Breakdown (Typical Single-Agent Episode)

**Example**: Agent moving at 5 m/s toward waypoint with 80% alignment, small steering change

```
progress_velocity = 5.0 * 0.8 * 2.0 = +8.0
heading_alignment = 0.8 * 1.0 = +0.8
lane_deviation = 0.0 (on road center)
action_smoothness = -0.5 * 0.05 = -0.025 (small change)
command_shaping = 8.0 + 0.8 + 0.0 = +8.8
time_penalty = -0.005
Total per step = +8.77
```

**Episode with 50 steps, 1 waypoint, smooth driving**:

```
Per-step rewards: +8.77 * 50 = +438.5
Waypoint bonus: +50.0
Total episode reward: +488.5 ✓
```

**Episode with 50 steps, 1 waypoint, erratic steering (avg change 0.8)**:

```
Per-step rewards (with large smoothness penalty):
  progress + heading = +8.8
  action_smoothness = -0.5 * 0.8 = -0.4 (per step)
  time_penalty = -0.005
  Total per step = +8.395
Total: 8.395 * 50 + 50 = +469.75
Penalty for erratic steering: -18.75 ❌
```

Compare to hierarchical training (Version 3.0) where single-agent got:

```
command_shaping = 0.0  (bug!)
time_penalty = -0.005 * 50 = -0.25
Collision penalty: -50.0
Total: -50.25 ❌ (No learning signal!)
```

---

## 5. Tuning Recommendations

Based on training performance, consider adjusting these parameters:

### If Agent Explores Too Slowly

-   Increase `waypoint_progress_bonus`: 50.0 → 100.0
-   Increase `progress_velocity_coef`: 2.0 → 3.0

### If Collisions Remain Frequent

-   Increase `collision_penalty`: 10.0 → 20.0
-   Add collision proximity penalty (future work)

### If Agent Is Too Cautious/Slow

-   Increase `progress_velocity_coef`: 2.0 → 3.0
-   Reduce `time_penalty`: -0.005 → -0.01

### If Agent Behavior Is Jerky

-   **Increase `action_smoothness_coef`**: 0.5 → 1.0 (more strict)
-   For very smooth driving: 0.5 → 2.0
-   If too conservative (won't turn): 0.5 → 0.2
-   Monitor: Average steering change per step should be < 0.3

### If Agent Goes Off-Road Frequently

-   Increase `lane_deviation_penalty_coef`: 1.0 → 2.0
-   Reduce off-road termination threshold: 0.2 → 0.3 (more strict)

---

**Configuration**: All coefficients and thresholds are configurable in `RewardConfig` (`src/airsim_env/reward.py`). The values in this document reflect the current defaults optimized for single-agent waypoint navigation. Hierarchical command-specific shaping is preserved for backward compatibility.
