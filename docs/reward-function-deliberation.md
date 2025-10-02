# Reward Function Deliberation (October 2, 2025)

**Context**: After fixing sparse rewards for single-agent training, evaluated whether further reward components should be added.

---

## Current Reward Structure (Version 4.0)

### Sparse Rewards (Event-Driven)

-   **Waypoint progress bonus**: +50 (NEW - critical for exploration)
-   **Completion bonus**: +100 (reaching all waypoints)
-   **Collision penalty**: -10 (reduced from -50)

### Dense Rewards (Per-Step)

-   **Progress velocity**: Speed × alignment × 2.0 (toward waypoint)
-   **Heading alignment**: cos(θ) × 1.0 (facing waypoint)
-   **Command shaping**: Combines above for single-agent navigation
-   **Time penalty**: -0.005 (minimal drain)
-   **Idle penalty**: -0.5 (when stopped inappropriately)

---

## Potential Additions Considered

### 1. Distance-Based Progress Reward ❌ Rejected

**Proposal**: Reward reducing distance to waypoint directly

```python
distance_progress = (prev_distance - current_distance) * coef
```

**Analysis**:

-   ✅ Provides denser signal than just waypoint crossing
-   ✅ Helps when agent moves slowly but in right direction
-   ❌ Mathematically redundant with velocity-based reward
-   ❌ Could encourage zigzagging behavior
-   ❌ Adds complexity without clear benefit

**Decision**: NOT NEEDED - `velocity * alignment` already captures distance progress implicitly.

---

### 2. Action Smoothness Penalty ⏸️ Deferred

**Proposal**: Penalize jerky steering/acceleration changes

```python
smoothness_penalty = -coef * |action[t] - action[t-1]|
```

**Analysis**:

-   ✅ Encourages human-like smooth driving
-   ✅ Could improve passenger comfort metrics
-   ❌ SAC's continuous action space naturally promotes smoothness
-   ❌ May over-constrain exploration early in training
-   ⚠️ Adds state dependency (needs previous action)

**Decision**: MONITOR FIRST - Add only if trained agent shows jerky behavior in evaluation.

---

### 3. Speed Optimization Rewards ❌ Rejected

**Proposal**: Reward faster speeds or penalize excessive speed

```python
speed_bonus = speed_mps * speed_coef if speed < max_safe_speed
speed_penalty = -speed_coef * (speed - max_safe_speed) if speed > max_safe_speed
```

**Analysis**:

-   ✅ Could encourage more aggressive driving
-   ❌ Speed already rewarded in `progress_velocity`
-   ❌ Collision risk naturally limits reckless speed
-   ❌ Optimal speed depends on context (turns vs straights)

**Decision**: NOT NEEDED - Current formulation already rewards fast forward progress.

---

### 4. Lane Centerline Reward for Single-Agent ❌ Rejected

**Proposal**: Apply lane deviation penalty even without FOLLOW_LANE command

```python
if not active_command:
    command_shaping += lane_deviation_penalty  # Add to existing rewards
```

**Analysis**:

-   ✅ Could improve lane-keeping behavior
-   ❌ Not all scenarios have lane segmentation data
-   ❌ Waypoint following is primary objective, not lane centering
-   ❌ May conflict with necessary lane changes

**Decision**: KEEP CURRENT - Lane deviation only for explicit FOLLOW_LANE command.

---

### 5. Exploration Bonus ❌ Rejected

**Proposal**: Reward visiting new grid cells or areas

```python
exploration_bonus = +bonus if cell not in visited_cells
```

**Analysis**:

-   ✅ Common in sparse-reward RL
-   ❌ Waypoints already provide exploration structure
-   ❌ Adds complexity (state tracking)
-   ❌ May encourage wandering away from goal

**Decision**: NOT NEEDED - Waypoint bonuses provide sufficient exploration incentive.

---

### 6. Incremental Waypoint Distance Reward ❌ Rejected

**Proposal**: Reward getting closer to waypoint, not just crossing threshold

```python
distance_shaping = -distance_to_waypoint * distance_coef
```

**Analysis**:

-   ✅ Denser signal than discrete waypoint bonus
-   ❌ Could dominate other reward components
-   ❌ Agent might hover near waypoint instead of crossing
-   ❌ Mathematically similar to velocity-based reward

**Decision**: NOT NEEDED - Waypoint bonus (+50) is sufficient sparse signal.

---

## Final Recommendation

**KEEP CURRENT REWARD STRUCTURE** with no additional components.

### Rationale:

1. **Simplicity**: Current 6-component design is easy to understand and debug
2. **Coverage**: Dense (navigation) + sparse (milestones) + penalties (safety)
3. **Proven**: Similar structures work in autonomous driving research
4. **Extensible**: Easy to add components later if specific issues arise

### Suggested Tuning Parameters (If Needed):

| Problem             | Adjustment                          | Reasoning                      |
| ------------------- | ----------------------------------- | ------------------------------ |
| Slow exploration    | `waypoint_progress_bonus`: 50 → 100 | Stronger pull toward waypoints |
| Frequent collisions | `collision_penalty`: 10 → 20        | Increase safety emphasis       |
| Too cautious        | `progress_velocity_coef`: 2.0 → 3.0 | Reward faster movement         |
| Too slow            | `time_penalty`: -0.005 → -0.01      | Penalize inefficiency more     |
| Jerky behavior      | Add `action_smoothness_penalty`     | Only if observed in eval       |

---

## Implementation Status

✅ **Version 4.0 Implemented** (October 2, 2025)

-   Single-agent command shaping: `progress + heading_alignment`
-   Waypoint progress bonus: +50
-   Collision penalty reduced: -10
-   All tests passing (12/12)

📄 **Documentation Updated**:

-   `docs/reward-contract.md` - Version 4.0 contract
-   `docs/reward-system-fix.md` - Implementation details
-   `docs/reward-function-deliberation.md` - This document

🧪 **Next Steps**:

1. Train 50-100 episodes with new reward structure
2. Analyze learning curves and episode statistics
3. Tune parameters if specific issues emerge
4. Consider deferred additions only if clear benefit demonstrated

---

**Conclusion**: The current reward design is theoretically sound and empirically testable. Further additions should be data-driven based on observed training behavior, not speculative optimization.
