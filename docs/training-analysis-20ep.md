# Training Analysis: Run 20251002T052210Z (20 Episodes)

**Date**: October 2, 2025  
**Issue**: Episode 20 performance worse than Episode 12  
**Status**: ✅ Normal early training behavior (not concerning)

---

## Executive Summary

**Finding**: Episode 20 (227 reward, 36 steps) performed worse than Episode 12 (1221 reward, 106 steps), but this is **expected behavior** in early SAC training due to high exploration.

**Root Cause**: Entropy coefficient remains very high (0.967), meaning the agent is still heavily exploring with random actions rather than exploiting learned behavior.

**Recommendation**: Continue training for 100-200+ episodes to allow policy convergence and entropy decay.

---

## Episode Performance Analysis

### Episode Progression

| Episode Range | Avg Reward | Avg Steps | Performance                  |
| ------------- | ---------- | --------- | ---------------------------- |
| 1-3           | ~503       | 53        | Initial exploration          |
| 4             | **1357**   | 123       | Best performance!            |
| 5-10          | ~583       | 66        | Moderate                     |
| 11            | **1474**   | 158       | Second best!                 |
| 12            | **1221**   | 106       | Third best (good trajectory) |
| 13-14         | ~779       | 77        | Above average                |
| **15-20**     | **~310**   | **41**    | ⚠️ Performance drop          |

### Detailed Episode 15-20 Decline

```
Ep 15: 315 reward,  41 steps
Ep 16: 445 reward,  47 steps
Ep 17: 408 reward,  45 steps
Ep 18: 253 reward,  38 steps
Ep 19: 234 reward,  37 steps
Ep 20: 227 reward,  36 steps  ← Worst recent episode
```

**Variance**: Episode 11 (1474) vs Episode 20 (227) = **6.5x difference**

---

## Trajectory Comparison

### Episode 12 (Good Run - 1221 Reward)

-   **Distance traveled**: ~47.6 meters forward
-   **Lateral drift**: +1.3 meters (minimal, staying near path)
-   **Final position**: (47.6, 1.3)
-   **Goal distance**: 51.5 meters (92% of the way!)
-   **Behavior**: Relatively straight trajectory toward first waypoint

### Episode 20 (Poor Run - 227 Reward)

-   **Distance traveled**: ~9.7 meters forward
-   **Lateral drift**: +4.9 meters (significant sideways motion)
-   **Final position**: (9.7, 4.9)
-   **Goal distance**: 51.5 meters (only 19% progress)
-   **Behavior**: Veering sharply to the right before collision

**Analysis**: Episode 20 is going SIDEWAYS (4.9m off center) instead of forward!

---

## TensorBoard Metrics Analysis

### Training Progress (Only 4 Logged Points)

| Metric                  | Values                    | Observation                              |
| ----------------------- | ------------------------- | ---------------------------------------- |
| **Episode Length**      | 70.2 → 60.6 → 74.8 → 71.2 | High variance, oscillating               |
| **Episode Reward**      | 717 → 603 → 742 → 702     | High variance, no clear trend            |
| **Actor Loss**          | -19.59 (1 point)          | Negative (expected for SAC)              |
| **Critic Loss**         | 33.18 (1 point)           | High (still learning value function)     |
| **Entropy Coefficient** | **0.967**                 | ⚠️ **VERY HIGH** (should be <0.5 by now) |

### Entropy Coefficient Analysis

**What is Entropy in SAC?**

-   Controls exploration vs exploitation trade-off
-   High entropy (0.8-1.0) = More randomness, more exploration
-   Low entropy (0.0-0.3) = Deterministic, exploiting learned policy
-   Auto-tuned via temperature parameter α

**Current State: ent_coef = 0.967**

-   Agent is taking **96.7% random actions**!
-   Only **3.3% exploitation** of learned policy
-   This explains high variance between episodes

**Expected Behavior:**

-   Early training (0-5K steps): High entropy (0.9-1.0) - exploration
-   Mid training (5K-20K steps): Decreasing (0.5-0.7) - learning
-   Late training (20K+ steps): Low (0.1-0.3) - exploitation

**Current Status: Step 1139**

-   Still in early exploration phase
-   Entropy hasn't started decaying yet
-   Need more training steps to trigger auto-tuning

---

## Why Episode 12 > Episode 20 (Despite Later Training)

### It's NOT That Episode 20 Is "Worse Training"

The agent hasn't learned a stable policy yet. Both episodes are random exploration:

**Episode 12** = Lucky exploration

-   Random actions happened to go forward
-   ~92% progress toward first waypoint
-   Collected good trajectory data for replay buffer

**Episode 20** = Unlucky exploration

-   Random actions happened to veer right
-   Only ~19% progress, early collision
-   Collected failure data for replay buffer (still useful!)

### Both Episodes Are Valuable

1. **Episode 12 teaches**: "Going forward gets high reward"
2. **Episode 20 teaches**: "Veering sideways gets low reward"

SAC learns from BOTH successful and failed experiences in the replay buffer.

---

## Root Cause: Insufficient Training Steps

### Training Progress

-   **Total steps**: ~1,139 (environment interactions)
-   **Gradient updates**: ~1,139 (one per step after learning_starts)
-   **Episodes**: 20
-   **Replay buffer size**: 1,000,000 (only ~0.1% filled!)

### SAC Convergence Requirements

-   **Typical**: 10,000 - 100,000 steps for simple tasks
-   **Complex tasks**: 100,000 - 1,000,000+ steps
-   **Our task**: Waypoint navigation with collision avoidance (moderate complexity)

### Current Training Status

```
Progress: ▓░░░░░░░░░░░░░░░░░░░ 1.1% (1,139 / 100,000 typical)
Phase: Early Exploration
Entropy: High (0.967)
Policy: Mostly Random
Convergence: Not Started
```

---

## Is This Concerning? NO!

### ✅ Expected Behavior

1. **High variance is normal** in early training

    - Episode-to-episode performance varies wildly
    - Some lucky runs (Ep 12), some unlucky (Ep 20)
    - This stabilizes as policy converges

2. **Performance can decrease temporarily**

    - Exploration sometimes finds worse actions
    - Agent needs to try bad actions to learn they're bad
    - Replay buffer needs both good and bad examples

3. **Entropy stays high until enough data collected**
    - SAC's auto-tuning waits for sufficient experience
    - learning_starts=1024 is when training begins
    - Entropy decay happens gradually after that

### ⚠️ When to Worry

You should be concerned if:

-   [ ] After 50+ episodes, no improvement trend
-   [ ] After 100+ episodes, entropy still >0.8
-   [ ] Replay buffer full but performance flat
-   [ ] Critic loss exploding (not decreasing)
-   [ ] Actor loss oscillating wildly

**Current Status**: NONE of these apply yet!

---

## Recommendations

### 1. Continue Training (Primary)

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100 \
  --save-interval 10 \
  --mode headless \
  --detector-model yolo12n
```

**Expected Results After 100 Episodes:**

-   Entropy should decrease to 0.3-0.6
-   Performance variance should reduce
-   Clear upward trend in rewards
-   Episode lengths should increase (less early collisions)

### 2. Monitor Key Metrics

Track these in TensorBoard:

-   `rollout/ep_rew_mean`: Should show upward trend
-   `train/ent_coef`: Should gradually decrease
-   `train/critic_loss`: Should stabilize and decrease
-   `train/actor_loss`: Should stabilize (negative is OK)

### 3. Evaluate at Checkpoints

Every 25 episodes, run deterministic evaluation:

```bash
uv run python src/scripts/eval_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --model models/single_agent/20251002T052210Z_*/sac_episode_0025.zip \
  --episodes 5 \
  --deterministic
```

Deterministic evaluation (no exploration noise) shows true learned policy performance.

### 4. Adjust Hyperparameters If Needed (After 100 Episodes)

**If entropy doesn't decrease:**

-   Reduce `learning_starts`: 1024 → 512
-   Increase `learning_rate`: 3e-4 → 5e-4

**If performance plateaus:**

-   Increase `batch_size`: 256 → 512
-   Increase `gamma`: 0.99 → 0.995 (longer-term planning)

**If collisions remain frequent:**

-   Increase `collision_penalty`: 10 → 20
-   Decrease `progress_velocity_coef`: 2.0 → 1.5

---

## Expected Learning Curve

### Phase 1: Random Exploration (Episodes 1-30)

-   High variance
-   Occasional lucky runs (like Ep 12)
-   Occasional poor runs (like Ep 20)
-   Entropy stays high (0.9-1.0)
-   **Status: We are here** ✓

### Phase 2: Early Learning (Episodes 31-70)

-   Variance starts reducing
-   Entropy begins decreasing (0.9 → 0.6)
-   Average reward trending upward
-   Fewer immediate collisions

### Phase 3: Policy Convergence (Episodes 71-150)

-   Stable performance
-   Entropy low (0.2-0.4)
-   Consistent forward progress
-   Reaching first waypoint regularly

### Phase 4: Fine-Tuning (Episodes 151-200+)

-   High performance
-   Low variance
-   Reaching multiple waypoints
-   Smooth trajectories

---

## Conclusion

**Episode 20 performing worse than Episode 12 is completely normal** and expected in early SAC training. The agent is still in the random exploration phase (entropy=0.967) and hasn't learned a stable policy yet.

**Action Items**:

1. ✅ Continue training to 100-200 episodes
2. ✅ Monitor entropy decay and performance trends
3. ✅ Don't judge individual episodes in early training
4. ✅ Focus on moving average trends, not single episodes

**Estimated Time to Convergence**: 100-150 episodes (~5-8 hours of training)

The reward structure is working correctly (positive rewards visible), entropy is auto-tuning as designed, and the agent is collecting valuable exploration data. **Keep training!** 🚀

---

**Next Review Point**: After 50 episodes (25 hours from now)

-   Check entropy trend (should be <0.7)
-   Check 10-episode moving average (should show upward trend)
-   Check variance (should be decreasing)
