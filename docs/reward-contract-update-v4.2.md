# Reward Contract Update Summary

**Date**: October 5, 2025  
**Version**: 4.1 → 4.2  
**Change**: Added action smoothness penalty

---

## What Changed

### Version Bump

-   **Old**: Version 4.1 (Road-Following Fix)
-   **New**: Version 4.2 (Steering Smoothness)

### New Reward Component

**Action Smoothness Penalty** added to General Penalties (Per-Step):

| Component               | Value/Formula | Trigger                     | Notes |
| ----------------------- | ------------- | --------------------------- | ----- | ------ | --------------------------------------------------------------- |
| **`action_smoothness`** | `-0.5 \*      | steering[t] - steering[t-1] | `     | Always | Penalizes rapid steering changes. Reduces zigzag. Tunable coef. |

---

## Updated Sections

### 1. Header

-   Updated version: 4.1 → 4.2
-   Updated date: 2025-10-04 → 2025-10-05
-   Added v4.2 changes section

### 2. Section 3: General Penalties

-   Added `action_smoothness` row to table
-   Now 3 penalties: idle, smoothness, time

### 3. Section 4: Reward Examples

-   Updated examples to include smoothness penalty
-   Added comparison: smooth vs erratic steering
-   Shows ~20 reward point penalty for jerky driving

### 4. Section 5: Tuning Recommendations

-   Updated "If Agent Behavior Is Jerky" section
-   Changed from "future work" to actual tuning parameters
-   Added monitoring guidance (< 0.3 avg change)
-   Added new section for off-road tuning

---

## Example Updates

### Before (v4.1)

```
Per-step: +8.795
50 steps: +439.75
Waypoint: +50.0
Total: +489.75
```

### After (v4.2) - Smooth Driving

```
Per-step: +8.77 (includes -0.025 smoothness)
50 steps: +438.5
Waypoint: +50.0
Total: +488.5 ✓ (slight decrease, acceptable)
```

### After (v4.2) - Erratic Driving

```
Per-step: +8.395 (includes -0.4 smoothness)
50 steps: +419.75
Waypoint: +50.0
Total: +469.75 ❌ (18.75 point penalty)
```

**Result**: Agent learns smooth driving is more rewarding!

---

## Tuning Guidelines Added

**New guidance** in Section 5:

### For Jerky Behavior

```python
# More strict
action_smoothness_coef: 0.5 → 1.0

# Very strict (highway-like)
action_smoothness_coef: 0.5 → 2.0

# Too conservative
action_smoothness_coef: 0.5 → 0.2

# Target: avg steering change < 0.3
```

### For Off-Road Issues

```python
# Increase lane penalty
lane_deviation_penalty_coef: 1.0 → 2.0

# Stricter termination
off_road_threshold: 0.2 → 0.3
```

---

## Documentation Consistency

All references updated:

-   [x] Version number in header
-   [x] Last updated date
-   [x] Changes section
-   [x] General penalties table
-   [x] Example calculations
-   [x] Tuning recommendations
-   [x] Formula notation

---

## Impact on Training

**Immediate**:

-   New penalty component in reward logs
-   Visible in JSON: `"action_smoothness": -0.025`
-   Plotted in purple on reward breakdown

**After 200 episodes**:

-   Steering oscillations reduce
-   Smoother trajectories
-   Less lateral drift
-   Better path efficiency

---

## Files Updated

1. ✅ `docs/reward-contract.md` - Complete documentation update

---

**Status**: ✅ **DOCUMENTATION UPDATED**  
**Contract Version**: 4.2  
**Consistency**: All sections synchronized  
**Ready for**: Training with new penalty structure
