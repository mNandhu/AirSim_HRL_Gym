# Lane Coverage Calculation Bug - Investigation

**Date**: October 5, 2025  
**Status**: 🔴 **CRITICAL BUG CONFIRMED**  
**Impact**: Agent terminating at spawn due to incorrect coverage calculation

---

## Problem Summary

**User drove around entire map, max coverage: 12.7%**

```
Sample 1 (spawn): 0.0%
Sample 2 (spawn): 11.9%
Sample 3 (12m): 5.2%
Sample 4 (-54m): 11.6%
Sample 5 (-50m): 12.7% ← MAXIMUM on road!
Sample 6-11: 0-3.6%
```

**Expected**: Road center should show 60-80% coverage  
**Actual**: Maximum 12.7% even when centered on road  
**Result**: Every position triggers off-road termination (< 35%)

---

## Root Cause Analysis

### The Debug Script Was Wrong

**Original debug script** (INCORRECT):

```python
# Tried to match RGB colors
road_mask = (
    (RGB ≈ (128, 64, 128)) |  # Purple-ish
    (RGB ≈ (244, 35, 232))    # Pink
)
```

**Problem**: This is NOT how the production code works!

### The Production Code

**`src/perception/segmentation.py`** (CORRECT):

```python
# Converts RGB to segment ID
rgb = array.reshape(height, width, 3).astype(np.uint32)
mask = rgb[:,:,0] + (rgb[:,:,1] << 8) + (rgb[:,:,2] << 16)
# Result: Segment IDs (integers)
```

**`src/scripts/airsim_util.py`** (CORRECT):

```python
# Strategy 1: Count segment ID 0 (often road)
road_pixels_count = np.sum(seg_mask == 0)

# Strategy 2: If <10%, use low IDs (< 2M)
if road_pixels_count < total_pixels * 0.1:
    road_pixels = seg_mask < 2000000
    road_pixels_count = np.sum(road_pixels)
```

---

## The Fix

### Updated Debug Script

**Now matches production code**:

```python
# Convert RGB to segment IDs (same as SegmentationAdapter)
rgb_uint32 = img_rgb.astype(np.uint32)
seg_mask = rgb_uint32[:, :, 0] + (rgb_uint32[:, :, 1] << 8) + (rgb_uint32[:, :, 2] << 16)

# Use same strategy as airsim_util.py
road_pixels_count = np.sum(seg_mask == 0)
if road_pixels_count < total_pixels * 0.1:
    road_pixels = seg_mask < 2000000
    road_pixels_count = np.sum(road_pixels)

coverage = road_pixels_count / total_pixels
```

---

## Next Steps

### 1. Re-run Debug Script

```bash
uv run python src/scripts/debug_lane_coverage.py --save-images
```

**Expected output NOW**:

```
📊 Sample 1:
  ℹ️  Got RGB segmentation (256x144x3), converting to IDs...
  ℹ️  Using segment ID 0, found 15000 pixels
  ℹ️  Unique segment IDs: 5 (e.g., [0, 1234567, 2345678, ...])
  [████████████░░░░░░░░]  60% ✓ GOOD
  Coverage: 0.605 (60.5%)
```

**What to check**:

1. Are segment IDs being calculated?
2. How many unique segment IDs exist?
3. Which ID represents road? (0? or something else?)
4. Does coverage match expectations now?

### 2. If Still Low Coverage

**Problem**: Segment ID 0 is NOT road in this map

**Solution**: Identify correct road segment ID

**How to debug**:

```bash
# Run with --save-images
uv run python src/scripts/debug_lane_coverage.py --save-images

# Check saved images in debug_coverage/
# Look at image when car is CLEARLY on road
# Note the position and coverage value

# Then update strategy in both files:
# - src/scripts/airsim_util.py
# - src/scripts/debug_lane_coverage.py

# Change from:
road_pixels_count = np.sum(seg_mask == 0)

# To (example if road is ID 7):
road_pixels_count = np.sum(seg_mask == 7)
```

### 3. Check Segmentation Image Format

**The debug script now handles both**:

-   Single channel (direct segment IDs)
-   RGB format (needs conversion)

**Check which format AirSim returns**:

```
ℹ️  Got single-channel segmentation (256x144)
   OR
ℹ️  Got RGB segmentation (256x144x3), converting to IDs...
```

---

## Potential Issues

### Issue 1: Segment ID 0 is NOT Road

**Symptom**: Still showing low coverage after fix

**Cause**: In this AirSim map, ID 0 might be sky/buildings, not road

**Solution**:

1. Save images with `--save-images`
2. Drive to OBVIOUS road center position
3. Check what segment IDs appear
4. Update both files to use correct ID

**Example fix**:

```python
# If road is segment ID 7 (example)
road_pixels_count = np.sum((seg_mask == 7) | (seg_mask == 8))  # Road + markings
```

### Issue 2: All Segment IDs > 2M

**Symptom**: Debug shows "Using low-value strategy" but still 0% coverage

**Cause**: AirSim encoding segment IDs differently

**Solution**:

```python
# Check unique IDs output
# If all IDs are large (> 2M), find the road ID pattern

# Example: If road IDs are 8355840-8421376 range
road_pixels_count = np.sum((seg_mask >= 8355840) & (seg_mask <= 8421376))
```

### Issue 3: Segmentation Not Working At All

**Symptom**: No segment IDs found, or all same ID

**Cause**: Segmentation not enabled in settings.json

**Solution**:

```json
// In settings.json
"CaptureSettings": [
  {
    "ImageType": 5,  // Segmentation
    "Width": 256,
    "Height": 144,
    "FOV_Degrees": 90,
    "AutoExposureSpeed": 100,
    "MotionBlurAmount": 0
  }
]
```

Then restart AirSim!

---

## Files Modified

1. ✅ `src/scripts/debug_lane_coverage.py` - Fixed to match production logic
2. ⏭️ `src/scripts/airsim_util.py` - May need road ID update
3. ⏭️ `settings.json` - May need segmentation config check

---

## Testing Checklist

### Phase 1: Verify Debug Script Works

-   [ ] Run debug script
-   [ ] Check it shows segment IDs being calculated
-   [ ] Note unique segment ID values
-   [ ] Save images for inspection

### Phase 2: Identify Road Segment ID

-   [ ] Drive to obvious road center
-   [ ] Capture coverage
-   [ ] Check unique segment IDs at that position
-   [ ] Identify which ID(s) represent road

### Phase 3: Update Production Code

-   [ ] Update `airsim_util.py` with correct road ID
-   [ ] Re-run training for 5 episodes
-   [ ] Verify coverage values are realistic (30-80%)
-   [ ] Check termination happens at reasonable points

### Phase 4: Adjust Threshold if Needed

-   [ ] If road coverage typically 40-60%, threshold OK
-   [ ] If road coverage typically 20-30%, lower threshold to 0.25
-   [ ] Re-test with adjusted threshold

---

## Expected Timeline

**Immediate** (15 minutes):

1. Re-run debug script with fix
2. Check if segment IDs are calculated
3. Note segment ID values

**If Low Coverage Persists** (30 minutes):

1. Save images to identify road segment ID
2. Update both debug script and airsim_util.py
3. Re-test

**If Still Broken** (1 hour):

1. Check settings.json segmentation config
2. Restart AirSim
3. Try different camera or image type
4. Consider map-specific segmentation issues

---

## Related Documents

-   `docs/issue1-resolution-lane-segmentation.md` - Original fix
-   `docs/debug-lane-coverage-tool.md` - Tool documentation
-   `src/perception/segmentation.py` - RGB→ID conversion logic
-   `src/scripts/airsim_util.py` - Coverage calculation logic

---

**Status**: 🔴 **BUG CONFIRMED, DEBUG SCRIPT FIXED**  
**Next**: Re-run debug script to identify road segment ID  
**Priority**: P0 - Blocks all training
