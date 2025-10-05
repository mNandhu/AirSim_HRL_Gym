# Lane Coverage Debug Tool

**Script**: `src/scripts/debug_lane_coverage.py`  
**Purpose**: Verify lane segmentation is working correctly at various map positions

---

## Usage

```bash
# Basic usage (just display coverage)
uv run python src/scripts/debug_lane_coverage.py

# Save segmentation images for inspection
uv run python src/scripts/debug_lane_coverage.py --save-images
```

---

## How It Works

1. Start AirSim and load your map (Neighbourhood)
2. Run the script
3. Drive manually to different positions:
    - **Road center** (should show 60-80% coverage)
    - **Road edge** (should show 30-50% coverage)
    - **Off-road** (should show <20% coverage)
    - **Spawn point** (check initial coverage)
4. Press **ENTER** at each position to capture coverage
5. Type **quit** to exit

---

## What It Shows

**For each sample**:

```
📊 Sample 1:
  Position: (0.0, 0.0, -0.5)
  Calculating lane coverage...
  [████████████░░░░░░░░]  60% ✓ GOOD
  Coverage: 0.601 (60.1%)
  ✓ Good coverage, well on road
  ✓ API control disabled - you can drive manually
```

**Coverage indicators**:

-   `[████████████████████]` - Visual bar (20 chars)
-   Percentage (0-100%)
-   Status:
    -   `✓ GOOD` - 50%+ (well on road)
    -   `⚠ THRESHOLD` - 35-50% (near v4.3 threshold)
    -   `⚠ EDGE` - 20-35% (would terminate in v4.3)
    -   `✗ OFF-ROAD` - <20% (would terminate in v4.2)

**Threshold warnings**:

```
⚠️  WOULD TERMINATE in v4.3 (< 35%)
ℹ️  Would be OK in v4.2 (>= 20%)
```

---

## Testing Checklist

### Spawn Point Test

```
Location: Start position (0, 0)
Expected: 20-30% coverage
Status: Should NOT terminate (grace period)
```

### Road Center Test

```
Location: Drive to middle of road
Expected: 60-80% coverage
Status: ✓ GOOD
```

### Road Edge Test

```
Location: Drive to edge of lane markings
Expected: 30-50% coverage
Status: ⚠ THRESHOLD or ⚠ EDGE
```

### Off-Road Test

```
Location: Drive onto grass/sidewalk
Expected: 5-15% coverage
Status: ✗ OFF-ROAD
```

### Different Locations Test

```
Test at:
- Straight road sections
- Curves
- Intersections
- Near buildings
- Various lighting conditions
```

---

## Saving Images

**With `--save-images` flag**:

```bash
uv run python src/scripts/debug_lane_coverage.py --save-images
```

**Output**:

-   Saves to: `debug_coverage/` directory
-   Filename format: `seg_YYYYMMDD_HHMMSS_xXXX_yYYY_covX.XXX.png`
-   Contains: RGB segmentation image from AirSim

**Example filenames**:

```
debug_coverage/
  seg_20251005_110230_x0_y0_cov0.250.png
  seg_20251005_110245_x45_y-2_cov0.601.png
  seg_20251005_110300_x100_y5_cov0.123.png
```

**Use images to**:

-   Verify segmentation colors are correct
-   Check if road detection works
-   Identify any segmentation issues
-   Share with team for debugging

---

## Troubleshooting

### Problem: Coverage always 0.000

**Cause**: Segmentation not working

**Solutions**:

1. Check `settings.json` has segmentation enabled:
    ```json
    "CaptureSettings": [
      {
        "ImageType": 5,
        "Width": 256,
        "Height": 144
      }
    ]
    ```
2. Restart AirSim after settings change
3. Check map has segmentation data (not all maps do)

### Problem: Coverage always 0.250 at spawn

**Status**: EXPECTED!

**Explanation**:

-   Spawn coverage ~25% is normal
-   Camera angle shows mix of road/sky/buildings
-   This is why we added grace period (10 steps)
-   After driving forward, should increase to 40-60%

### Problem: "No image response received"

**Cause**: Camera not configured or AirSim not responding

**Solutions**:

1. Check AirSim is running
2. Verify camera index (script uses camera "0")
3. Try restarting AirSim

### Problem: Coverage seems wrong (too high/low)

**Cause**: Segmentation color detection may need adjustment

**Solution**:

1. Save images with `--save-images`
2. Open saved PNG files
3. Check what colors represent road
4. Adjust RGB thresholds in script if needed:
    ```python
    # Current detection (lines 113-121):
    # Road (segment 0): RGB ~(128, 64, 128)
    # Markings (segment 1): RGB ~(244, 35, 232)
    ```

---

## Expected Results

### Spawn (Step 0-9)

```
Coverage: 0.250 (25.0%)
Status: ⚠ EDGE
Note: Grace period active, won't terminate
```

### After Driving Forward 20m

```
Coverage: 0.601 (60.1%)
Status: ✓ GOOD
Note: Should maintain this on straight roads
```

### At Road Edge (45m)

```
Coverage: 0.195 (19.5%)
Status: ✗ OFF-ROAD
Note: Would terminate in v4.3 (and v4.2)
```

---

## Integration with Training

**Use this script BEFORE training runs to**:

1. Verify segmentation is working
2. Understand what coverage values mean
3. Validate threshold choices (35% vs 20%)
4. Test different map locations
5. Debug spawn coverage issues

**If coverage looks wrong**:

-   Don't train until fixed!
-   Agent will learn based on wrong data
-   Terminations will be incorrect

---

## Examples

### Good Road Driving

```
$ uv run python src/scripts/debug_lane_coverage.py

Press ENTER to capture coverage, 'quit' to exit:

📊 Sample 1:
  Position: (45.3, -1.2, -0.5)
  Calculating lane coverage...
  [████████████░░░░░░░░]  62% ✓ GOOD
  Coverage: 0.618 (61.8%)
  ✓ Good coverage, well on road
```

### Near Threshold

```
📊 Sample 2:
  Position: (45.8, -3.5, -0.5)
  Calculating lane coverage...
  [███████░░░░░░░░░░░░░]  36% ⚠ THRESHOLD
  Coverage: 0.361 (36.1%)
  ⚠️  Near threshold, agent struggling
```

### Would Terminate

```
📊 Sample 3:
  Position: (46.2, -4.8, -0.5)
  Calculating lane coverage...
  [█████░░░░░░░░░░░░░░░]  28% ⚠ EDGE
  Coverage: 0.283 (28.3%)
  ⚠️  WOULD TERMINATE in v4.3 (< 35%)
  ℹ️  Would be OK in v4.2 (>= 20%)
```

---

## Related Files

-   `src/airsim_env/env.py` - Termination check with grace period
-   `src/perception/segmentation.py` - Segmentation adapter
-   `docs/hyperparameter-fixes-v4.3.md` - Threshold documentation
-   `docs/spawn-grace-period-fix.md` - Grace period explanation

---

**Created**: October 5, 2025  
**Version**: 1.0  
**Status**: ✅ Ready to use
