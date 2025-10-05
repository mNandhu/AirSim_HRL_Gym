"""
Debug script to check lane coverage at various positions on the map.

This script helps you verify that lane segmentation is working correctly by:
1. Manually driving to different positions in the simulator
2. Pressing Enter to capture lane coverage at that position
3. Viewing coverage percentage and visual feedback
4. Testing spawn points, road centers, edges, and off-road areas

Usage:
    python src/scripts/debug_lane_coverage.py [--save-images]

Controls:
    - Drive manually to desired location
    - Press ENTER to capture and display lane coverage
    - Type "quit" to exit

The script automatically:
    - Disables API control after each capture (so you can drive)
    - Re-enables API control only to read state and capture image
    - Calculates lane coverage percentage
    - Optionally saves segmentation images for inspection
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import airsim
except ImportError:
    print("ERROR: airsim package not found. Please install it:")
    print("  pip install airsim")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy package not found. Please install it:")
    print("  pip install numpy")
    sys.exit(1)


def connect_airsim() -> airsim.CarClient:
    """Connect to AirSim and return client."""
    try:
        client = airsim.CarClient()
        client.confirmConnection()
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to AirSim: {e}")
        print("\nMake sure:")
        print("  1. AirSim is running")
        print("  2. You're using the car vehicle (not drone)")
        sys.exit(1)


def get_current_position(client: airsim.CarClient) -> tuple[float, float, float]:
    """Get current vehicle position."""
    car_state = client.getCarState()
    position = car_state.kinematics_estimated.position
    return (
        float(position.x_val),
        float(position.y_val),
        float(position.z_val),
    )


def calculate_lane_coverage(client: airsim.CarClient) -> tuple[float, np.ndarray | None]:
    """
    Calculate lane coverage percentage from segmentation image.

    Returns:
        Tuple of (coverage_percentage, segmentation_array)
        coverage_percentage: 0.0-1.0 (or None if failed)
        segmentation_array: numpy array of segmentation image (or None if failed)
    """
    try:
        # Request segmentation image
        responses = client.simGetImages(
            [
                airsim.ImageRequest(
                    "0",  # Camera 0 (front camera)
                    airsim.ImageType.Segmentation,
                    pixels_as_float=False,
                    compress=False,
                )
            ]
        )

        if not responses or len(responses) == 0:
            print("  ⚠️  No image response received")
            return None, None

        response = responses[0]

        # Check if image data is valid
        if response.width == 0 or response.height == 0:
            print(f"  ⚠️  Invalid image dimensions: {response.width}x{response.height}")
            return None, None

        # Convert to numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        height = response.height
        width = response.width
        expected_single = height * width
        expected_rgb = height * width * 3

        # Check what format we got
        if img1d.size == expected_single:
            # Single channel - direct segment IDs
            print(f"  ℹ️  Got single-channel segmentation ({width}x{height})")
            seg_mask = img1d.reshape(height, width)
            img_rgb = None  # No RGB version

        elif img1d.size == expected_rgb:
            # RGB format - need to convert to segment IDs
            print(f"  ℹ️  Got RGB segmentation ({width}x{height}x3), converting to IDs...")
            img_rgb = img1d.reshape(height, width, 3)

            # Convert RGB to segment ID (same logic as SegmentationAdapter)
            rgb_uint32 = img_rgb.astype(np.uint32)
            seg_mask = (
                rgb_uint32[:, :, 0] + (rgb_uint32[:, :, 1] << 8) + (rgb_uint32[:, :, 2] << 16)
            )
            seg_mask = seg_mask.astype(np.int32)

        else:
            print(
                f"  ⚠️  Unexpected image size: {img1d.size} (expected {expected_single} or {expected_rgb})"
            )
            return None, None

        # Calculate coverage using same logic as airsim_util.py
        # After testing, confirmed that Neighbourhood map uses:
        # - Segment ID 9306614: Main road surface (pink/magenta)
        # - Segment ID 15268432: Road markings/lines
        # - Segment ID 0: Sky/background (NOT road)
        total_pixels = seg_mask.size

        # Count road-related segment IDs
        road_pixels = (seg_mask == 9306614) | (seg_mask == 15268432)
        road_pixels_count = np.sum(road_pixels)

        coverage = float(road_pixels_count) / float(total_pixels) if total_pixels > 0 else 0.0

        print(f"  ℹ️  Road pixels (ID 9306614 + 15268432): {road_pixels_count}/{total_pixels}")

        # Debug: Show segment ID distribution
        unique_ids = np.unique(seg_mask)
        print(
            f"  ℹ️  Unique segment IDs: {len(unique_ids)} (e.g., {unique_ids[:10] if len(unique_ids) > 10 else unique_ids})"
        )

        # Debug: Show TOP segment IDs by pixel count to identify road
        print("  ℹ️  Top 5 segment IDs by pixel count:")
        unique_ids_full, counts = np.unique(seg_mask, return_counts=True)
        # Sort by count (descending)
        sorted_indices = np.argsort(-counts)
        for i in range(min(5, len(unique_ids_full))):
            idx = sorted_indices[i]
            seg_id = unique_ids_full[idx]
            count = counts[idx]
            percentage = (count / total_pixels) * 100
            print(f"      ID {seg_id:8d}: {count:6d} pixels ({percentage:5.1f}%)")

        return coverage, img_rgb

    except Exception as e:
        print(f"  ⚠️  Error calculating lane coverage: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def save_segmentation_image(
    img_rgb: np.ndarray,
    position: tuple[float, float, float],
    coverage: float,
    output_dir: Path,
) -> None:
    """Save segmentation image to file."""
    try:
        from PIL import Image
    except ImportError:
        print("  ⚠️  PIL not available, cannot save images")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    x, y, z = position
    filename = f"seg_{timestamp}_x{x:.0f}_y{y:.0f}_cov{coverage:.3f}.png"
    filepath = output_dir / filename

    # Convert RGB to PIL Image and save
    img_pil = Image.fromarray(img_rgb, mode="RGB")
    img_pil.save(filepath)

    print(f"  💾 Saved: {filepath}")


def get_coverage_visual(coverage: float) -> str:
    """Get visual bar representation of coverage."""
    if coverage is None:
        return "[ERROR]"

    percentage = int(coverage * 100)
    bar_length = 20
    filled = int(bar_length * coverage)
    bar = "█" * filled + "░" * (bar_length - filled)

    # Color coding (text representation)
    if coverage >= 0.50:
        status = "✓ GOOD"
    elif coverage >= 0.35:
        status = "⚠ THRESHOLD"
    elif coverage >= 0.20:
        status = "⚠ EDGE"
    else:
        status = "✗ OFF-ROAD"

    return f"[{bar}] {percentage:3d}% {status}"


def print_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("  AirSim Lane Coverage Debug Tool")
    print("=" * 70)
    print("\nINSTRUCTIONS:")
    print("  1. Drive manually to different locations:")
    print("     - Road center (should show high coverage ~60-80%)")
    print("     - Road edge (should show medium coverage ~30-50%)")
    print("     - Off-road (should show low coverage <20%)")
    print("     - Spawn point (check initial coverage)")
    print("  2. Press ENTER to capture and display lane coverage")
    print("  3. Type 'quit' to exit")
    print("\nTHRESHOLDS:")
    print("  - Current off-road threshold: 35% (v4.3)")
    print("  - Grace period: First 10 steps (no termination)")
    print("  - Old threshold: 20% (v4.2)")
    print("\nCOVERAGE GUIDE:")
    print("  - 60-80%: Road center (ideal)")
    print("  - 40-60%: Road area (good)")
    print("  - 35-40%: Near threshold (caution)")
    print("  - 20-35%: Road edge (would terminate in v4.3)")
    print("  - <20%:   Off-road (would terminate in v4.2)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Debug lane coverage at various map positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save segmentation images to debug_coverage/ directory",
    )
    args = parser.parse_args()

    # Setup output directory if saving images
    output_dir = Path("debug_coverage")
    if args.save_images:
        print(f"Images will be saved to: {output_dir.absolute()}")

    # Connect to AirSim
    print("Connecting to AirSim...")
    client = connect_airsim()
    print("✓ Connected to AirSim\n")

    # Ensure API control is disabled initially
    client.enableApiControl(False)

    # Print instructions
    print_instructions()

    # Sample counter
    sample_count = 0

    try:
        while True:
            # Wait for user input
            user_input = input("Press ENTER to capture coverage, 'quit' to exit: ").strip().lower()

            if user_input == "quit":
                print("\n✓ Exiting lane coverage debug tool.")
                break

            elif user_input == "":
                sample_count += 1
                print(f"\n📊 Sample {sample_count}:")

                # Enable API control temporarily
                client.enableApiControl(True)

                try:
                    # Get position
                    position = get_current_position(client)
                    print(f"  Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

                    # Calculate coverage
                    print("  Calculating lane coverage...")
                    coverage, img_rgb = calculate_lane_coverage(client)

                    if coverage is not None:
                        # Display results
                        print(f"  {get_coverage_visual(coverage)}")
                        print(f"  Coverage: {coverage:.3f} ({coverage * 100:.1f}%)")

                        # Threshold checks
                        if coverage < 0.35:
                            print("  ⚠️  WOULD TERMINATE in v4.3 (< 35%)")
                            if coverage >= 0.20:
                                print("  ℹ️  Would be OK in v4.2 (>= 20%)")
                        elif coverage < 0.50:
                            print("  ⚠️  Near threshold, agent struggling")
                        else:
                            print("  ✓ Good coverage, well on road")

                        # Save image if requested
                        if args.save_images and img_rgb is not None:
                            save_segmentation_image(img_rgb, position, coverage, output_dir)
                    else:
                        print("  ✗ Failed to calculate coverage (check error above)")

                finally:
                    # CRITICAL: Disable API control so user can drive
                    client.enableApiControl(False)
                    print("  ✓ API control disabled - you can drive manually\n")

            else:
                print(f"⚠️  Unknown command: '{user_input}'. Use ENTER or 'quit'.\n")

    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user. Exiting.")
    finally:
        # Ensure API control is disabled
        client.enableApiControl(False)

    # Summary
    print("\n" + "=" * 70)
    print(f"  Captured {sample_count} coverage sample(s)")
    if args.save_images and sample_count > 0:
        print(f"  Images saved to: {output_dir.absolute()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
