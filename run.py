"""Simple launcher script for HRL AirSim training and evaluation."""

import subprocess
import sys
from pathlib import Path


def check_airsim():
    """Check if AirSim is available."""
    try:
        import airsim  # noqa: F401

        return True
    except ImportError:
        return False


def check_stable_baselines():
    """Check if stable-baselines3 is available."""
    try:
        import stable_baselines3  # noqa: F401

        return True
    except ImportError:
        return False


def install_requirements():
    """Install missing requirements."""
    print("Installing required packages...")

    packages = []
    if not check_airsim():
        packages.append("airsim")
    if not check_stable_baselines():
        packages.append("stable-baselines3[extra]")

    if packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
            print("✅ Packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            for pkg in packages:
                print(f"  pip install {pkg}")
            return False
    return True


def main():
    print("🚗 AirSim HRL Training Framework")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("src/scripts/train_and_eval.py").exists():
        print("❌ Please run this from the project root directory")
        print("   Current directory should contain: src/, configs/, etc.")
        return 1

    # Check dependencies
    print("🔍 Checking dependencies...")

    missing_deps = []
    if not check_airsim():
        missing_deps.append("airsim")
    if not check_stable_baselines():
        missing_deps.append("stable-baselines3")

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        response = input("Would you like to install them? (y/n): ").lower().strip()

        if response == "y":
            if not install_requirements():
                return 1
        else:
            print("Please install the missing dependencies manually.")
            return 1
    else:
        print("✅ All dependencies available!")

    # Show usage instructions
    print("\n📖 Usage Instructions:")
    print("-" * 30)

    print("\n1️⃣  TRAINING MODE (Watch the car learn to drive):")
    print(
        "   uv run python src/scripts/train_and_eval.py train --config configs/experiments/training.yaml"
    )
    print("   Options:")
    print("     --episodes 50        # Number of training episodes")
    print("     --mode gui           # Show AirSim window (use 'headless' to hide)")
    print("     --save-interval 10   # Save models every N episodes")
    print("     --resume             # Continue from saved models")

    print("\n2️⃣  EVALUATION MODE (Watch trained car drive):")
    print(
        "   uv run python src/scripts/train_and_eval.py eval --config configs/experiments/baseline.yaml --models models"
    )
    print("   Options:")
    print("     --continuous         # Run multiple episodes continuously")
    print("     --mode gui           # Show AirSim window")

    print("\n⚙️  Prerequisites:")
    print("   • AirSim simulator running (Neighbourhood or similar environment)")
    print("   • Python packages: airsim, stable-baselines3 (auto-installed above)")

    print("\n🎮 Quick Start:")
    print("   1. Launch AirSim (Neighbourhood environment recommended)")
    print("   2. Run training command above")
    print("   3. Watch the car learn to drive in the AirSim window!")
    print("   4. After training, use eval mode to see the trained agent")

    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
