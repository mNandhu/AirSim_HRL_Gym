"""Evaluate a trained single-agent SAC policy in AirSim."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from airsim_env.single_agent_env import SingleAgentAirSimEnv
from config.loader import load_experiment
from scripts.airsim_util import (
    AirSimSimulatorAdapter,
    build_perception,
    extract_waypoints,
    get_airsim_client,
    sim_context,
)
from utils.artifacts import ArtifactManager
from utils.metrics_tracker import MetricsTracker

try:
    from stable_baselines3 import SAC
except ImportError as exc:  # pragma: no cover - surfaced to CLI users
    raise SystemExit(
        f"Missing required dependency: {exc}. Install with: pip install stable-baselines3"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single-agent SAC policy")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--model", required=True, help="Path to SAC checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--settings", default="settings.json", help="AirSim settings file")
    parser.add_argument("--mode", choices=["gui", "headless"], default="gui")
    parser.add_argument("--detector-model", default="yolov8n")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--disable-segmentation", action="store_true")
    parser.add_argument("--perception-sync", action="store_true")
    parser.add_argument("--output", default="artifacts", help="Artifact directory root")
    return parser.parse_args()


def _log_episode(
    artifact_manager: ArtifactManager,
    episode: int,
    reward: float,
    steps: int,
    info: dict[str, Any],
) -> None:
    payload = {
        "episode": episode,
        "reward": reward,
        "steps": steps,
        "goal_reached": bool(info.get("done_flags", {}).get("goal_reached", False)),
        "collision": bool(info.get("done_flags", {}).get("collision", False)),
        "config_hash": info.get("config_hash"),
        "telemetry": info.get("telemetry"),
        "reward_components": info.get("reward_components"),
    }
    artifact_manager.append_jsonl("evaluation_log.jsonl", payload)


def main() -> int:
    args = parse_args()

    experiment = load_experiment(args.config)
    artifact_manager = ArtifactManager(args.output)
    run_paths = artifact_manager.start_run(f"eval_sac_{experiment.id}")

    settings_src = os.environ.get("AIRSIM_SETTINGS_PATH", args.settings)
    try:
        src_path = Path(settings_src)
        if src_path.exists():
            shutil.copy2(src_path, run_paths.run_dir / "settings_used.json")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not copy settings file: {exc}")

    waypoints = extract_waypoints(experiment)
    metrics_tracker = MetricsTracker(run_paths.run_dir, update_interval=0)

    results: list[dict[str, Any]] = []

    with sim_context(mode=args.mode, settings_path=args.settings):
        client = get_airsim_client()
        client.enableApiControl(True)
        client.armDisarm(True)

        perception = build_perception(
            client,
            args.camera_name,
            args.detector_model,
            async_mode=not args.perception_sync,
            enable_segmentation=not args.disable_segmentation,
        )

        detector_obj = getattr(perception, "_detector", None)
        enable_rgb = detector_obj is not None and detector_obj.__class__.__name__ != "DummyDetector"

        simulator = AirSimSimulatorAdapter(
            client,
            horizon=experiment.horizon,
            enable_rgb=enable_rgb,
        )

        env = SingleAgentAirSimEnv(
            experiment,
            simulator=simulator,
            perception=perception,
        )

        model = SAC.load(args.model, device="auto")

        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            metrics_tracker.start_episode(episode, waypoints=waypoints)

            done = False
            total_reward = 0.0
            steps = 0

            last_info = info
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, step_info = env.step(action)

                telemetry = step_info.get("telemetry", {})
                reward_components = step_info.get("reward_components", {})
                action_dict = step_info.get("applied_action", {})

                metrics_tracker.log_step(
                    step=steps,
                    reward=reward,
                    action=action_dict,
                    telemetry=telemetry,
                    reward_components=reward_components,
                    command=None,
                    next_telemetry=None,
                )

                total_reward += reward
                steps += 1
                done = terminated or truncated
                last_info = step_info

            success = bool(last_info.get("done_flags", {}).get("goal_reached", False))
            metrics_tracker.finish_episode(completed_successfully=success)

            results.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps,
                    "goal_reached": success,
                    "collision": bool(last_info.get("done_flags", {}).get("collision", False)),
                }
            )
            _log_episode(artifact_manager, episode, total_reward, steps, last_info)
            print(
                f"Episode {episode}/{args.episodes}: reward={total_reward:.2f}, steps={steps}, "
                f"goal_reached={'yes' if success else 'no'}"
            )

        env.close()

    success_count = sum(1 for item in results if item.get("goal_reached"))
    collision_count = sum(1 for item in results if item.get("collision"))
    avg_reward = sum(item["reward"] for item in results) / max(len(results), 1)

    summary = {
        "episodes": len(results),
        "success_rate": success_count / max(len(results), 1),
        "collision_rate": collision_count / max(len(results), 1),
        "average_reward": avg_reward,
    }
    artifact_manager.write_json("evaluation_summary.json", summary)

    print("\n✅ Evaluation completed!")
    print(f"Success rate: {summary['success_rate'] * 100:.1f}%")
    print(f"Collision rate: {summary['collision_rate'] * 100:.1f}%")
    print(f"Average reward: {summary['average_reward']:.2f}")
    print(f"Artifacts saved to: {run_paths.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
