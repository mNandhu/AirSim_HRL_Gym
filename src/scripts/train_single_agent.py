"""Train a single SAC agent in AirSim using the shared environment stack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

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
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Missing required dependency: {exc}. Install with: pip install stable-baselines3"
    ) from exc


class MetricsCallback(BaseCallback):
    """Bridge SB3 training loop with repo metrics tracker and artifact logging."""

    def __init__(
        self,
        *,
        metrics_tracker: MetricsTracker,
        artifact_manager: ArtifactManager,
        model_dir: Path,
        save_interval: int,
        waypoints: Sequence[tuple[float, float]],
    ) -> None:
        super().__init__()
        self._tracker = metrics_tracker
        self._artifact_manager = artifact_manager
        self._model_dir = model_dir
        self._save_interval = max(save_interval, 1)
        self._waypoints = list(waypoints)

        self._episode_index = 0
        self._episode_reward = 0.0
        self._episode_step = 0
        self._best_reward = float("-inf")
        self._history: list[dict[str, Any]] = []
        self._best_model_dir = model_dir / "best-models"

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    @property
    def best_reward(self) -> float:
        return self._best_reward

    @property
    def best_model_dir(self) -> Path:
        return self._best_model_dir

    def _on_training_start(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._best_model_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if not infos or not rewards:
            return True

        info = infos[0] or {}
        reward = float(rewards[0])
        done = bool(dones[0])

        if self._episode_step == 0:
            self._tracker.start_episode(self._episode_index + 1, waypoints=self._waypoints)

        telemetry = info.get("telemetry", {})
        reward_components = info.get("reward_components", {})
        action = info.get("applied_action", {})

        self._tracker.log_step(
            step=self._episode_step,
            reward=reward,
            action=action,
            telemetry=telemetry,
            reward_components=reward_components,
            command=None,
            next_telemetry=None,
        )

        self._episode_reward += reward
        self._episode_step += 1

        if done:
            flags = info.get("done_flags", {})
            success = bool(flags.get("goal_reached", False))
            collision = bool(flags.get("collision", False))
            self._tracker.finish_episode(completed_successfully=success)

            stats = {
                "episode": self._episode_index + 1,
                "reward": self._episode_reward,
                "steps": self._episode_step,
                "goal_reached": success,
                "collision": collision,
                "config_hash": info.get("config_hash"),
            }
            self._history.append(stats)
            self._artifact_manager.append_jsonl("training_log.jsonl", stats)

            if (self._episode_index + 1) % self._save_interval == 0:
                self._save_checkpoint(
                    self._model_dir / f"sac_episode_{self._episode_index + 1:04d}.zip"
                )

            if self._episode_reward >= self._best_reward:
                self._best_reward = self._episode_reward
                self._save_checkpoint(self._best_model_dir / "sac_single_agent.zip")

            self._episode_index += 1
            self._episode_reward = 0.0
            self._episode_step = 0

        return True

    def _on_training_end(self) -> None:
        if self._episode_step > 0:
            self._tracker.finish_episode(completed_successfully=False)

    def _save_checkpoint(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.model is not None:
                self.model.save(str(path))
        except Exception as exc:  # pragma: no cover - non-critical
            print(f"Warning: failed to save checkpoint to {path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single SAC agent in AirSim")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--total-timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--max-steps", type=int, help="Optional per-episode step cap")
    parser.add_argument("--models", default="models/single_agent", help="Model directory root")
    parser.add_argument("--resume", type=str, help="Path to SAC checkpoint to resume")
    parser.add_argument("--output", default="artifacts", help="Artifact directory root")
    parser.add_argument("--settings", default="settings.json", help="AirSim settings file")
    parser.add_argument("--mode", choices=["gui", "headless"], default="gui")
    parser.add_argument("--detector-model", default="yolov8n")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--disable-segmentation", action="store_true")
    parser.add_argument("--perception-sync", action="store_true")
    parser.add_argument("--metrics-update-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Optional random seed for SAC")
    parser.add_argument("--max-target-speed", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    experiment = load_experiment(args.config)
    artifact_manager = ArtifactManager(args.output)
    run_paths = artifact_manager.start_run(f"train_sac_{experiment.id}")

    model_root = Path(args.models)
    model_root.mkdir(parents=True, exist_ok=True)
    run_tag = run_paths.run_dir.name
    model_dir = model_root / run_tag

    settings_src = os.environ.get("AIRSIM_SETTINGS_PATH", args.settings)

    # Persist experiment metadata for reproducibility
    artifact_manager.write_json(
        "experiment_config.json",
        json.loads(experiment.model_dump_json()),
    )
    hparams = {
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "gamma": args.gamma,
            "tau": args.tau,
            "ent_coef": args.ent_coef,
        },
        "environment": {
            "target_speed_range": [0.0, args.max_target_speed],
            "metrics_update_interval": args.metrics_update_interval,
        },
    }
    artifact_manager.write_json("hparams.json", hparams)

    try:
        src_path = Path(settings_src)
        if src_path.exists():
            shutil.copy2(src_path, run_paths.run_dir / "settings_used.json")
    except Exception as exc:  # pragma: no cover - non-critical
        print(f"Warning: could not copy settings file: {exc}")

    total_timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else args.episodes * (args.max_steps or experiment.horizon)
    )

    metrics_tracker = MetricsTracker(
        run_paths.run_dir, update_interval=args.metrics_update_interval
    )

    waypoints = extract_waypoints(experiment)

    # Build simulator context and environment
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

        base_env = SingleAgentAirSimEnv(
            experiment,
            simulator=simulator,
            perception=perception,
            target_speed_range=(0.0, args.max_target_speed),
        )
        env = Monitor(base_env)

        tensorboard_dir = run_paths.logs_dir / "sb3" / "single_agent"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        if args.resume:
            model = SAC.load(
                args.resume,
                env=env,
                tensorboard_log=str(tensorboard_dir),
                device=args.device,
            )
            model.set_env(env)
        else:
            model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                gamma=args.gamma,
                tau=args.tau,
                ent_coef=args.ent_coef,
                train_freq=(1, "step"),
                gradient_steps=1,
                tensorboard_log=str(tensorboard_dir),
                verbose=1,
                seed=args.seed,
                device=args.device,
            )

        callback = MetricsCallback(
            metrics_tracker=metrics_tracker,
            artifact_manager=artifact_manager,
            model_dir=model_dir,
            save_interval=args.save_interval,
            waypoints=waypoints,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
        )

        # Always persist the final model snapshot
        final_model_path = model_dir / "sac_single_agent.zip"
        model.save(str(final_model_path))

        env.close()

    stats_payload = {
        "episodes": args.episodes,
        "total_timesteps": total_timesteps,
        "best_reward": callback.best_reward,
        "history": callback.history,
    }
    artifact_manager.write_json("training_stats.json", stats_payload)

    summary = metrics_tracker.get_summary_stats()
    if summary:
        artifact_manager.write_json("training_summary.json", summary)

    print("\n✅ Training completed!")
    print(f"Models saved to: {model_dir}")
    if callback.best_reward > float("-inf"):
        print(f"Best reward {callback.best_reward:.2f} saved to: {callback.best_model_dir}")
    print(f"Logs saved to: {run_paths.run_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
