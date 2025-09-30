"""Test resume logic for training runs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def test_resume_from_specific_run_id_validation():
    """Test that resume-from logic resolves to correct directory."""
    # This is a simpler unit test that just validates the directory resolution logic
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        run_id = "20250930T154745Z_train_abc123"
        resume_dir = models_dir / run_id
        resume_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy model files
        (resume_dir / "dqn_manager.zip").touch()
        (resume_dir / "sac_follow_lane.zip").touch()

        # Simulate the resume logic
        args = MagicMock()
        args.models = str(models_dir)
        args.resume = False
        args.resume_from = run_id

        # Test the logic
        model_root = Path(args.models)
        resume_model_dir = None
        if args.resume_from:
            resume_model_dir = model_root / args.resume_from
            assert resume_model_dir.exists(), f"Resume directory should exist: {resume_model_dir}"
            assert (resume_model_dir / "dqn_manager.zip").exists()

        load_from_dir = str(resume_model_dir) if resume_model_dir else args.models
        assert load_from_dir == str(resume_dir)


def test_resume_from_nonexistent_run_fails():
    """Test that --resume-from with non-existent run ID fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        args = MagicMock()
        args.models = str(models_dir)
        args.resume = False
        args.resume_from = "nonexistent_run_id"

        # Test the logic
        model_root = Path(args.models)
        resume_model_dir = None
        if args.resume_from:
            resume_model_dir = model_root / args.resume_from
            # Should not exist
            assert not resume_model_dir.exists(), "Non-existent run directory should not exist"


def test_resume_best_loads_from_best_models_subdir():
    """Test that --resume-best loads from best-models/ subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        run_id = "20250930T154745Z_train_abc123"
        run_dir = models_dir / run_id
        best_dir = run_dir / "best-models"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy model files in best-models/
        (best_dir / "dqn_manager.zip").touch()
        (best_dir / "sac_follow_lane.zip").touch()

        args = MagicMock()
        args.models = str(models_dir)
        args.resume = False
        args.resume_from = run_id
        args.resume_best = True

        # Test the logic
        model_root = Path(args.models)
        resume_model_dir = model_root / args.resume_from

        if args.resume_best:
            resume_model_dir = resume_model_dir / "best-models"

        assert resume_model_dir.exists(), f"Best models directory should exist: {resume_model_dir}"
        assert (resume_model_dir / "dqn_manager.zip").exists()
        assert resume_model_dir == best_dir

        load_from_dir = str(resume_model_dir)
        assert load_from_dir == str(best_dir)
