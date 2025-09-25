import json
from pathlib import Path

from utils.artifacts import ArtifactManager


def test_artifact_manager_creates_run_directory(tmp_path: Path):
    manager = ArtifactManager(tmp_path)
    paths = manager.start_run("episode-1")

    assert paths.run_dir.exists()
    log = manager.write_json("metrics.json", {"reward": 1.23})
    assert json.loads(log.read_text(encoding="utf-8"))["reward"] == 1.23
