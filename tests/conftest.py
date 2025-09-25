from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from utils.airsim_runner import SimulatorSession, airsim_session


@pytest.fixture(scope="session")
def airsim_session_fixture(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimulatorSession]:
    settings_path = tmp_path_factory.mktemp("airsim") / "settings.json"
    default_settings = Path("settings.json")
    if default_settings.exists():
        settings_path.write_text(default_settings.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        settings_path.write_text("{}", encoding="utf-8")

    with airsim_session(mode="headless", settings_path=str(settings_path)) as session:
        yield session
