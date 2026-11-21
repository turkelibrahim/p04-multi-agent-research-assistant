"""Ortak pytest test fikstürleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from p04_research.config import AppConfig
from p04_research.db.repository import ResearchRepository


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_p04.db"


@pytest.fixture()
def app_config(tmp_db_path: Path) -> AppConfig:
    return AppConfig(
        openai_api_key=None,
        tavily_api_key=None,
        db_path=tmp_db_path,
    )


@pytest.fixture()
def repo(tmp_db_path: Path) -> ResearchRepository:
    repository = ResearchRepository(tmp_db_path)
    repository.ensure_initialized()
    return repository
