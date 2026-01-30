"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Temporary directory for filesystem-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
