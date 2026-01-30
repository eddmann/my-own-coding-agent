"""Behavior tests for extension loading."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from agent.extensions.api import ExtensionAPI
from agent.extensions.loader import ExtensionLoader

if TYPE_CHECKING:
    from pathlib import Path


def write_ext(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


@pytest.mark.asyncio
async def test_extension_loader_errors_for_missing_file(temp_dir):
    api = ExtensionAPI()

    error = await ExtensionLoader.load(temp_dir / "missing.py", api)

    assert error is not None
    assert "not found" in error


@pytest.mark.asyncio
async def test_extension_loader_rejects_non_py(temp_dir):
    api = ExtensionAPI()
    path = temp_dir / "ext.txt"
    path.write_text("x")

    error = await ExtensionLoader.load(path, api)

    assert error is not None
    assert "Python file" in error


@pytest.mark.asyncio
async def test_extension_loader_requires_setup(temp_dir):
    api = ExtensionAPI()
    path = temp_dir / "ext.py"
    write_ext(path, "x = 1")

    error = await ExtensionLoader.load(path, api)

    assert error is not None
    assert "setup" in error


@pytest.mark.asyncio
async def test_extension_loader_directory_skips_private_files(temp_dir):
    api = ExtensionAPI()
    directory = temp_dir / "exts"
    directory.mkdir()

    write_ext(directory / "_skip.py", "raise RuntimeError('skip')")
    write_ext(
        directory / "ok.py",
        """
        def setup(api):
            pass
        """,
    )

    errors = await ExtensionLoader.load_directory(directory, api)

    assert errors == []
