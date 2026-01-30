"""Behavior tests for context file discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.core.context_loader import load_all_context, load_ancestor_context

if TYPE_CHECKING:
    from pathlib import Path


def write_ctx(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_load_all_context_orders_sources(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = home / "work" / "project"
    parent = project.parent

    monkeypatch.setenv("HOME", str(home))

    project_agents = project / "AGENTS.md"
    parent_agents = parent / "AGENTS.md"
    explicit = temp_dir / "explicit.md"

    write_ctx(project_agents, "project")
    write_ctx(parent_agents, "parent")
    write_ctx(explicit, "explicit")

    contexts = load_all_context(
        cwd=project,
        explicit_paths=[explicit],
        include_ancestors=True,
    )

    assert contexts[0].content == "project"
    assert contexts[1].content == "parent"
    assert contexts[2].content == "explicit"


def test_load_ancestor_context_closest_wins(temp_dir, monkeypatch):
    home = temp_dir / "home"
    project = home / "work" / "project" / "sub"
    parent = project.parent
    grandparent = parent.parent

    monkeypatch.setenv("HOME", str(home))

    write_ctx(parent / "AGENTS.md", "parent")
    write_ctx(grandparent / "AGENTS.md", "grandparent")

    contexts = load_ancestor_context(cwd=project)

    assert len(contexts) == 1
    assert contexts[0].content == "parent"
