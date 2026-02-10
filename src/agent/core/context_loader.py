"""Context file loader - loads AGENTS.md, CLAUDE.md from project and ancestors."""

from __future__ import annotations

from pathlib import Path

from agent.core.prompt_builder import ContextFile

# Context file names to search for (in priority order)
CONTEXT_FILE_NAMES = ["AGENTS.md", "CLAUDE.md"]


def _load_context_file(path: Path, source: str = "explicit") -> ContextFile | None:
    """Load a single context file.

    Args:
        path: Path to the context file
        source: Source identifier ("project", "ancestor", "explicit")

    Returns:
        ContextFile if loaded successfully, None otherwise
    """
    if not path.exists() or not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8")
        return ContextFile(path=path, content=content, source=source)
    except Exception:
        return None


def _load_project_context(cwd: Path | None = None) -> list[ContextFile]:
    """Load context files from the current project directory.

    Looks for AGENTS.md and CLAUDE.md in cwd.

    Args:
        cwd: Current working directory (defaults to Path.cwd())

    Returns:
        List of loaded context files
    """
    cwd = cwd or Path.cwd()
    context_files: list[ContextFile] = []

    for name in CONTEXT_FILE_NAMES:
        path = cwd / name
        if ctx := _load_context_file(path, source="project"):
            context_files.append(ctx)

    return context_files


def load_ancestor_context(cwd: Path | None = None, stop_at_home: bool = True) -> list[ContextFile]:
    """Load context files from ancestor directories.

    Walks up the directory tree looking for AGENTS.md and CLAUDE.md files.
    Stops at the user's home directory by default.

    Args:
        cwd: Starting directory (defaults to Path.cwd())
        stop_at_home: Whether to stop at the home directory

    Returns:
        List of loaded context files (ordered from closest to farthest ancestor)
    """
    cwd = cwd or Path.cwd()
    home = Path.home()
    context_files: list[ContextFile] = []
    seen_names: set[str] = set()

    # Start from parent (don't include cwd, use _load_project_context for that)
    current = cwd.parent

    while current != current.parent:  # Stop at filesystem root
        if stop_at_home and current == home:
            break

        for name in CONTEXT_FILE_NAMES:
            # Only load each filename once (closest wins)
            if name in seen_names:
                continue

            path = current / name
            if ctx := _load_context_file(path, source="ancestor"):
                context_files.append(ctx)
                seen_names.add(name)

        current = current.parent

    return context_files


def _load_explicit_context(paths: list[Path]) -> list[ContextFile]:
    """Load explicitly specified context files.

    Args:
        paths: List of paths to load

    Returns:
        List of loaded context files
    """
    context_files: list[ContextFile] = []

    for path in paths:
        if ctx := _load_context_file(path, source="explicit"):
            context_files.append(ctx)

    return context_files


def load_all_context(
    cwd: Path | None = None,
    explicit_paths: list[Path] | None = None,
    include_ancestors: bool = True,
) -> list[ContextFile]:
    """Load all context files from project, ancestors, and explicit paths.

    Loading order (all included in result):
    1. Project context (cwd)
    2. Ancestor context (parent directories up to home)
    3. Explicit paths (additional files specified in config)

    Args:
        cwd: Current working directory
        explicit_paths: Additional paths to load
        include_ancestors: Whether to include ancestor directories

    Returns:
        List of all loaded context files
    """
    context_files: list[ContextFile] = []

    # Project context
    context_files.extend(_load_project_context(cwd))

    # Ancestor context
    if include_ancestors:
        context_files.extend(load_ancestor_context(cwd))

    # Explicit paths
    if explicit_paths:
        context_files.extend(_load_explicit_context(explicit_paths))

    return context_files
