"""Prompt template loader - discovers and loads templates from directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.prompts.template import PromptTemplate, TemplateSource

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True)
class TemplateLoadError:
    """Error encountered while loading a template."""

    path: Path
    error: str


@dataclass(slots=True)
class TemplateLoadDiagnostics:
    """Diagnostics from template loading."""

    loaded_count: int = 0
    error_count: int = 0
    collision_count: int = 0
    errors: list[TemplateLoadError] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Check if there were any errors."""
        return self.error_count > 0


# Priority order for source resolution (higher index = higher priority)
SOURCE_PRIORITY = {
    TemplateSource.USER: 0,
    TemplateSource.PATH: 1,
    TemplateSource.PROJECT: 2,  # Project templates override all others
}


class PromptTemplateLoader:
    """Loads prompt templates from directories.

    Default search directories (in priority order, lowest to highest):
    1. ~/.agent/prompts/ (user templates)
    2. Custom paths from config
    3. .agent/prompts/ (project templates - highest priority)
    """

    __slots__ = ("_dirs", "_templates", "_diagnostics")

    def __init__(self, dirs: list[tuple[Path, TemplateSource]] | None = None) -> None:
        """Initialize the template loader.

        Args:
            dirs: List of (directory, source) tuples to search for templates
        """
        self._dirs: list[tuple[Path, TemplateSource]] = dirs or []
        self._templates: dict[str, PromptTemplate] | None = None
        self._diagnostics: TemplateLoadDiagnostics | None = None

    @classmethod
    def with_defaults(
        cls,
        extra_dirs: list[Path] | None = None,
        cwd: Path | None = None,
    ) -> PromptTemplateLoader:
        """Create a loader with default directories.

        Default directories:
        - ~/.agent/prompts/ (user templates)
        - .agent/prompts/ (project templates)
        - Any extra_dirs provided (custom paths)

        Args:
            extra_dirs: Additional directories to search
            cwd: Current working directory (defaults to Path.cwd())

        Returns:
            Configured PromptTemplateLoader
        """
        dirs: list[tuple[Path, TemplateSource]] = []
        cwd = cwd or Path.cwd()

        # User templates (lowest priority)
        user_dir = Path.home() / ".agent" / "prompts"
        if user_dir.exists():
            dirs.append((user_dir, TemplateSource.USER))

        # Custom paths (medium priority)
        if extra_dirs:
            for path in extra_dirs:
                if path.exists() and path.is_dir():
                    dirs.append((path, TemplateSource.PATH))

        # Project templates (highest priority)
        project_dir = cwd / ".agent" / "prompts"
        if project_dir.exists():
            dirs.append((project_dir, TemplateSource.PROJECT))

        return cls(dirs)

    def add_directory(self, path: Path, source: TemplateSource = TemplateSource.PATH) -> None:
        """Add a directory to search for templates.

        Args:
            path: Directory path
            source: Source type for templates from this directory
        """
        if path.exists() and path.is_dir():
            self._dirs.append((path, source))
            self._templates = None  # Invalidate cache
            self._diagnostics = None

    @property
    def templates(self) -> dict[str, PromptTemplate]:
        """Get all loaded templates (lazy loaded)."""
        if self._templates is None:
            self._load_all()
        return self._templates  # type: ignore

    @property
    def diagnostics(self) -> TemplateLoadDiagnostics:
        """Get diagnostics from the last load operation."""
        if self._diagnostics is None:
            self._load_all()
        return self._diagnostics  # type: ignore

    def _load_all(self) -> None:
        """Load all templates from configured directories."""
        self._templates = {}
        self._diagnostics = TemplateLoadDiagnostics()

        for directory, source in self._dirs:
            if not directory.exists():
                continue

            # Look for .md files in the directory
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() != ".md":
                    continue

                try:
                    template = PromptTemplate.from_markdown(file_path, source=source)

                    # Handle collisions
                    if template.name in self._templates:
                        existing = self._templates[template.name]

                        # Compare priorities
                        existing_priority = SOURCE_PRIORITY[existing.source]
                        new_priority = SOURCE_PRIORITY[template.source]

                        if new_priority > existing_priority:
                            # New template wins
                            self._templates[template.name] = template
                            self._diagnostics.collision_count += 1
                        else:
                            # Existing template wins
                            self._diagnostics.collision_count += 1
                    else:
                        self._templates[template.name] = template
                        self._diagnostics.loaded_count += 1

                except Exception as e:
                    self._diagnostics.errors.append(TemplateLoadError(path=file_path, error=str(e)))
                    self._diagnostics.error_count += 1

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name."""
        return self.templates.get(name)

    def find(self, name: str) -> PromptTemplate | None:
        """Find a template by name (alias for get)."""
        return self.get(name)

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def __iter__(self) -> Iterator[PromptTemplate]:
        return iter(self.templates.values())

    def __len__(self) -> int:
        return len(self.templates)
