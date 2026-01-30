"""Skill loader - discovers and loads skills from directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.skills.skill import Skill, SkillSource
from agent.skills.validator import ValidationResult, validate_skill

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True)
class SkillLoadError:
    """Error encountered while loading a skill."""

    path: Path
    error: str
    validation_result: ValidationResult | None = None


@dataclass(slots=True)
class SkillCollision:
    """Collision between two skills with the same name."""

    name: str
    kept_path: Path
    kept_source: SkillSource
    discarded_path: Path
    discarded_source: SkillSource


@dataclass(slots=True)
class SkillLoadDiagnostics:
    """Diagnostics from skill loading."""

    loaded_count: int = 0
    error_count: int = 0
    collision_count: int = 0
    errors: list[SkillLoadError] = field(default_factory=list)
    collisions: list[SkillCollision] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Check if there were any errors or collisions."""
        return self.error_count > 0 or self.collision_count > 0


# Priority order for source resolution (higher index = higher priority)
SOURCE_PRIORITY = {
    SkillSource.USER: 0,
    SkillSource.PATH: 1,
    SkillSource.PROJECT: 2,  # Project skills override all others
}


class SkillLoader:
    """Loads skills from directories with validation and collision detection.

    Default search directories (in priority order, lowest to highest):
    1. ~/.agent/skills/ (user skills)
    2. Custom paths from config
    3. .agent/skills/ (project skills - highest priority)
    """

    __slots__ = ("_dirs", "_skills", "_diagnostics")

    def __init__(self, dirs: list[tuple[Path, SkillSource]] | None = None) -> None:
        """Initialize the skill loader.

        Args:
            dirs: List of (directory, source) tuples to search for skills
        """
        self._dirs: list[tuple[Path, SkillSource]] = dirs or []
        self._skills: dict[str, Skill] | None = None
        self._diagnostics: SkillLoadDiagnostics | None = None

    @classmethod
    def with_defaults(
        cls, extra_dirs: list[Path] | None = None, cwd: Path | None = None
    ) -> SkillLoader:
        """Create a loader with default directories.

        Default directories:
        - ~/.agent/skills/ (user skills)
        - .agent/skills/ (project skills)
        - Any extra_dirs provided (custom paths)

        Args:
            extra_dirs: Additional directories to search
            cwd: Current working directory (defaults to Path.cwd())

        Returns:
            Configured SkillLoader
        """
        dirs: list[tuple[Path, SkillSource]] = []
        cwd = cwd or Path.cwd()

        # User skills (lowest priority)
        user_dir = Path.home() / ".agent" / "skills"
        if user_dir.exists():
            dirs.append((user_dir, SkillSource.USER))

        # Custom paths (medium priority)
        if extra_dirs:
            for path in extra_dirs:
                if path.exists() and path.is_dir():
                    dirs.append((path, SkillSource.PATH))

        # Project skills (highest priority)
        project_dir = cwd / ".agent" / "skills"
        if project_dir.exists():
            dirs.append((project_dir, SkillSource.PROJECT))

        return cls(dirs)

    def add_directory(self, path: Path, source: SkillSource = SkillSource.PATH) -> None:
        """Add a directory to search for skills.

        Args:
            path: Directory path
            source: Source type for skills from this directory
        """
        if path.exists() and path.is_dir():
            self._dirs.append((path, source))
            self._skills = None  # Invalidate cache
            self._diagnostics = None

    @property
    def skills(self) -> dict[str, Skill]:
        """Get all loaded skills (lazy loaded)."""
        if self._skills is None:
            self._load_all()
        return self._skills  # type: ignore

    @property
    def diagnostics(self) -> SkillLoadDiagnostics:
        """Get diagnostics from the last load operation."""
        if self._diagnostics is None:
            self._load_all()
        return self._diagnostics  # type: ignore

    def _load_all(self) -> None:
        """Load all skills from configured directories with validation."""
        self._skills = {}
        self._diagnostics = SkillLoadDiagnostics()

        for directory, source in self._dirs:
            if not directory.exists():
                continue

            # Look for SKILL.md or README.md in immediate subdirectories
            for subdir in directory.iterdir():
                if not subdir.is_dir():
                    continue

                # Try SKILL.md first, then README.md
                skill_file = subdir / "SKILL.md"
                if not skill_file.exists():
                    skill_file = subdir / "README.md"
                if not skill_file.exists():
                    continue

                try:
                    skill = Skill.from_markdown(skill_file, source=source)

                    # Validate the skill
                    validation = validate_skill(
                        skill.name,
                        skill.description,
                        skill_file,
                    )

                    if not validation.valid:
                        self._diagnostics.errors.append(
                            SkillLoadError(
                                path=skill_file,
                                error="; ".join(e.message for e in validation.errors),
                                validation_result=validation,
                            )
                        )
                        self._diagnostics.error_count += 1
                        continue

                    # Handle collisions
                    if skill.name in self._skills:
                        existing = self._skills[skill.name]

                        # Compare priorities
                        existing_priority = SOURCE_PRIORITY[existing.source]
                        new_priority = SOURCE_PRIORITY[skill.source]

                        if new_priority > existing_priority:
                            # New skill wins
                            self._diagnostics.collisions.append(
                                SkillCollision(
                                    name=skill.name,
                                    kept_path=skill.readme_path,
                                    kept_source=skill.source,
                                    discarded_path=existing.readme_path,
                                    discarded_source=existing.source,
                                )
                            )
                            self._diagnostics.collision_count += 1
                            self._skills[skill.name] = skill
                        else:
                            # Existing skill wins
                            self._diagnostics.collisions.append(
                                SkillCollision(
                                    name=skill.name,
                                    kept_path=existing.readme_path,
                                    kept_source=existing.source,
                                    discarded_path=skill.readme_path,
                                    discarded_source=skill.source,
                                )
                            )
                            self._diagnostics.collision_count += 1
                    else:
                        self._skills[skill.name] = skill
                        self._diagnostics.loaded_count += 1

                except Exception as e:
                    self._diagnostics.errors.append(SkillLoadError(path=skill_file, error=str(e)))
                    self._diagnostics.error_count += 1

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self.skills.get(name)

    def find(self, name: str) -> Skill | None:
        """Find a skill by name (alias for get)."""
        return self.get(name)

    def get_context(self, skill_name: str) -> str | None:
        """Get skill README content for injection into system prompt."""
        skill = self.get(skill_name)
        if skill:
            return skill.get_context()
        return None

    def list_skills(self) -> list[str]:
        """List all available skill names."""
        return list(self.skills.keys())

    def get_invocable_skills(self) -> list[Skill]:
        """Get skills that can be invoked by the model.

        Returns only skills where disable_model_invocation is False.
        """
        return [s for s in self.skills.values() if not s.disable_model_invocation]

    def __iter__(self) -> Iterator[Skill]:
        return iter(self.skills.values())

    def __len__(self) -> int:
        return len(self.skills)
