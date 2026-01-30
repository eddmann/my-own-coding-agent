"""Skill definition - markdown with YAML frontmatter."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class SkillSource(StrEnum):
    """Source location of a skill."""

    USER = "user"  # ~/.agent/skills/
    PROJECT = "project"  # .agent/skills/
    PATH = "path"  # Custom path from config


@dataclass(slots=True)
class Skill:
    """A skill loaded from a README.md or SKILL.md file with YAML frontmatter."""

    name: str
    description: str
    readme_path: Path
    readme_content: str
    base_dir: Path
    source: SkillSource
    metadata: dict[str, object] = field(default_factory=dict)
    disable_model_invocation: bool = False

    @classmethod
    def from_markdown(
        cls,
        readme_path: Path,
        source: SkillSource = SkillSource.PATH,
    ) -> Self:
        """Load a skill from a README.md or SKILL.md file.

        Expected format:
        ```markdown
        ---
        name: skill-name
        description: Brief description
        disable_model_invocation: false  # optional
        ---

        # Skill Title

        Rest of the README content...
        ```

        Args:
            readme_path: Path to the README.md or SKILL.md file
            source: Where this skill was loaded from

        Returns:
            Loaded Skill instance
        """
        content = readme_path.read_text(encoding="utf-8")
        metadata, body = cls._parse_frontmatter(content)

        raw_name = metadata.get("name")
        raw_description = metadata.get("description")
        raw_disable = metadata.get("disable_model_invocation")

        return cls(
            name=raw_name if isinstance(raw_name, str) else readme_path.parent.name,
            description=raw_description if isinstance(raw_description, str) else "",
            readme_path=readme_path,
            readme_content=body,
            base_dir=readme_path.parent,
            source=source,
            metadata=metadata,
            disable_model_invocation=raw_disable if isinstance(raw_disable, bool) else False,
        )

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
        """Parse YAML frontmatter from markdown content.

        Returns:
            Tuple of (metadata dict, body content)
        """
        if not content.startswith("---"):
            return {}, content

        # Find the closing ---
        lines = content.split("\n")
        end_idx = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end_idx = i
                break

        if end_idx == -1:
            return {}, content

        # Parse YAML frontmatter
        frontmatter = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1 :]).lstrip()

        try:
            loaded = yaml.safe_load(frontmatter)
            metadata = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError:
            metadata = {}

        return metadata, body

    def get_context(self) -> str:
        """Get the skill content for injection into the system prompt."""
        return f"## Skill: {self.name}\n\n{self.readme_content}"

    def read_body(self) -> str:
        """Read the current skill file and return the body without frontmatter."""
        content = self.readme_path.read_text(encoding="utf-8")
        _, body = self._parse_frontmatter(content)
        return body.strip()
