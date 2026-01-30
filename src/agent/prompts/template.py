"""Prompt template definition - markdown files with YAML frontmatter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class TemplateSource(StrEnum):
    """Source location of a prompt template."""

    USER = "user"  # ~/.agent/prompts/
    PROJECT = "project"  # .agent/prompts/
    PATH = "path"  # Custom path from config


@dataclass(slots=True)
class PromptTemplate:
    """A prompt template loaded from a markdown file.

    Templates support argument substitution:
    - $1, $2, ... - Positional arguments
    - $@ or $ARGUMENTS - All arguments joined with spaces
    - ${@:N} - Arguments from position N onwards
    - ${@:N:L} - L arguments starting from position N

    Usage:
        /template-name arg1 arg2 arg3
    """

    name: str
    description: str
    content: str
    source: TemplateSource
    file_path: Path
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_markdown(
        cls,
        file_path: Path,
        source: TemplateSource = TemplateSource.PATH,
    ) -> Self:
        """Load a prompt template from a markdown file.

        Expected format:
        ```markdown
        ---
        name: template-name  # optional, defaults to filename without extension
        description: Brief description of what this template does
        ---

        Template content here with $1 $2 $@ substitution patterns...
        ```

        Args:
            file_path: Path to the template file
            source: Where this template was loaded from

        Returns:
            Loaded PromptTemplate instance
        """
        raw_content = file_path.read_text(encoding="utf-8")
        metadata, body = cls._parse_frontmatter(raw_content)

        # Default name from filename without extension
        default_name = file_path.stem

        raw_name = metadata.get("name")
        raw_description = metadata.get("description")

        return cls(
            name=raw_name if isinstance(raw_name, str) else default_name,
            description=raw_description if isinstance(raw_description, str) else "",
            content=body,
            source=source,
            file_path=file_path,
            metadata=metadata,
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
