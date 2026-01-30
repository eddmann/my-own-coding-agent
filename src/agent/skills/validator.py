"""Skill validation - validates skill names and descriptions per Agent Skills spec."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Validation constants
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$")


@dataclass(slots=True, frozen=True)
class ValidationError:
    """A validation error with details."""

    field: str
    message: str
    value: str | None = None


@dataclass(slots=True, frozen=True)
class ValidationResult:
    """Result of skill validation."""

    valid: bool
    errors: list[ValidationError]

    @classmethod
    def success(cls) -> ValidationResult:
        """Create a successful validation result."""
        return cls(valid=True, errors=[])

    @classmethod
    def failure(cls, errors: list[ValidationError]) -> ValidationResult:
        """Create a failed validation result."""
        return cls(valid=False, errors=errors)


def validate_skill_name(name: str, parent_dir_name: str | None = None) -> list[ValidationError]:
    """Validate a skill name.

    Rules:
    - Must contain only lowercase a-z, 0-9, and hyphens
    - Must not start or end with a hyphen
    - Maximum 64 characters
    - If parent_dir_name is provided, name must match it

    Args:
        name: The skill name to validate
        parent_dir_name: Optional parent directory name that must match

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[ValidationError] = []

    # Check empty
    if not name:
        errors.append(ValidationError(field="name", message="Name is required"))
        return errors

    # Check length
    if len(name) > MAX_NAME_LENGTH:
        errors.append(
            ValidationError(
                field="name",
                message=f"Name must be at most {MAX_NAME_LENGTH} characters (got {len(name)})",
                value=name,
            )
        )

    # Check pattern
    if not NAME_PATTERN.match(name):
        errors.append(
            ValidationError(
                field="name",
                message="Name must contain only lowercase a-z, 0-9, and hyphens, "
                "and must not start or end with a hyphen",
                value=name,
            )
        )

    # Check directory match
    if parent_dir_name and name != parent_dir_name:
        errors.append(
            ValidationError(
                field="name",
                message=f"Skill name '{name}' must match parent directory name '{parent_dir_name}'",
                value=name,
            )
        )

    return errors


def validate_skill_description(description: str | None) -> list[ValidationError]:
    """Validate a skill description.

    Rules:
    - Required (cannot be empty)
    - Maximum 1024 characters

    Args:
        description: The skill description to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[ValidationError] = []

    # Check required
    if not description or not description.strip():
        errors.append(ValidationError(field="description", message="Description is required"))
        return errors

    # Check length
    if len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            ValidationError(
                field="description",
                message=f"Description must be at most {MAX_DESCRIPTION_LENGTH} characters "
                f"(got {len(description)})",
                value=description[:100] + "..." if len(description) > 100 else description,
            )
        )

    return errors


def validate_skill(
    name: str,
    description: str | None,
    readme_path: Path | None = None,
) -> ValidationResult:
    """Validate a complete skill definition.

    Args:
        name: Skill name
        description: Skill description
        readme_path: Optional path to README for directory name matching

    Returns:
        ValidationResult with valid flag and any errors
    """
    errors: list[ValidationError] = []

    # Get parent directory name for matching
    parent_dir_name = readme_path.parent.name if readme_path else None

    # Validate name
    errors.extend(validate_skill_name(name, parent_dir_name))

    # Validate description
    errors.extend(validate_skill_description(description))

    if errors:
        return ValidationResult.failure(errors)
    return ValidationResult.success()
