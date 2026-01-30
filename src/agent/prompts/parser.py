"""Argument parsing and substitution for prompt templates."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedCommand:
    """Parsed slash command with template name and arguments."""

    template_name: str
    arguments: list[str]
    raw_args: str


def parse_command(input_text: str) -> ParsedCommand | None:
    """Parse a slash command from user input.

    Format: /template-name arg1 arg2 "arg with spaces" ...

    Args:
        input_text: User input text

    Returns:
        ParsedCommand if input is a valid slash command, None otherwise
    """
    input_text = input_text.strip()

    # Must start with /
    if not input_text.startswith("/"):
        return None

    # Remove leading /
    command_text = input_text[1:]
    if not command_text:
        return None

    # Split into template name and arguments
    parts = command_text.split(None, 1)  # Split on first whitespace
    template_name = parts[0]
    raw_args = parts[1] if len(parts) > 1 else ""

    # Parse arguments (handles quoted strings)
    try:
        arguments = shlex.split(raw_args) if raw_args else []
    except ValueError:
        # If shlex fails, fall back to simple split
        arguments = raw_args.split() if raw_args else []

    return ParsedCommand(
        template_name=template_name,
        arguments=arguments,
        raw_args=raw_args,
    )


def substitute_arguments(content: str, arguments: list[str]) -> str:
    """Substitute arguments into template content.

    Substitution patterns:
    - $1, $2, ... $N - Positional arguments (1-indexed)
    - $@ - All arguments joined with spaces
    - $ARGUMENTS - Same as $@
    - ${@:N} - Arguments from position N onwards (1-indexed)
    - ${@:N:L} - L arguments starting from position N (1-indexed)

    Args:
        content: Template content with substitution patterns
        arguments: List of arguments to substitute

    Returns:
        Content with arguments substituted
    """
    result = content

    # Handle ${@:N:L} pattern - slice with offset and length
    result = _substitute_slice_with_length(result, arguments)

    # Handle ${@:N} pattern - slice from offset
    result = _substitute_slice(result, arguments)

    # Handle $@ and $ARGUMENTS - all arguments
    all_args = " ".join(arguments)
    result = result.replace("$@", all_args)
    result = result.replace("$ARGUMENTS", all_args)

    # Handle $N patterns - positional arguments (must do after $@ to avoid conflicts)
    result = _substitute_positional(result, arguments)

    return result


def _substitute_positional(content: str, arguments: list[str]) -> str:
    """Substitute positional arguments ($1, $2, etc.)."""
    # Match $N where N is one or more digits, not followed by more special chars
    pattern = re.compile(r"\$(\d+)")

    def replace_match(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1  # Convert to 0-indexed
        if 0 <= index < len(arguments):
            return arguments[index]
        return ""  # Missing argument becomes empty string

    return pattern.sub(replace_match, content)


def _substitute_slice(content: str, arguments: list[str]) -> str:
    """Substitute slice patterns (${@:N})."""
    # Match ${@:N} where N is a number
    pattern = re.compile(r"\$\{@:(\d+)\}")

    def replace_match(match: re.Match[str]) -> str:
        start = int(match.group(1)) - 1  # Convert to 0-indexed
        if start < 0:
            start = 0
        sliced = arguments[start:]
        return " ".join(sliced)

    return pattern.sub(replace_match, content)


def _substitute_slice_with_length(content: str, arguments: list[str]) -> str:
    """Substitute slice patterns with length (${@:N:L})."""
    # Match ${@:N:L} where N and L are numbers
    pattern = re.compile(r"\$\{@:(\d+):(\d+)\}")

    def replace_match(match: re.Match[str]) -> str:
        start = int(match.group(1)) - 1  # Convert to 0-indexed
        length = int(match.group(2))
        if start < 0:
            start = 0
        sliced = arguments[start : start + length]
        return " ".join(sliced)

    return pattern.sub(replace_match, content)


def expand_template(template_content: str, command: ParsedCommand) -> str:
    """Expand a template with the given command arguments.

    Args:
        template_content: The template content with substitution patterns
        command: Parsed command with arguments

    Returns:
        Expanded template content
    """
    return substitute_arguments(template_content, command.arguments)
