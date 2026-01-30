"""Dynamic system prompt builder - composable prompt construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agent.skills.skill import Skill


@dataclass(slots=True)
class ContextFile:
    """A context file loaded for system prompt injection."""

    path: Path
    content: str
    source: str  # "project", "ancestor", "explicit"


@dataclass(slots=True)
class SystemPromptOptions:
    """Options for building the system prompt."""

    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    append_system_prompt: str | None = None
    cwd: Path | None = None
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)


# Base system prompt - role description
BASE_PROMPT = (
    "You are a helpful coding assistant with access to tools for reading, writing, "
    "and editing files, as well as running shell commands.\n\n"
    "You help users with software engineering tasks including:\n"
    "- Understanding and navigating codebases\n"
    "- Writing and refactoring code\n"
    "- Debugging and fixing issues\n"
    "- Running commands and scripts\n"
    "- Answering technical questions"
)


# Tool descriptions for the system prompt
TOOL_DESCRIPTIONS = {
    "read": "Read file contents with line numbers",
    "write": "Create or overwrite files",
    "edit": "Find and replace text in files",
    "bash": "Execute shell commands",
    "grep": "Search file contents with regex",
    "find": "Find files by glob pattern",
    "ls": "List directory contents",
}


# Dynamic guidelines based on available tools
TOOL_GUIDELINES = {
    # If file exploration tools available
    ("grep", "find", "ls"): (
        "When exploring files, prefer grep/find/ls over bash commands for better structured output."
    ),
    # If read and edit both available
    ("read", "edit"): (
        "Always read files before editing them to understand the context "
        "and ensure accurate replacements."
    ),
    # If bash available
    ("bash",): (
        "Use bash for running builds, tests, and other shell commands. "
        "Prefer specific tools like grep/find/ls for file exploration."
    ),
}


def build_tool_section(tool_names: list[str]) -> str:
    """Build the available tools section.

    Args:
        tool_names: List of enabled tool names

    Returns:
        Formatted tool section text
    """
    if not tool_names:
        return ""

    lines = ["Available tools:"]
    for name in sorted(tool_names):
        if desc := TOOL_DESCRIPTIONS.get(name):
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")

    return "\n".join(lines)


def build_guidelines_section(tool_names: list[str]) -> str:
    """Build dynamic guidelines based on available tools.

    Args:
        tool_names: List of enabled tool names

    Returns:
        Formatted guidelines section text
    """
    tool_set = set(tool_names)
    guidelines = []

    for required_tools, guideline in TOOL_GUIDELINES.items():
        # Check if all required tools are available
        if all(tool in tool_set for tool in required_tools):
            guidelines.append(f"- {guideline}")

    if not guidelines:
        return ""

    return "Guidelines:\n" + "\n".join(guidelines)


def build_context_section(context_files: list[ContextFile]) -> str:
    """Build the context files section.

    Args:
        context_files: List of loaded context files

    Returns:
        Formatted context section text
    """
    if not context_files:
        return ""

    sections = []
    for ctx_file in context_files:
        # Use appropriate header based on source
        if ctx_file.source == "project":
            header = f"## Project Context ({ctx_file.path.name})"
        elif ctx_file.source == "ancestor":
            header = f"## Context from {ctx_file.path}"
        else:
            header = f"## Context ({ctx_file.path.name})"

        sections.append(f"{header}\n\n{ctx_file.content}")

    return "\n\n".join(sections)


def build_skills_section(skills: list[Skill]) -> str:
    """Build the skills section as XML.

    Args:
        skills: List of available skills

    Returns:
        XML formatted skills section
    """
    if not skills:
        return ""

    from agent.skills.formatter import format_skills_xml

    return format_skills_xml(skills)


def build_environment_section(cwd: Path | None = None) -> str:
    """Build the environment information section.

    Args:
        cwd: Current working directory

    Returns:
        Formatted environment section text
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    lines = [
        "Environment:",
        f"- Date/Time: {date_str}",
    ]

    if cwd:
        lines.append(f"- Working Directory: {cwd}")

    return "\n".join(lines)


def build_system_prompt(options: SystemPromptOptions | None = None) -> str:
    """Build the complete system prompt from components.

    Prompt layers (in order):
    1. Custom prompt OR base prompt
    2. Tool descriptions (only enabled tools)
    3. Dynamic guidelines (based on tool combinations)
    4. Context files (AGENTS.md, CLAUDE.md from ancestors)
    5. Skills XML
    6. Environment (date/time, working directory)
    7. Appended content (if any)

    Args:
        options: Options for building the prompt

    Returns:
        Complete system prompt text
    """
    options = options or SystemPromptOptions()
    sections: list[str] = []

    # 1. Base or custom prompt
    if options.custom_prompt:
        sections.append(options.custom_prompt)
    else:
        sections.append(BASE_PROMPT)

    # 2. Tool descriptions
    if options.selected_tools and (tool_section := build_tool_section(options.selected_tools)):
        sections.append(tool_section)

    # 3. Dynamic guidelines
    if options.selected_tools and (guidelines := build_guidelines_section(options.selected_tools)):
        sections.append(guidelines)

    # 4. Context files
    if options.context_files and (context_section := build_context_section(options.context_files)):
        sections.append(context_section)

    # 5. Skills (only if read tool is available)
    if (
        options.skills
        and (not options.selected_tools or "read" in options.selected_tools)
        and (skills_section := build_skills_section(options.skills))
    ):
        sections.append(skills_section)

    # 6. Environment
    env_section = build_environment_section(options.cwd)
    sections.append(env_section)

    # 7. Appended content
    if options.append_system_prompt:
        sections.append(options.append_system_prompt)

    return "\n\n".join(sections)
