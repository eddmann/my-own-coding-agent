"""Skill formatter - formats skills as XML for system prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from agent.skills.skill import Skill


def escape_xml(text: str) -> str:
    """Escape special XML characters in text.

    Args:
        text: Text to escape

    Returns:
        XML-safe text
    """
    return escape(text)


def format_skill_xml(skill: Skill) -> str:
    """Format a single skill as XML.

    Args:
        skill: The skill to format

    Returns:
        XML string for the skill
    """
    name = escape_xml(skill.name)
    description = escape_xml(skill.description)
    location = escape_xml(str(skill.readme_path))

    return f"""  <skill>
    <name>{name}</name>
    <description>{description}</description>
    <location>{location}</location>
  </skill>"""


def format_skills_xml(skills: list[Skill]) -> str:
    """Format multiple skills as XML for system prompt injection.

    Output format:
    ```xml
    <available_skills>
      <skill>
        <name>skill-name</name>
        <description>What it does</description>
        <location>/path/to/SKILL.md</location>
      </skill>
      ...
    </available_skills>
    ```

    Args:
        skills: List of skills to format

    Returns:
        XML string containing all skills
    """
    if not skills:
        return ""

    skill_entries = "\n".join(format_skill_xml(skill) for skill in skills)

    return "\n".join(
        [
            "The following skills provide specialized instructions for specific tasks.",
            "Use the read tool to load a skill's file when the task matches its description.",
            "",
            "<available_skills>",
            skill_entries,
            "</available_skills>",
        ]
    )


def format_skills_summary(skills: list[Skill]) -> str:
    """Format a brief summary of available skills.

    Args:
        skills: List of skills to summarize

    Returns:
        Brief text summary
    """
    if not skills:
        return "No skills available."

    lines = ["Available skills:"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")

    return "\n".join(lines)
