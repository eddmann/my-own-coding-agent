"""Behavior tests for prompt templates."""

from __future__ import annotations

from agent.prompts.parser import expand_template, parse_command


def test_parse_command_handles_quotes():
    command = parse_command('/review "file name.py" next')

    assert command is not None
    assert command.template_name == "review"
    assert command.arguments == ["file name.py", "next"]


def test_expand_template_substitutes_arguments():
    command = parse_command("/tmpl one two three")
    assert command is not None

    content = "A=$1 B=$2 REST=$@ SLICE=${@:2}"
    expanded = expand_template(content, command)

    assert "A=one" in expanded
    assert "B=two" in expanded
    assert "REST=one two three" in expanded
    assert "SLICE=two three" in expanded
