# Docs Index

Start here if you’re new to the project.

- [`architecture.md`](architecture.md) — system overview and module responsibilities
- [`agent-loop.md`](agent-loop.md) — step‑by‑step loop walkthrough
- [`tools.md`](tools.md) — tool schemas, registry, and built‑ins
- [`skills.md`](skills.md) — skill format, validation rules, search paths
- [`prompts.md`](prompts.md) — template format and argument expansion
- [`extensions.md`](extensions.md) — extension API and lifecycle hooks
- [`llm.md`](llm.md) — provider adapters and streaming events
- [`tui.md`](tui.md) — Textual UI behavior and commands
- [`configuration.md`](configuration.md) — config files, env vars, context files
- [`sessions.md`](sessions.md) — JSONL sessions, forking, compaction

## Common commands

Use the Makefile targets where possible:

```bash
make deps
make run
make run-headless PROMPT="List all Python files"
make test
make lint
make format
make can-release
```

## Examples

- Skills:
  - [`examples/skills/repo-scan/SKILL.md`](../examples/skills/repo-scan/SKILL.md)
  - [`examples/skills/safe-refactor/SKILL.md`](../examples/skills/safe-refactor/SKILL.md)
- Prompt templates:
  - [`examples/prompts/quick-plan.md`](../examples/prompts/quick-plan.md)
  - [`examples/prompts/review.md`](../examples/prompts/review.md)
  - [`examples/prompts/summarize-changes.md`](../examples/prompts/summarize-changes.md)
- Extensions:
  - [`examples/extensions/protected-paths.py`](../examples/extensions/protected-paths.py)
  - [`examples/extensions/commit-guard.py`](../examples/extensions/commit-guard.py)
  - [`examples/extensions/todo-capture.py`](../examples/extensions/todo-capture.py)
