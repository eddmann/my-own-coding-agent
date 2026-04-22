# Docs Index

Start here if you’re new to the project.

- [`architecture.md`](architecture.md) — system overview and module responsibilities
- [`delivery.md`](delivery.md) — TUI, headless CLI, and web delivery surfaces over the same runtime
- [`cli.md`](cli.md) — Typer command surface, headless mode, and session utilities
- [`web.md`](web.md) — FastAPI/WebSocket delivery shell and browser protocol
- [`agent-loop.md`](agent-loop.md) — step‑by‑step loop walkthrough
- [`tools.md`](tools.md) — tool schemas, registry, and built‑ins
- [`skills.md`](skills.md) — skill format, validation rules, search paths
- [`prompts.md`](prompts.md) — template format and argument expansion
- [`extensions.md`](extensions.md) — extension host API, lifecycle hooks, and UI surface
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
make run-web
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
  - [`examples/extensions/protected-paths.md`](../examples/extensions/protected-paths.md)
  - [`examples/extensions/commit-guard.py`](../examples/extensions/commit-guard.py)
  - [`examples/extensions/commit-guard.md`](../examples/extensions/commit-guard.md)
  - [`examples/extensions/mcp-adapter.py`](../examples/extensions/mcp-adapter.py)
  - [`examples/extensions/mcp-adapter.md`](../examples/extensions/mcp-adapter.md)
  - [`examples/extensions/plan-mode.py`](../examples/extensions/plan-mode.py)
  - [`examples/extensions/plan-mode.md`](../examples/extensions/plan-mode.md)
  - [`examples/extensions/subagents.py`](../examples/extensions/subagents.py)
  - [`examples/extensions/subagents.md`](../examples/extensions/subagents.md)
