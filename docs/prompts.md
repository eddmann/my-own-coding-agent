# Prompt Templates

Prompt templates are Markdown files invoked as slash commands (`/template-name`).

## Template format

```markdown
---
name: template-name
description: Brief description
---

Template content with $1 $2 $@ substitutions.
```

## Argument substitution

- `$1`, `$2`, … — positional arguments
- `$@` / `$ARGUMENTS` — all args joined
- `${@:2}` — args from position 2 onward
- `${@:2:3}` — 3 args starting at position 2

## Discovery order

1. `~/.agent/prompts/`
2. `prompt_template_dirs` from config
3. `.agent/prompts/` (project‑local)

## Execution

- `/template-name args...` expands the template before the model call.
- If a slash command matches an extension command and no template exists, the extension command runs instead.

## Examples

- [`examples/prompts/quick-plan.md`](../examples/prompts/quick-plan.md) — short planning template.
- [`examples/prompts/review.md`](../examples/prompts/review.md) — bug‑focused review template.
- [`examples/prompts/summarize-changes.md`](../examples/prompts/summarize-changes.md) — change summary template.

Copy any of these into `.agent/prompts/` or add `prompt_template_dirs` in config to load them.
