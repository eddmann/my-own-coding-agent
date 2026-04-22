# Protected Paths Extension

This example blocks write-capable operations against paths that should be treated as sensitive.

It is intentionally small and shows the classic guard-style extension pattern:
- subscribe to `tool_call`
- inspect the pending tool name and arguments
- return `ToolCallResult(block=True, reason=...)` when the operation should be refused

## Load it

```bash
uv run agent run -e examples/extensions/protected-paths.py
```

## What it blocks

The extension treats these path fragments as protected:

- `.git`
- `.agent`
- `.env`
- `secrets`

It blocks:
- `write` and `edit` calls whose `path` points into one of those locations
- `bash` commands that contain one of those protected path fragments

## Why it exists

This is a minimal example of an extension that enforces safety policy without changing the rest of the agent loop.
