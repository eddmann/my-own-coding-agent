# Commit Guard Extension

This example blocks risky git operations when the working tree is dirty.

It combines two extension patterns:
- a slash command for quick inspection
- a `tool_call` guard that blocks `bash` commands when policy says they are unsafe

## Load it

```bash
uv run agent run -e examples/extensions/commit-guard.py
```

## Commands

- `/dirty`
  Show the current `git status --porcelain` result in a short human-readable form.

## What it blocks

When the extension sees a `bash` tool call containing:
- `git commit`
- `git push`

it runs `git status --porcelain`.

If the tree is dirty, the command is blocked and the reason is returned to the main thread.

## Why it exists

This is a compact example of an extension that mixes command registration with a runtime policy guard.
