# Subagents Extension

This example extension adds delegated child-agent runs on top of the extension host.

It is implemented in [subagents.py](./subagents.py).

## What a subagent is

A subagent is a delegated agent run with its own scoped task and execution policy.

In this example, a subagent:

- runs separately from the parent agent turn
- can use a different profile, instructions, thinking level, and active tools
- does focused work on one task
- reports a structured result back to the parent flow

This also helps with context management: the parent thread stays cleaner, while the delegated work happens in a separate child run and only the useful result needs to come back.

The parent thread can then inspect that result, keep it in the side panel, or apply it back into the main conversation.

## Load it

One-off in the TUI:

```bash
uv run agent run -e examples/extensions/subagents.py
```

Or load it by default in `.agent/config.toml`:

```toml
extensions = ["examples/extensions/subagents.py"]
```

## Profiles

- `researcher`
  - read-only
  - tools: `read`, `grep`, `find`, `ls`
  - use for context gathering and option synthesis

- `reviewer`
  - read-only
  - tools: `read`, `grep`, `find`, `ls`
  - use for bug, regression, and testing review

- `implementer`
  - write-enabled
  - tools: `read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`
  - use for scoped implementation work

## Commands

- `/subagent <profile> <task>`
  - launch a subagent directly

- `/subagent`
  - launch interactively in the TUI with profile select plus task input

- `/subagents`
  - list recent runs for the current session

- `/subagent-show <id>`
  - inspect one result

- `/subagent-apply <id> [summary|full]`
  - push a subagent result back into the main thread as a queued user message

- `/subagent-cancel <id>`
  - cancel a running subagent

## Result shape

Subagents are prompted to return structured JSON with:

- `summary`
- `details`
- `findings`
- `recommended_next_step`

The extension also records:

- `status`
- `files_changed`
- `commands_run`
- `error`

## UI behavior

In the TUI, the extension uses:

- `select(...)` to choose a profile for `/subagent`
- `input(...)` to collect the delegated task
- `present(view)` for result inspection
- `set_widget("right_panel", ...)` for persistent progress/status
- `confirm(...)` before launching the write-enabled `implementer`

Completed runs stay visible in `/subagents` history. Once a result is applied, it is marked as applied and removed from the `right_panel`, so the side panel only shows active or unapplied runs.

Headless mode can still load and run the extension, but the interactive launcher and widget surfaces are TUI-only.
