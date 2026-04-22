# Plan Mode Extension

A plan mode keeps planning work out of the main implementation thread until you are ready to act on it.

This example extension lets you:
- enter a planning-only mode
- build up a structured plan over multiple revisions
- persist the current plan to `.agent/plans/<session-id>.json`
- inspect the plan
- apply the approved plan back into the main thread as the next task

It also helps with context management: the plan is condensed into a structured artifact instead of leaving every planning iteration mixed into the parent thread.

## Load it

One-off:

```bash
uv run agent run -e examples/extensions/plan-mode.py
```

Project config:

```toml
extensions = ["examples/extensions/plan-mode.py"]
```

## Commands

- `/plan on`
  Enable planning mode. Active tools are narrowed to `read`, `grep`, `find`, and `ls`, and thinking is raised to `high`.

- `/plan <request>`
  Create or revise the current plan. If plan mode is not already active, it is enabled automatically first.

- `/plan show`
  Show the current structured plan.

- `/plan apply`
  Exit planning mode and queue the approved plan back into the main thread as the next user task.

- `/plan off`
  Leave planning mode without applying the plan.

- `/plan clear`
  Clear the current saved plan file and in-memory plan.

## Plan shape

The planner asks the model to return JSON with:

- `summary`
- `steps`
- `risks`
- `validation`
- `notes`

The current plan is written to:

```text
.agent/plans/<session-id>.json
```

## TUI behavior

When the TUI is available, the extension also:
- shows a footer widget while plan mode is active
- opens a presented view for `/plan show`
- asks for confirmation before `/plan apply`

In headless mode, the same commands still work, but the footer and presented view are skipped.
