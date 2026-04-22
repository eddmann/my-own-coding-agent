# CLI

The CLI lives in `src/agent/cli/` and is the lighter delivery surface alongside the TUI.

## Structure

The CLI is split by role rather than by framework detail:

- command surface and dispatch
- headless stdout delivery
- session utility commands

## Delivery modes

### Interactive handoff

`agent run` with no prompt/headless flag creates config + provider + session state, then hands off to the Textual TUI.

### Headless mode

`agent run --headless "..."` or `make run-headless` runs one prompt through the same runtime and renders:

- thinking markers
- streamed text
- tool activity
- system messages

directly to stdout.

## Session commands

The CLI also exposes delivery-level session utilities:

- `agent fork`
- `agent tree`
- `agent sessions`
- `agent config-show`

These are delivery commands over the same JSONL session model used by the runtime and TUI.

## Design role

The CLI owns:

- command parsing
- config override/application
- provider creation
- session selection
- delivery dispatch

It should not own:

- the runtime agent loop
- tool execution semantics
- session semantics themselves
- extension dispatch behavior

Those stay below the delivery boundary.
