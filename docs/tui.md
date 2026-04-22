# TUI (Textual)

The interactive terminal UI is built with Textual (`src/agent/tui/*`).

## Internal structure

The TUI is split into a few clear roles:

- shell/lifecycle
- runtime composition
- delivery command handling
- chat/session rendering
- extension UI hosting

The widget and modal files sit underneath that split as the leaf UI components.

## Features

- Streaming assistant output + tool calls
- Skill and slash‑command autocomplete
- Status bar showing model + token usage + thinking level + extension status
- Model selector and context viewer
- Extension UI host for `notify`, `input`, `confirm`, `select`, `present(view)`, and persistent widgets

## Built‑in commands

- `/clear` — clear chat
- `/new` — new session
- `/load` — load a session
- `/resume` — pick a session to resume
- `/fork` — fork from a message
- `/tree` — move session leaf (tree view; linear list when no branches)
- `/context` — show context files
- `/model` — open model/thinking modal
- `/model <name>` — switch model directly
- `/help` — quick help
- `/quit` — exit

Model switching is routed through `Agent.set_model(...)` and provider `set_model(...)`. Invalid provider/model pairs are rejected and shown as a system message.

These commands are delivery-level commands owned by the TUI controller. They are handled before input enters the runtime hook/LLM path.

## Extension UI surfaces

When a TUI `AgentApp` is active, extensions get a bound `ctx.ui` surface. That supports:

- notifications rendered into the chat stream
- status text in the status bar
- prompt / confirm / select modals
- temporary presented custom views with view-defined controls
- persistent widget slots:
  - `footer`
  - `right_panel`

Presented views are host-framed dialogs, but the view decides which controls appear inside them. The supported control kinds are `input`, `select`, and `button`.

The bundled subagent example uses `select`, `input`, and `confirm` for launching, plus a `right_panel` widget for live progress and a presented read-only result view for inspection.

## Rendering model

The TUI does not render the main conversation from generic runtime events.

Instead:

- streamed runtime chunks drive the chat/tool/thinking UI
- extension-owned UI is driven through the bound `ctx.ui` bridge

That split keeps the main conversation rendering incremental and presentation-focused, while leaving lifecycle/event reactions to the extension host.

## Design role

The TUI owns:

- interactive delivery behavior
- focus/cancel state
- model/session controls
- rendering streamed runtime chunks
- hosting extension-owned prompts, views, and widgets

It should not own:

- the runtime loop
- session semantics
- tool execution semantics
- extension dispatch itself

## Keybindings

- `Ctrl+C` — quit
- `Ctrl+L` — clear
- `Esc` — cancel or refocus input
