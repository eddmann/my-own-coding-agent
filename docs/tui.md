# TUI (Textual)

The interactive terminal UI is built with Textual (`src/agent/tui/*`).

## Features

- Streaming assistant output + tool calls
- Skill and slash‑command autocomplete
- Status bar showing model + token usage
- Model selector and context viewer

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

## Keybindings

- `Ctrl+C` — quit
- `Ctrl+L` — clear
- `Esc` — cancel or refocus input
