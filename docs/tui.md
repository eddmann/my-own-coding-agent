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
- `/model` — change model
- `/help` — quick help
- `/quit` — exit

## Keybindings

- `Ctrl+C` — quit
- `Ctrl+L` — clear
- `Esc` — cancel or refocus input
