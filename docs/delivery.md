# Delivery

Delivery is the outermost layer of the system.

The important architectural point is:

- `agent.runtime` owns the agent loop and session/model/tool behavior
- `agent.extensions` is an optional hook host on top of the runtime
- delivery layers decide how users interact with that runtime

So the same agent can be delivered through multiple shells without changing the runtime itself.

## Delivery modes

### TUI

- Implementation: `src/agent/tui/`
- Entry from CLI: `agent run` with no prompt/headless flag
- Shape:
  - builds the interactive runtime stack
  - handles TUI-native commands and runtime actions
  - renders runtime chunks into Textual widgets
  - hosts extension-owned UI surfaces

The TUI is the richest delivery surface because it supports:

- interactive chat
- model/session controls
- extension-owned UI prompts and widgets
- incremental rendering of thinking, text, and tool activity

### Headless CLI

- Implementation: `src/agent/cli/headless.py`
- Entry: `agent run --headless "..."` or `make run-headless`
- Shape:
  - builds the same runtime stack
  - optionally attaches the extension host
  - renders runtime chunks directly to stdout

This is a thin delivery shell over the same runtime.

It is useful for:

- one-shot prompts
- scripting
- CI or automation flows
- low-ceremony debugging of the runtime

### Web

- Implementation: `src/agent/web/`
- Entry: `agent web` or `make run-web`
- Shape:
  - exposes HTTP endpoints for health, sessions, and models
  - hosts a WebSocket session bound to one runtime stack
  - streams runtime chunks to the browser
  - hosts extension-owned UI surfaces through a web bridge

The web shell is local-first and uses the same runtime and extension host as the TUI and CLI.

## Delivery contract

A delivery layer is responsible for:

- loading config
- creating the provider
- loading or selecting the session
- creating the `Agent`
- optionally attaching an `ExtensionHost`
- rendering runtime output
- mapping delivery-specific controls back into runtime actions

A delivery layer should not own:

- the agent loop
- session semantics
- tool execution semantics
- provider behavior
- extension dispatch semantics

Those belong below delivery.

## Delivery shape

The delivery layer exists in two places:

- `src/agent/cli/`
- `src/agent/tui/app.py`

The boundary is:

- CLI owns command parsing, headless stdout delivery, and session utility commands
- TUI owns interactive chat delivery, model/session controls, and extension-hosted UI
- Web owns HTTP/WebSocket transport, browser-facing state snapshots, and extension-hosted UI transport

## Design rule

When adding a new delivery mode, prefer:

- reusing `Agent`
- reusing `ExtensionHost`
- keeping delivery-specific rendering and input handling at the edge

Avoid moving runtime concerns upward into delivery just because one shell needs them.
