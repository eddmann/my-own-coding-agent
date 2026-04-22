# Web

The web delivery shell lives in `src/agent/web/` and exposes the runtime over FastAPI and WebSockets.

## Shape

The web package is split by delivery responsibility:

- `server.py` — FastAPI app, HTTP routes, WebSocket endpoint, connection lifecycle
- `compose.py` — build `Agent` + `ExtensionHost` for one web client
- `bridge.py` — map `ctx.ui` calls onto browser-facing WebSocket events
- `schemas.py` — inbound WebSocket payload models and shared response shapes
- `sessions.py` — web-facing session lookup and summary helpers

## Transport

### HTTP

The HTTP surface provides:

- `GET /api/health`
- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/models`

### WebSocket

The `/ws` endpoint binds one browser client to one runtime stack.

Client messages include:

- `run.start`
- `run.cancel`
- `session.new`
- `session.load`
- `model.set`
- `model.set_thinking`
- `ui.response`
- `ui.view_action`

Server messages include:

- `state.snapshot`
- `run.started`
- `run.finished`
- `run.failed`
- `chunk.text`
- `chunk.thinking`
- `chunk.tool_start`
- `chunk.tool_call`
- `chunk.tool_result`
- `chunk.message`
- `session.changed`
- `model.changed`
- `ui.notify`
- `ui.status`
- `ui.request`
- `ui.view_present`
- `ui.view_close`
- `ui.widget`

## Extension UI

The web bridge uses the same extension UI contract as the TUI:

- `notify`
- `set_status`
- `input`
- `confirm`
- `select`
- `present`
- `set_widget`

That keeps extension code delivery-agnostic. The host decides how those surfaces are rendered.

## Design role

The web shell owns:

- HTTP/WebSocket transport
- browser-facing state snapshots
- browser-facing chunk serialization
- browser responses to extension-owned UI requests

It does not own:

- the runtime loop
- session semantics
- tool execution semantics
- extension dispatch semantics
