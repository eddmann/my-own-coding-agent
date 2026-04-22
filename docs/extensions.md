# Extensions

Extensions are Python files that expose a `setup(api: ExtensionAPI)` function. They register handlers, tools, and commands.

## Extension shape

```python
# my_extension.py
from agent.extensions import ExtensionAPI, ToolCallResult

def setup(api: ExtensionAPI):
    async def block_rm(event, ctx):
        if "rm -rf" in str(event.input):
            return ToolCallResult(block=True, reason="dangerous command")
    api.on("tool_call", block_rm)
```

## Event hooks

Extensions can subscribe to event names with `api.on(name, handler)`.

### Stable public events

These are the supported extension contract and should be considered stable:

- `input` — transform or block user input before it is persisted
- `context` — modify messages before the LLM call
- `tool_call` — inspect or block tool execution
- `tool_result` — rewrite tool output
- `session_start` — session became active
- `session_end` — session is about to be replaced or closed
- `turn_start` — one agent turn is starting
- `turn_end` — one agent turn completed
- `agent_start` — the agent started processing an input
- `agent_end` — the agent finished processing an input
- `model_select` — active model changed
- `compaction` — context compaction occurred

### Internal events

These events may be observable, but they are not part of the stable public API
and may change without notice:

- `message_start`
- `message_update`
- `message_end`
- `thinking_start`
- `thinking_delta`
- `thinking_end`
- `tool_execution_start`
- `tool_execution_update`
- `tool_execution_end`

## Custom tools and commands

- `api.register_tool(tool)` to add new tools
- `api.register_command("name", handler)` to add `/name` slash commands

## Extension context

Handlers receive `ctx`, which exposes the host capability surface:

- `ctx.cwd`
- `ctx.config`
  - safe snapshot of provider/model/runtime config values
- `ctx.runtime`
  - `is_idle()`
  - `abort()`
  - `send_user_message(text)`
  - `get_system_prompt()`
- `ctx.session`
  - `id`
  - `parent_id`
  - `messages()`
  - `entries()`
  - `await fork(from_message_id)`
  - `set_leaf(entry_id)`
  - `await new()`
- `ctx.model`
  - `get()`
  - `set(model)`
  - `get_thinking_level()`
  - `set_thinking_level(level)`
- `ctx.tools`
  - `available()`
  - `active()`
  - `set_active(names)`
  - `register(tool)`
- `ctx.ui`
  - `notify(message, level="info")`
  - `set_status(text)`
  - `input(prompt, default=None)`
  - `confirm(prompt)`
  - `select(prompt, options)`
  - `present(view)`
    - `PresentedView` implementations define their own host-rendered controls with `ViewControl`
    - views implement `render()`, `controls()`, `handle_action(...)`, `is_done()`, and `result()`
    - control kinds are `input`, `select`, and `button`
  - `set_widget(slot, view)`
  - `None` when no UI host is bound

## Loading extensions

- Config: `extensions = ["/path/to/ext.py", "./extensions/"]`
- CLI: `agent run -e ./extensions/my_ext.py`

## Examples

- [`examples/extensions/protected-paths.py`](../examples/extensions/protected-paths.py) — blocks writes/edits to protected paths.
- [`examples/extensions/protected-paths.md`](../examples/extensions/protected-paths.md) — usage notes for the protected-path guard example.
- [`examples/extensions/commit-guard.py`](../examples/extensions/commit-guard.py) — blocks `git commit`/`git push` when the tree is dirty; also provides `/dirty`.
- [`examples/extensions/commit-guard.md`](../examples/extensions/commit-guard.md) — usage notes for the commit/push guard example.
- [`examples/extensions/mcp-adapter.py`](../examples/extensions/mcp-adapter.py) — stdio-only MCP extension with lazy server startup, a metadata cache, a single `mcp` proxy tool, optional direct-tool promotion from cache, `/mcp` management commands, a presented browser view, and a footer status widget.
- [`examples/extensions/mcp-adapter.md`](../examples/extensions/mcp-adapter.md) — usage notes for loading and configuring the MCP adapter example.
- [`examples/extensions/plan-mode.py`](../examples/extensions/plan-mode.py) — session-scoped planning mode with read-only tool narrowing, persisted plan files, footer status widget, `/plan show`, and `/plan apply`.
- [`examples/extensions/plan-mode.md`](../examples/extensions/plan-mode.md) — usage notes for loading and trying the plan mode example.
- [`examples/extensions/subagents.py`](../examples/extensions/subagents.py) — `researcher`, `reviewer`, and `implementer` child-agent profiles with `/subagent` launch via `select` + `input` + optional `confirm`, structured results, progress widget, result inspection, and apply-back flow.
- [`examples/extensions/subagents.md`](../examples/extensions/subagents.md) — usage notes for loading and trying the subagent example.
