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

- `input` — transform or block user input
- `context` — modify messages before LLM call
- `tool_call` — block tool execution
- `tool_result` — rewrite tool output
- any `AgentEvent` type (observability)

## Custom tools and commands

- `api.register_tool(tool)` to add new tools
- `api.register_command("name", handler)` to add `/name` slash commands

## Loading extensions

- Config: `extensions = ["/path/to/ext.py", "./extensions/"]`
- CLI: `agent run -e ./extensions/my_ext.py`

## Examples

- [`examples/extensions/protected-paths.py`](../examples/extensions/protected-paths.py) — blocks writes/edits to protected paths.
- [`examples/extensions/commit-guard.py`](../examples/extensions/commit-guard.py) — blocks `git commit`/`git push` when the tree is dirty; also provides `/dirty`.
- [`examples/extensions/todo-capture.py`](../examples/extensions/todo-capture.py) — `/todo` command appends to TODO.md.
