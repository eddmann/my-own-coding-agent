# Tools

Tools are Pydantic‑typed units of capability with OpenAI‑style JSON schemas. The registry lives in `src/agent/tools/registry.py`.

## Built‑in tools

- `read` — read a file with line numbers
- `write` — create or overwrite a file
- `edit` — exact find/replace edits
- `bash` — run shell commands
- `grep` — regex search (uses `rg` when available)
- `find` — glob search
- `ls` — list directories

## How tools are executed

1. The model returns a tool call with `name` + JSON args.
2. The registry validates args via Pydantic.
3. The tool executes and returns a string result (or raises `ToolError`).
4. The registry wraps output into `ToolExecutionResult { content, is_error }`.
5. Tool results are appended as `Role.TOOL` messages.

## Extending tools

Extensions can register tools via `ExtensionAPI.register_tool()`. Once registered, they are treated exactly like built‑ins and appear in the system prompt tool list.
