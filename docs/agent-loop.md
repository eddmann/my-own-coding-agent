# Agent Loop (Detailed)

The agent loop is implemented in `src/agent/core/agent.py` and is built around streaming responses and tool execution.

## Step 1: Input intake & preprocessing

- Extensions can block or transform input (`ExtensionRunner.emit_input`).
- `$skill-name` expands into a `<skill>` block + optional args.
- `/template-name args` expands using prompt templates.
- If the input is an extension slash command and no template matched, the command is executed and the run ends.

## Step 2: Persist user message + compaction check

- The user message is appended to the JSONL session.
- The `ContextManager` checks token usage and triggers compaction when needed.

## Step 3: Build system prompt + call LLM

- System prompt is composed from:
  - base prompt or custom system prompt
  - tool descriptions + guidelines
  - context files (AGENTS.md / CLAUDE.md)
  - skills (XML list)
  - environment info
- The model is called via a provider‑specific event stream.

## Step 4: Stream + tool execution

- Text, thinking, and toolcall events stream back.
- Tool calls are accumulated, validated, and executed via the tool registry.
- Extensions can block tools or modify tool results.

## Step 5: Turn finalization

- Assistant message + tool results are appended to the session.
- Turn and agent lifecycle events are emitted.
- Token counts are updated for the status bar.

## Cancellation

- A shared `asyncio.Event` lets the UI/CLI cancel a running stream or tool loop.
