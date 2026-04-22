# Agent Loop (Detailed)

The agent loop is implemented in `src/agent/runtime/agent.py` and is built around streaming responses and tool execution.

## Step 1: Input intake & preprocessing

- The runtime first asks its hook host to resolve raw input via `resolve_input(...)`.
- That hook can:
  - block the input
  - replace the text
  - fully handle the input and return output without calling the LLM
- After hook-based input resolution, built-in preprocessing runs:
  - `$skill-name` expands into a `<skill>` block + optional args
  - `/template-name args` expands using prompt templates
- The extension host uses input resolution to implement extension slash commands.
- Hook hosts can queue follow-up user messages with run-scoped control, so one command can trigger later LLM turns.

## Step 2: Persist user message + compaction check

- The user message is appended to the JSONL session.
- The `ContextManager` checks token usage and triggers compaction when needed.

## Step 3: Build system prompt + call LLM

- System prompt is composed from:
  - base prompt or custom system prompt
  - active tool descriptions + guidelines
  - context files (AGENTS.md / CLAUDE.md)
  - skills (XML list)
  - environment info
- The model is called via a provider‑specific event stream.

## Step 4: Stream + tool execution

- Text, thinking, and toolcall events stream back.
- Tool calls are accumulated, validated, and executed via the tool registry.
- Hook hosts can block tools or modify tool results before results are appended.

## Step 5: Turn finalization

- Assistant message + tool results are appended to the session.
- Turn, agent, session, model-select, and compaction lifecycle events are emitted.
- Any queued extension follow-up user messages are drained back into the run loop.
- Token counts are updated for the status bar.

## Cancellation

- A shared `asyncio.Event` lets the UI/CLI cancel a running stream or tool loop.
