# LLM Providers & Streaming

Providers implement a shared interface (`LLMProvider`) that exposes:

- `stream(messages, tools, options)` — event‑based streaming
- `count_tokens`, `count_messages_tokens` — for context budgeting
- `supports_thinking()` — capability detection
- `list_models()` — optional discovery

## Built‑in providers

- **OpenAI** (`src/agent/llm/openai/`)
  - Uses OpenAI API keys for model access
  - Supports reasoning‑effort controls for supported models
- **OpenAI Codex** (`src/agent/llm/openai_codex/`)
  - Uses ChatGPT Codex subscription auth (OAuth)
  - Calls `chatgpt.com/backend-api/codex/responses`
- **Anthropic** (`src/agent/llm/anthropic/`)
  - Uses Anthropic API keys or OAuth credentials
  - Uses Claude’s Messages API with extended thinking signatures
  - Claude Code emulation when using Anthropic OAuth tokens (`sk-ant-oat...`)
- **OpenAI‑compatible** (`src/agent/llm/openai_compat.py`)
  - Works with Ollama, OpenRouter, Groq, LM Studio, etc.

## Streaming events

The stream yields structured events:

- `text_start`, `text_delta`, `text_end`
- `thinking_start`, `thinking_delta`, `thinking_end`
- `toolcall_start`, `toolcall_delta`, `toolcall_end`
- `done` / `error`

These events are consumed by the agent loop and TUI to render incremental updates.
