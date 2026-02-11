# LLM Providers & Streaming

Providers implement a shared interface (`LLMProvider`) that exposes:

- `stream(messages, tools, options)` — event‑based streaming
- `count_tokens`, `count_messages_tokens` — for context budgeting
- `supports_thinking()` — capability detection
- `list_models()` — optional discovery

## Provider bootstrap

- `src/agent/llm/factory.py` is the bootstrap boundary:
  - `resolve_provider_config(...)` resolves base URL/model/api key from flat inputs + optional provider overrides.
  - `create_provider(...)` validates provider/model compatibility and returns a concrete provider instance.
- Delivery (CLI/TUI) constructs providers through the factory and injects them into core `Agent`.

## Built‑in providers

### OpenAI

- Implementation: `src/agent/llm/openai/`
- Uses OpenAI API keys for model access
- Supports reasoning‑effort controls for supported models

### OpenAI Codex

- Implementation: `src/agent/llm/openai_codex/`
- Uses ChatGPT Codex subscription auth (OAuth)
- Calls `chatgpt.com/backend-api/codex/responses`

### Anthropic

- Implementation: `src/agent/llm/anthropic/`
- Uses Anthropic API keys or OAuth credentials
- Uses Claude’s Messages API with extended thinking signatures
- Claude Code emulation when using Anthropic OAuth tokens (`sk-ant-oat...`)

### OpenAI‑compatible

- Implementation: `src/agent/llm/openai_compat.py`
- Works with Ollama, OpenRouter, Groq, LM Studio, etc.
- Requires an explicit `model` for compatible providers unless configured via provider override.

#### OpenRouter

- Available via `provider=openrouter`
- Accepts OpenRouter model IDs such as `openrouter/free` or `meta-llama/llama-3.3-70b-instruct:free`
- Defaults to `https://openrouter.ai/api`; request paths are appended by the OpenAI-compatible transport

#### Ollama

- Available via `provider=ollama`
- Uses local base URL default `http://localhost:11434`

## Model validation policy

- Provider/model compatibility is enforced in the LLM module:
  - factory-time validation in `create_provider(...)`
  - runtime validation in provider `set_model(...)`
- Invalid combinations fail with a `ValueError` (for example, Claude model on OpenAI provider).

## Streaming events

The stream yields structured events:

- `text_start`, `text_delta`, `text_end`
- `thinking_start`, `thinking_delta`, `thinking_end`
- `toolcall_start`, `toolcall_delta`, `toolcall_end`
- `done` / `error`

These events are consumed by the agent loop and TUI to render incremental updates.
