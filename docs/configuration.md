# Configuration

Configuration is layered and merged in this order (lowest → highest):

1) Defaults
2) Global config (`~/.agent/config.toml` or `.yaml`)
3) Project config (`.agent/config.toml` or `.yaml`)
4) Environment variables (`AGENT_*`)

## Runtime split

- `Config` (`src/agent/config/runtime.py`) is a delivery/bootstrap concern.
- Core runtime uses `AgentSettings` (`src/agent/core/settings.py`), projected from config via `Config.to_agent_settings()`.
- Provider construction and provider/model resolution are handled in `src/agent/llm/factory.py`, not in core.

## Common settings

- `provider`, `model`, `api_key`, `base_url`
- `context_max_tokens`, `max_output_tokens`, `temperature`
- `skills_dirs`, `prompt_template_dirs`, `extensions`
- `custom_system_prompt`, `append_system_prompt`

`model` can be omitted for native providers (`openai`, `anthropic`, `openai-codex`) because they have built-in defaults. For OpenAI-compatible providers (`ollama`, `openrouter`, `groq`, custom), set `model` explicitly unless you configured a provider-specific override model.

## Authentication

OpenAI API keys are read from `OPENAI_API_KEY`. Anthropic API keys are read from `ANTHROPIC_API_KEY`.

OAuth credentials are stored per provider: Anthropic in `~/.agent/anthropic-oauth.json`, and OpenAI Codex in `~/.agent/openai-codex-oauth.json`.

Use `agent auth login anthropic` or `agent auth login openai-codex` to authenticate, `agent auth logout anthropic` or `agent auth logout openai-codex` to clear credentials, and `agent auth status` to inspect current OAuth state.

Resolution behavior: Anthropic OAuth is used when no Anthropic API key is configured. OpenAI Codex uses OAuth credentials by default, or `OPENAI_CODEX_OAUTH_TOKEN` if explicitly set.

## OpenRouter

Use OpenRouter through the OpenAI-compatible path:

- Set `provider` to `openrouter`
- Set `model` to an OpenRouter model ID (for example `vendor/model:free`)
- Set `OPENROUTER_API_KEY`

Example:

```bash
export OPENROUTER_API_KEY=your_key
uv run agent run -p openrouter -m "openrouter/free" "hey"
```

The default OpenRouter base URL is `https://openrouter.ai/api`. Only set `AGENT_BASE_URL` if you need a custom endpoint.

Free-model examples (availability can change by account, region, and provider routing):

```bash
uv run agent run -p openrouter -m "openrouter/free" "hey"
uv run agent run -p openrouter -m "openrouter/aurora-alpha" "hey"
uv run agent run -p openrouter -m "z-ai/glm-4.5-air:free" "hey"
uv run agent run -p openrouter -m "meta-llama/llama-3.3-70b-instruct:free" "hey"
```

## Ollama (local)

Use Ollama through the OpenAI-compatible path:

- Set `provider` to `ollama`
- Set `model` to a local Ollama model name (from `ollama list`)
- Ensure Ollama is running locally

Example:

```bash
ollama pull gpt-oss
uv run agent run -p ollama -m gpt-oss "hey"
```

The default Ollama base URL is `http://localhost:11434`.
Only set `AGENT_BASE_URL` if your Ollama endpoint is different.

## Context files

`AGENTS.md` and `CLAUDE.md` are auto‑loaded from the project and its ancestors to seed the system prompt.
