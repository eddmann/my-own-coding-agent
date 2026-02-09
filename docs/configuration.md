# Configuration

Configuration is layered and merged in this order (lowest → highest):

1) Defaults
2) Global config (`~/.agent/config.toml` or `.yaml`)
3) Project config (`.agent/config.toml` or `.yaml`)
4) Environment variables (`AGENT_*`)

## Common settings

- `provider`, `model`, `api_key`, `base_url`
- `context_max_tokens`, `max_output_tokens`, `temperature`
- `skills_dirs`, `prompt_template_dirs`, `extensions`
- `custom_system_prompt`, `append_system_prompt`

## Authentication

OpenAI API keys are read from `OPENAI_API_KEY`. Anthropic API keys are read from `ANTHROPIC_API_KEY`.

OAuth credentials are stored per provider: Anthropic in `~/.agent/anthropic-oauth.json`, and OpenAI Codex in `~/.agent/openai-codex-oauth.json`.

Use `agent auth login anthropic` or `agent auth login openai-codex` to authenticate, `agent auth logout anthropic` or `agent auth logout openai-codex` to clear credentials, and `agent auth status` to inspect current OAuth state.

Resolution behavior: Anthropic OAuth is used when no Anthropic API key is configured. OpenAI Codex uses OAuth credentials by default, or `OPENAI_CODEX_OAUTH_TOKEN` if explicitly set.

## Context files

`AGENTS.md` and `CLAUDE.md` are auto‑loaded from the project and its ancestors to seed the system prompt.
