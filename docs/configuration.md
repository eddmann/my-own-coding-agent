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

- Anthropic OAuth credentials are stored in `~/.agent/anthropic-oauth.json`.
- Use CLI subcommands: `agent auth login anthropic`, `agent auth logout anthropic`, `agent auth status`.
- OAuth credentials are only used when no Anthropic API key is configured.

## Context files

`AGENTS.md` and `CLAUDE.md` are auto‑loaded from the project and its ancestors to seed the system prompt.
