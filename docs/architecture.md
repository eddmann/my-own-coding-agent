# Architecture Overview

This project is intentionally modular so each piece can be read and tested in isolation.

## Delivery layer

- `src/agent/cli.py`
  - Entrypoint commands, config loading, provider bootstrap, and run-mode dispatch (TUI/headless).
- `src/agent/tui/*`
  - Interactive Textual app that drives an `Agent` instance and renders stream/tool/session state.

## Bootstrap and composition

- `src/agent/config/runtime.py`
  - Delivery config loading and merge order (global, project, env).
  - Produces `Config` and projects to core `AgentSettings` via `to_agent_settings()`.
- `src/agent/llm/factory.py`
  - Provider bootstrap from flat params (`provider`, `model`, `api_key`, `base_url`, overrides).
  - Handles provider resolution and provider/model validation before constructing concrete adapters.

## Core runtime

- `src/agent/core/agent.py`
  - Agent loop orchestration: input preprocessing, compaction checks, model streaming, tool execution.
  - Depends on injected `LLMProvider` and `AgentSettings`.
- `src/agent/core/session.py`
  - JSONL session storage, resume/fork/tree behavior, and model-change entries.
- `src/agent/core/context.py`
  - Context window management and compaction summarization.
- `src/agent/core/prompt_builder.py`
  - System prompt composition from tools, skills, and context files.
- `src/agent/core/settings.py`
  - Core runtime DTO (`AgentSettings`) and thinking-level policy.

## LLM and streaming

- `src/agent/llm/provider.py`
  - Shared provider protocol consumed by core and delivery.
- `src/agent/llm/openai/*`, `src/agent/llm/anthropic/*`, `src/agent/llm/openai_codex/*`, `src/agent/llm/openai_compat.py`
  - Concrete provider implementations.
- `src/agent/llm/events.py`, `src/agent/llm/stream.py`, `src/agent/llm/models.py`
  - Event schema, stream abstraction, and model capability/policy helpers.

## Tooling

- `src/agent/tools/*`
  - Built‑in tools + registry (`tools/registry.py`), Pydantic schemas, and execution.

## Skills + prompts

- `src/agent/skills/*`
  - Skill discovery, validation, and injection into prompts.
- `src/agent/prompts/*`
  - Prompt template parsing, argument expansion, and loading.

## Extensions

- `src/agent/extensions/*`
  - Extension API, runner, and loader for user‑defined plugins.

## UI / CLI

- `src/agent/tui/*`
  - Textual UI, autocomplete, model selection, context viewing.
- `src/agent/cli.py`
  - Headless CLI and TUI entrypoints.
