# Architecture Overview

This project is intentionally modular so each piece can be read and tested in isolation.

## Core modules

- `src/agent/core/agent.py`
  - The agent loop: message persistence, compaction checks, model streaming, tool execution.
- `src/agent/core/session.py`
  - JSONL session storage, resume, and fork support.
- `src/agent/core/context.py`
  - Context window management and compaction summarization.
- `src/agent/core/prompt_builder.py`
  - System prompt composition from tools, skills, and context files.
- `src/agent/core/config.py`
  - Config merging (global, project, env vars) and provider selection.

## Providers + streaming

- `src/agent/llm/*`
  - Provider adapters and a unified event stream (`llm/events.py`, `llm/stream.py`).

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
