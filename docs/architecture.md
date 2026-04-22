# Architecture Overview

This project is intentionally modular so each piece can be read and tested in isolation.

## Delivery layer

Delivery is the outermost layer of the system.

It owns:

- command parsing and mode selection
- user interaction surfaces
- rendering streamed runtime output
- mapping delivery controls back into runtime/session/model actions

It does not own:

- the agent loop
- session semantics
- tool execution semantics
- provider behavior
- extension dispatch semantics

The same runtime stack can be exposed through multiple delivery shells:

- Textual TUI
- headless CLI
- local web delivery

See [`delivery.md`](delivery.md) for the delivery model.

## Bootstrap and composition

Bootstrap sits between delivery and runtime.

It owns:

- loading and merging config
- projecting delivery config into runtime settings
- resolving and constructing the active LLM provider
- assembling the runtime stack for a delivery shell

## Runtime

The runtime is the execution center of the system.

It owns:

- the agent loop
- session state and branching
- context loading and compaction
- system prompt construction
- model/tool execution flow
- runtime lifecycle events

It exposes a small neutral hook boundary so integrations can:

- resolve input
- prepare context
- authorize tool calls
- process tool results
- observe runtime events
- keep run-scoped control state

## LLM and streaming

The LLM layer is a supporting subsystem under the runtime.

It owns:

- provider implementations
- streaming event contracts
- model capability and policy helpers
- provider/model validation rules

## Tooling

The tools layer owns the tool registry, built-in tools, schema validation, and active-tool scoping.

## Skills + prompts

These are built-in input/preprocessing subsystems.

They own:

- skill discovery and validation
- prompt template loading and argument expansion
- preprocessing that happens after hook-based input resolution and before the model call

## Extensions

Extensions are one implementation of the runtime hook boundary.

They own:

- extension loading
- handler registration and dispatch
- extension-facing event/result types
- extension author APIs (`ctx.session`, `ctx.model`, `ctx.tools`, `ctx.runtime`, `ctx.ui`)
- extension-owned mutable state such as run-scoped control and optional UI bindings

## Delivery shells

The system has three delivery shells:

- a Textual TUI
- a Typer CLI with headless mode and session utilities
- a FastAPI-backed web shell with a browser client

Both sit on top of the same runtime and extension host.
