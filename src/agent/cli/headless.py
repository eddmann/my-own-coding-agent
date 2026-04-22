"""Headless CLI delivery mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.runtime.chunk import MessageChunk, TextDeltaChunk, ThinkingDeltaChunk, ToolCallChunk

if TYPE_CHECKING:
    from agent.config import Config
    from agent.llm.provider import LLMProvider
    from agent.runtime.session import Session


async def run_headless(
    config: Config,
    prompt: str,
    session: Session | None,
    llm_provider: LLMProvider,
) -> None:
    """Run a single prompt without TUI."""
    from agent.extensions.host import ExtensionHost
    from agent.runtime.agent import Agent

    agent = Agent(
        config.to_agent_settings(),
        llm_provider,
        session,
    )
    extension_host: ExtensionHost | None = None

    if config.extensions:
        extension_host = ExtensionHost(agent, paths=config.extensions)
        agent.set_hooks(extension_host)
        errors = await extension_host.load_extensions()
        for error in errors:
            print(f"[Extension error: {error}]", flush=True)

    in_thinking = False
    try:
        async for chunk in agent.run(prompt):
            match chunk:
                case ThinkingDeltaChunk():
                    if not in_thinking:
                        print("\n[Thinking...]", flush=True)
                        in_thinking = True
                case TextDeltaChunk(payload=text):
                    if in_thinking:
                        print("\n[/Thinking]", flush=True)
                        in_thinking = False
                    print(text, end="", flush=True)
                case ToolCallChunk(payload=tool_call):
                    if in_thinking:
                        print("\n[/Thinking]", flush=True)
                        in_thinking = False
                    print(f"\n[Tool: {tool_call.name}]", flush=True)
                case MessageChunk(payload=msg):
                    if msg.role.value != "system":
                        continue
                    if in_thinking:
                        print("\n[/Thinking]", flush=True)
                        in_thinking = False
                    print(f"\n[System] {msg.content}", flush=True)
                case _:
                    pass
        if in_thinking:
            print("\n[/Thinking]", flush=True)
        print()
    except Exception as exc:
        if in_thinking:
            print("\n[/Thinking]", flush=True)
        print(f"\n[Error: {type(exc).__name__}: {exc}]", flush=True)
    finally:
        await agent.close()
