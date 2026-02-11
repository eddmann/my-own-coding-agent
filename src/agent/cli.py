"""CLI entry point using Typer."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from agent.config import Config
from agent.core.chunk import MessageChunk, TextDeltaChunk, ThinkingDeltaChunk, ToolCallChunk
from agent.core.message import Role
from agent.core.session import MessageEntry, Session, SessionStateEntry
from agent.core.settings import ThinkingLevel
from agent.llm.anthropic.oauth import (
    load_oauth_credentials as load_anthropic_oauth_credentials,
)
from agent.llm.anthropic.oauth import (
    login_flow as anthropic_login_flow,
)
from agent.llm.anthropic.oauth import (
    logout_flow as anthropic_logout_flow,
)
from agent.llm.factory import create_provider
from agent.llm.openai_codex.oauth import (
    load_oauth_credentials as load_openai_codex_oauth_credentials,
)
from agent.llm.openai_codex.oauth import (
    login_flow as openai_codex_login_flow,
)
from agent.llm.openai_codex.oauth import (
    logout_flow as openai_codex_logout_flow,
)

if TYPE_CHECKING:
    from agent.llm.provider import LLMProvider

app = typer.Typer(
    name="agent",
    help="A Python AI coding agent",
    no_args_is_help=False,
)

auth_app = typer.Typer(help="OAuth and API key management")
app.add_typer(auth_app, name="auth")


def main() -> None:
    """Entry point for the CLI."""
    app()


def _create_llm_provider(config: Config) -> LLMProvider:
    """Create the configured LLM provider instance."""
    return create_provider(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        provider_overrides=config.provider_overrides(),
    )


@auth_app.command("login")
def auth_login(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name (anthropic|openai-codex)"),
    ] = "anthropic",
) -> None:
    """Login to an OAuth provider."""
    if provider == "anthropic":
        anthropic_login_flow(typer.prompt, typer.echo)
        return
    if provider == "openai-codex":
        openai_codex_login_flow(typer.prompt, typer.echo)
        return
    typer.echo("Unsupported provider. Use: anthropic or openai-codex.", err=True)
    raise typer.Exit(1)


@auth_app.command("logout")
def auth_logout(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name (anthropic|openai-codex)"),
    ] = "anthropic",
) -> None:
    """Logout from a provider."""
    if provider == "anthropic":
        anthropic_logout_flow(typer.echo)
        return
    if provider == "openai-codex":
        openai_codex_logout_flow(typer.echo)
        return
    typer.echo("Unsupported provider. Use: anthropic or openai-codex.", err=True)
    raise typer.Exit(1)


@auth_app.command("status")
def auth_status() -> None:
    """Show stored credentials."""
    found = False
    if load_anthropic_oauth_credentials():
        typer.echo("anthropic: oauth")
        found = True
    if load_openai_codex_oauth_credentials():
        typer.echo("openai-codex: oauth")
        found = True
    if not found:
        typer.echo("No OAuth credentials found")


@app.command()
def run(
    prompt: Annotated[str | None, typer.Argument(help="Initial prompt to run")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model", help="Model to use")] = None,
    provider: Annotated[str | None, typer.Option("-p", "--provider", help="Provider name")] = None,
    thinking: Annotated[
        str | None,
        typer.Option(
            "-t",
            "--thinking",
            help="Thinking level: off, minimal, low, medium, high",
        ),
    ] = None,
    extension: Annotated[
        list[Path] | None,
        typer.Option("-e", "--extension", help="Extension file(s) to load"),
    ] = None,
    resume: Annotated[bool, typer.Option("-r", "--resume", help="Resume the last session")] = False,
    session: Annotated[
        Path | None, typer.Option("-s", "--session", help="Session file to load")
    ] = None,
    headless: Annotated[
        bool, typer.Option("--headless", help="Run without TUI (single prompt mode)")
    ] = False,
) -> None:
    """Start the coding agent.

    Run interactively (TUI mode) or with a single prompt (headless mode).
    """
    # Load base config
    config = Config.load()

    # Parse thinking level
    thinking_level = config.thinking_level
    if thinking:
        try:
            thinking_level = ThinkingLevel(thinking.lower())
        except ValueError as err:
            typer.echo(f"Invalid thinking level: {thinking}", err=True)
            typer.echo("Valid values: off, minimal, low, medium, high", err=True)
            raise typer.Exit(1) from err

    # Merge extensions from CLI with config
    extensions = list(config.extensions)
    if extension:
        extensions.extend(Path(ext) for ext in extension)

    # Apply CLI overrides
    config = Config(
        provider=provider or config.provider,
        model=model or config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        context_max_tokens=config.context_max_tokens,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        thinking_level=thinking_level,
        session_dir=config.session_dir,
        skills_dirs=config.skills_dirs,
        extensions=extensions,
        providers=config.providers,
        prompt_template_dirs=config.prompt_template_dirs,
        context_file_paths=config.context_file_paths,
        custom_system_prompt=config.custom_system_prompt,
        append_system_prompt=config.append_system_prompt,
    )

    llm_provider = _create_llm_provider(config)

    # Load or create session
    loaded_session: Session | None = None
    if session:
        if session.exists():
            loaded_session = Session.load(session)
            typer.echo(f"Loaded session: {session}")
        else:
            typer.echo(f"Session file not found: {session}", err=True)
            raise typer.Exit(1)
    elif resume:
        loaded_session = Session.get_latest(config.session_dir)
        if loaded_session:
            typer.echo(f"Resuming session: {loaded_session.metadata.id}")
        else:
            typer.echo("No previous session found", err=True)

    # Determine mode
    if prompt or headless:
        # Headless mode - run single prompt
        if not prompt:
            typer.echo("Error: Prompt required in headless mode", err=True)
            raise typer.Exit(1)
        asyncio.run(_run_headless(config, prompt, loaded_session, llm_provider))
    else:
        # Interactive TUI mode
        _run_tui(config, loaded_session, llm_provider)


def _resolve_message_id(session: Session, spec: str) -> str | None:
    """Resolve a message id from a spec (id, prefix, index, last)."""
    messages = session.messages
    if not messages:
        return None
    spec = spec.strip()
    if not spec or spec.lower() in {"last", "latest"}:
        return messages[-1].id
    if spec.lower() in {"assistant", "last-assistant"}:
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                return msg.id
        return messages[-1].id
    if spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(messages):
            return messages[idx].id
    for msg in messages:
        if msg.id == spec:
            return msg.id
    matches = [msg for msg in messages if msg.id.startswith(spec)]
    if len(matches) == 1:
        return matches[0].id
    return None


def _resolve_entry_id(session: Session, spec: str) -> str | None:
    """Resolve an entry id across all message entries in a session."""
    entries = list(session.entries)
    if not entries:
        return None
    spec = spec.strip()
    if not spec or spec.lower() in {"last", "latest"}:
        non_state_entries = [e for e in entries if not isinstance(e, SessionStateEntry)]
        if non_state_entries:
            return non_state_entries[-1].id
        return entries[-1].id
    if spec.lower() in {"assistant", "last-assistant"}:
        for entry in reversed(entries):
            if isinstance(entry, MessageEntry) and entry.message.role == Role.ASSISTANT:
                return entry.id
        return entries[-1].id
    for entry in entries:
        if entry.id == spec:
            return entry.id
    matches = [e for e in entries if e.id.startswith(spec)]
    if len(matches) == 1:
        return matches[0].id
    if spec.isdigit():
        msg_entries = [e for e in entries if isinstance(e, MessageEntry)]
        idx = int(spec)
        if 0 <= idx < len(msg_entries):
            return msg_entries[idx].id
    return None


def _run_tui(config: Config, session: Session | None, llm_provider: LLMProvider) -> None:
    """Run the interactive TUI."""
    from agent.tui.app import AgentApp

    app = AgentApp(config, provider=llm_provider, session=session)
    app.run()


async def _run_headless(
    config: Config,
    prompt: str,
    session: Session | None,
    llm_provider: LLMProvider,
) -> None:
    """Run a single prompt without TUI."""
    from agent.core.agent import Agent

    agent = Agent(
        config.to_agent_settings(),
        llm_provider,
        session,
    )

    # Load extensions
    if config.extensions:
        errors = await agent.load_extensions()
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
                    # Optionally print thinking (uncomment to see):
                    # print(chunk.payload.text, end="", flush=True)
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
        print()  # Final newline
    except Exception as e:
        if in_thinking:
            print("\n[/Thinking]", flush=True)
        print(f"\n[Error: {type(e).__name__}: {e}]", flush=True)
    finally:
        await agent.close()


@app.command()
def fork(
    session: Annotated[
        Path | None, typer.Option("-s", "--session", help="Session file to fork")
    ] = None,
    from_message: Annotated[
        str,
        typer.Option(
            "--from",
            help="Message id, id prefix, index, or 'last'/'assistant'",
        ),
    ] = "last",
) -> None:
    """Fork a session from a message and start the TUI."""
    config = Config.load()

    source_session: Session | None = None
    if session:
        if session.exists():
            source_session = Session.load(session)
            typer.echo(f"Loaded session: {session}")
        else:
            typer.echo(f"Session file not found: {session}", err=True)
            raise typer.Exit(1)
    else:
        source_session = Session.get_latest(config.session_dir)
        if not source_session:
            typer.echo("No previous session found", err=True)
            raise typer.Exit(1)

    message_id = _resolve_message_id(source_session, from_message)
    if not message_id:
        typer.echo(f"Could not resolve message: {from_message}", err=True)
        raise typer.Exit(1)

    new_session = source_session.fork(message_id, config.session_dir)
    typer.echo(f"Forked session: {new_session.path} (parent: {source_session.metadata.id})")

    llm_provider = _create_llm_provider(config)
    _run_tui(config, new_session, llm_provider)


@app.command()
def tree(
    session: Annotated[
        Path | None, typer.Option("-s", "--session", help="Session file to open")
    ] = None,
    to: Annotated[
        str,
        typer.Option(
            "--to",
            help="Message id, id prefix, index, or 'last'/'assistant'",
        ),
    ] = "last",
) -> None:
    """Move the session leaf to an entry and start the TUI."""
    config = Config.load()

    target_session: Session | None = None
    if session:
        if session.exists():
            target_session = Session.load(session)
            typer.echo(f"Loaded session: {session}")
        else:
            typer.echo(f"Session file not found: {session}", err=True)
            raise typer.Exit(1)
    else:
        target_session = Session.get_latest(config.session_dir)
        if not target_session:
            typer.echo("No previous session found", err=True)
            raise typer.Exit(1)

    message_id = _resolve_entry_id(target_session, to)
    if not message_id:
        typer.echo(f"Could not resolve message: {to}", err=True)
        raise typer.Exit(1)

    try:
        target_session.set_leaf(message_id)
    except Exception as exc:
        typer.echo(f"Failed to set leaf: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Set session leaf: {message_id}")

    llm_provider = _create_llm_provider(config)
    _run_tui(config, target_session, llm_provider)


@app.command()
def sessions(
    limit: Annotated[int, typer.Option("-n", "--limit", help="Number of sessions")] = 10,
) -> None:
    """List recent sessions."""
    config = Config.load()
    session_list = Session.list_sessions(config.session_dir, limit)

    if not session_list:
        typer.echo("No sessions found")
        return

    typer.echo(f"Recent sessions (in {config.session_dir}):\n")
    for path in session_list:
        try:
            sess = Session.load(path)
            msg_count = len(sess.messages)
            typer.echo(f"  {path.name}")
            typer.echo(f"    ID: {sess.metadata.id}")
            typer.echo(f"    Created: {sess.metadata.created_at}")
            typer.echo(f"    Messages: {msg_count}")
            typer.echo(f"    CWD: {sess.metadata.cwd}")
            typer.echo()
        except Exception as e:
            typer.echo(f"  {path.name} (error: {e})")


@app.command()
def config_show() -> None:
    """Show current configuration."""
    config = Config.load()

    typer.echo("Current configuration:")
    typer.echo(f"  Provider: {config.provider}")
    typer.echo(f"  Model: {config.model}")
    typer.echo(f"  API Key: {'[set]' if config.api_key else '[not set]'}")
    typer.echo(f"  Base URL: {config.base_url or '[default]'}")
    typer.echo(f"  Context Tokens: {config.context_max_tokens}")
    typer.echo(f"  Max Output Tokens: {config.max_output_tokens}")
    typer.echo(f"  Temperature: {config.temperature}")
    typer.echo(f"  Thinking Level: {config.thinking_level}")
    typer.echo(f"  Session Dir: {config.session_dir}")
    typer.echo(f"  Skills Dirs: {config.skills_dirs or '[none]'}")
    typer.echo(f"  Extensions: {config.extensions or '[none]'}")

    if config.providers:
        typer.echo("\nConfigured providers:")
        for name, prov in config.providers.items():
            typer.echo(f"  {name}:")
            typer.echo(f"    Base URL: {prov.base_url}")
            typer.echo(f"    Model: {prov.model}")


if __name__ == "__main__":
    main()
