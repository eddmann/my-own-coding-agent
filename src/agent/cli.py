"""CLI entry point using Typer."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from agent.core.config import Config, ThinkingLevel
from agent.core.message import Message, ThinkingContent, ToolCall
from agent.core.session import Session

app = typer.Typer(
    name="agent",
    help="A Python AI coding agent",
    no_args_is_help=False,
)


def main() -> None:
    """Entry point for the CLI."""
    app()


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
        asyncio.run(run_headless(config, prompt, loaded_session))
    else:
        # Interactive TUI mode
        run_tui(config, loaded_session)


def run_tui(config: Config, session: Session | None) -> None:
    """Run the interactive TUI."""
    from agent.tui.app import AgentApp

    app = AgentApp(config, session)
    app.run()


async def run_headless(config: Config, prompt: str, session: Session | None) -> None:
    """Run a single prompt without TUI."""
    from agent.core.agent import Agent

    agent = Agent(config, session)

    # Load extensions
    if config.extensions:
        errors = await agent.load_extensions()
        for error in errors:
            print(f"[Extension error: {error}]", flush=True)

    in_thinking = False
    try:
        async for chunk in agent.run(prompt):
            if isinstance(chunk, ThinkingContent):
                if not in_thinking:
                    print("\n[Thinking...]", flush=True)
                    in_thinking = True
                # Optionally print thinking (uncomment to see):
                # print(chunk.text, end="", flush=True)
            elif isinstance(chunk, str):
                if in_thinking:
                    print("\n[/Thinking]", flush=True)
                    in_thinking = False
                print(chunk, end="", flush=True)
            elif isinstance(chunk, ToolCall):
                if in_thinking:
                    print("\n[/Thinking]", flush=True)
                    in_thinking = False
                print(f"\n[Tool: {chunk.name}]", flush=True)
            elif isinstance(chunk, Message) and chunk.role.value == "system":
                if in_thinking:
                    print("\n[/Thinking]", flush=True)
                    in_thinking = False
                print(f"\n[System] {chunk.content}", flush=True)
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
