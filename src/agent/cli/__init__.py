"""CLI entry point using Typer."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from agent.config import Config
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
from agent.runtime.session import Session
from agent.runtime.settings import ThinkingLevel

from .headless import run_headless as _run_headless
from .sessions import fork_command, sessions_command, tree_command

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
    config = Config.load()

    thinking_level = config.thinking_level
    if thinking:
        try:
            thinking_level = ThinkingLevel(thinking.lower())
        except ValueError as err:
            typer.echo(f"Invalid thinking level: {thinking}", err=True)
            typer.echo("Valid values: off, minimal, low, medium, high", err=True)
            raise typer.Exit(1) from err

    extensions = list(config.extensions)
    if extension:
        extensions.extend(Path(ext) for ext in extension)

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

    if prompt or headless:
        if not prompt:
            typer.echo("Error: Prompt required in headless mode", err=True)
            raise typer.Exit(1)
        asyncio.run(_run_headless(config, prompt, loaded_session, llm_provider))
    else:
        _run_tui(config, loaded_session, llm_provider)


def _run_tui(config: Config, session: Session | None, llm_provider: LLMProvider) -> None:
    """Run the interactive TUI."""
    from agent.tui.app import AgentApp

    app = AgentApp(config, provider=llm_provider, session=session)
    app.run()


def _run_web(config: Config, host: str, port: int) -> None:
    """Run the local web delivery server."""
    from agent.web.server import run_web_server

    run_web_server(config, host=host, port=port)


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
    fork_command(
        session,
        from_message,
        create_llm_provider=_create_llm_provider,
        run_tui=_run_tui,
    )


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
    tree_command(
        session,
        to,
        create_llm_provider=_create_llm_provider,
        run_tui=_run_tui,
    )


@app.command()
def web(
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface for the local web server"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port for the local web server"),
    ] = 8000,
) -> None:
    """Run the local web delivery server."""
    config = Config.load()
    _run_web(config, host, port)


@app.command()
def sessions(
    limit: Annotated[int, typer.Option("-n", "--limit", help="Number of sessions")] = 10,
) -> None:
    """List recent sessions."""
    sessions_command(limit)


@app.command()
def config_show() -> None:
    """Show current configuration."""
    config = Config.load()
    model_display = config.model or "[provider default]"

    typer.echo("Current configuration:")
    typer.echo(f"  Provider: {config.provider}")
    typer.echo(f"  Model: {model_display}")
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
