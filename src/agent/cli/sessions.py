"""CLI session command helpers and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from agent.config import Config
from agent.runtime.message import Role
from agent.runtime.session import MessageEntry, Session, SessionStateEntry

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def resolve_message_id(session: Session, spec: str) -> str | None:
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


def resolve_entry_id(session: Session, spec: str) -> str | None:
    """Resolve an entry id across all message entries in a session."""
    entries = list(session.entries)
    if not entries:
        return None
    spec = spec.strip()
    if not spec or spec.lower() in {"last", "latest"}:
        non_state_entries = [entry for entry in entries if not isinstance(entry, SessionStateEntry)]
        if non_state_entries:
            return non_state_entries[-1].id
        return entries[-1].id
    if spec.lower() in {"assistant", "last-assistant"}:
        for entry in reversed(entries):
            if isinstance(entry, MessageEntry) and entry.message.role == Role.ASSISTANT:
                return entry.id
        return entries[-1].id
    if spec.isdigit():
        msg_entries = [entry for entry in entries if isinstance(entry, MessageEntry)]
        idx = int(spec)
        if 0 <= idx < len(msg_entries):
            return msg_entries[idx].id
    for entry in entries:
        if entry.id == spec:
            return entry.id
    matches = [entry for entry in entries if entry.id.startswith(spec)]
    if len(matches) == 1:
        return matches[0].id
    return None


def fork_command(
    session: Path | None,
    from_message: str,
    *,
    create_llm_provider: Callable[[Config], object],
    run_tui: Callable[..., None],
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

    message_id = resolve_message_id(source_session, from_message)
    if not message_id:
        typer.echo(f"Could not resolve message: {from_message}", err=True)
        raise typer.Exit(1)

    new_session = source_session.fork(message_id, config.session_dir)
    typer.echo(f"Forked session: {new_session.path} (parent: {source_session.metadata.id})")

    llm_provider = create_llm_provider(config)
    run_tui(config, new_session, llm_provider)


def tree_command(
    session: Path | None,
    to: str,
    *,
    create_llm_provider: Callable[[Config], object],
    run_tui: Callable[..., None],
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

    message_id = resolve_entry_id(target_session, to)
    if not message_id:
        typer.echo(f"Could not resolve message: {to}", err=True)
        raise typer.Exit(1)

    try:
        target_session.set_leaf(message_id)
    except Exception as exc:
        typer.echo(f"Failed to set leaf: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Set session leaf: {message_id}")

    target_session = Session.load(target_session.path)

    llm_provider = create_llm_provider(config)
    run_tui(config, target_session, llm_provider)


def sessions_command(limit: int) -> None:
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
        except Exception as exc:
            typer.echo(f"  {path.name} (error: {exc})")
