"""Session helpers for web delivery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.runtime.session import Session
from agent.web.schemas import SessionSummary

if TYPE_CHECKING:
    from pathlib import Path


def summarize_session(session: Session) -> SessionSummary:
    """Build a stable web-facing session summary."""
    metadata = session.metadata
    return SessionSummary(
        id=metadata.id,
        path=str(session.path),
        created_at=metadata.created_at.isoformat(),
        cwd=metadata.cwd,
        leaf_id=session.leaf_id,
        parent_session_id=metadata.parent_session_id,
        provider=metadata.provider,
        model=metadata.model,
    )


def list_session_summaries(session_dir: Path, limit: int = 20) -> list[SessionSummary]:
    """List recent sessions as summaries."""
    summaries: list[SessionSummary] = []
    for path in Session.list_sessions(session_dir, limit=limit):
        summaries.append(summarize_session(Session.load(path)))
    return summaries


def load_session_by_id(session_dir: Path, session_id: str) -> Session | None:
    """Load a session by its id."""
    for path in Session.list_sessions(session_dir, limit=1000):
        session = Session.load(path)
        if session.metadata.id == session_id:
            return session
    return None
