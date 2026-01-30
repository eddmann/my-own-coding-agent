"""Session management with JSONL persistence."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self
from uuid import uuid4

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Iterator

from agent.core.message import Message


class SessionMetadata(BaseModel):
    """Session header metadata."""

    id: str
    created_at: datetime
    cwd: str
    parent_session_id: str | None = None


class Session:
    """Manages conversation history with JSONL persistence."""

    __slots__ = ("path", "metadata", "messages")

    def __init__(self, path: Path, metadata: SessionMetadata) -> None:
        self.path = path
        self.metadata = metadata
        self.messages: list[Message] = []

    @classmethod
    def new(cls, session_dir: Path) -> Self:
        """Create a new session."""
        session_id = uuid4().hex[:8]
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        path = session_dir / filename

        metadata = SessionMetadata(
            id=session_id,
            created_at=datetime.now(UTC),
            cwd=str(Path.cwd()),
        )

        session = cls(path, metadata)
        session._write_header()
        return session

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load existing session from JSONL file."""
        with open(path) as f:
            # First line is metadata
            first_line = f.readline()
            metadata = SessionMetadata.model_validate_json(first_line)
            session = cls(path, metadata)
            # Remaining lines are messages
            for line in f:
                if line.strip():
                    session.messages.append(Message.model_validate_json(line))
        return session

    @classmethod
    def list_sessions(cls, session_dir: Path, limit: int = 20) -> list[Path]:
        """List recent sessions, sorted by modification time."""
        if not session_dir.exists():
            return []
        sessions = list(session_dir.glob("*.jsonl"))
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions[:limit]

    @classmethod
    def get_latest(cls, session_dir: Path) -> Self | None:
        """Get the most recent session, if any."""
        sessions = cls.list_sessions(session_dir, limit=1)
        if sessions:
            return cls.load(sessions[0])
        return None

    def append(self, message: Message) -> None:
        """Append message and persist to disk."""
        self.messages.append(message)
        with open(self.path, "a") as f:
            f.write(message.model_dump_json() + "\n")

    def fork(self, from_message_id: str, session_dir: Path) -> Self:
        """Create a new session branching from a message."""
        # Find the message index
        idx = next(
            (i for i, m in enumerate(self.messages) if m.id == from_message_id),
            len(self.messages),
        )

        # Create new session with copied history
        new_session = type(self).new(session_dir)
        new_session.metadata = SessionMetadata(
            id=new_session.metadata.id,
            created_at=new_session.metadata.created_at,
            cwd=new_session.metadata.cwd,
            parent_session_id=self.metadata.id,
        )
        # Rewrite header with parent info
        new_session._write_header()

        for msg in self.messages[: idx + 1]:
            new_msg = msg.model_copy(update={"parent_id": msg.id, "id": uuid4().hex[:12]})
            new_session.append(new_msg)

        return new_session

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace all messages (used for compaction). Rewrites the file."""
        self.messages = messages
        self._write_header()
        with open(self.path, "a") as f:
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")

    def _write_header(self) -> None:
        """Write session metadata header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(self.metadata.model_dump_json() + "\n")

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
