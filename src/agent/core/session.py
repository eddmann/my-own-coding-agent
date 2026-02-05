"""Session management with JSONL persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterator

from agent.core.message import Message, Role


class SessionMetadata(BaseModel):
    """Session header metadata."""

    id: str
    created_at: datetime
    cwd: str
    parent_session_id: str | None = None
    provider: str | None = None
    model: str | None = None


class SessionEntryBase(BaseModel):
    """Base session entry."""

    model_config = ConfigDict(populate_by_name=True)

    type: str
    id: str
    parent_id: str | None = Field(default=None, alias="parentId")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MessageEntry(SessionEntryBase):
    """Session entry wrapping a message."""

    type: Literal["message"] = "message"
    message: Message


class ModelChangeEntry(SessionEntryBase):
    """Session entry recording a model change."""

    type: Literal["model_change"] = "model_change"
    provider: str
    model_id: str = Field(alias="modelId")


SessionEntry = MessageEntry | ModelChangeEntry


class SessionContext(BaseModel):
    """Resolved session context (messages + last model selection)."""

    messages: list[Message]
    model: tuple[str | None, str | None] | None


class Session:
    """Manages conversation history with JSONL persistence."""

    __slots__ = ("path", "metadata", "messages", "entries", "_leaf_id")

    def __init__(self, path: Path, metadata: SessionMetadata) -> None:
        self.path = path
        self.metadata = metadata
        self.messages: list[Message] = []
        self.entries: list[SessionEntry] = []
        self._leaf_id: str | None = None

    @classmethod
    def new(cls, session_dir: Path, provider: str | None = None, model: str | None = None) -> Self:
        """Create a new session."""
        session_id = uuid4().hex[:8]
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        path = session_dir / filename

        metadata = SessionMetadata(
            id=session_id,
            created_at=datetime.now(UTC),
            cwd=str(Path.cwd()),
            provider=provider,
            model=model,
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
                    entry = cls._parse_entry(line)
                    if entry is None:
                        continue
                    session.entries.append(entry)
                    session._leaf_id = entry.id
                    if isinstance(entry, MessageEntry):
                        session.messages.append(entry.message)
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
        self._append_message_entry(message, persist=True)

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
        current_model = self.get_model_selection()
        self.messages = []
        self.entries = []
        self._leaf_id = None

        # Rebuild entries (preserve current model selection if available)
        if current_model and current_model[0] and current_model[1]:
            self._append_model_change_entry(
                provider=current_model[0],
                model=current_model[1],
                persist=False,
            )
        for msg in messages:
            self._append_message_entry(msg, persist=False)

        self._rewrite_file()

    def update_model_metadata(self, provider: str | None, model: str | None) -> None:
        """Update session metadata for provider/model and rewrite header."""
        self.metadata = self.metadata.model_copy(update={"provider": provider, "model": model})
        self._rewrite_file()

    def append_model_change(self, provider: str, model: str) -> None:
        """Record a model change as a session entry."""
        self._append_model_change_entry(provider=provider, model=model, persist=True)

    def get_model_selection(self) -> tuple[str | None, str | None] | None:
        """Get the latest (provider, model) from session entries/messages."""
        provider = self.metadata.provider
        model = self.metadata.model

        for entry in self.entries:
            if isinstance(entry, ModelChangeEntry):
                provider = entry.provider
                model = entry.model_id
            elif isinstance(entry, MessageEntry):
                msg = entry.message
                if msg.role == Role.ASSISTANT and msg.model:
                    provider = msg.provider or provider
                    model = msg.model

        if provider or model:
            return provider, model
        return None

    def _write_header(self) -> None:
        """Write session metadata header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(self.metadata.model_dump_json() + "\n")

    def _rewrite_file(self) -> None:
        """Rewrite header + all entries."""
        self._write_header()
        with open(self.path, "a") as f:
            for entry in self.entries:
                f.write(self._serialize_entry(entry) + "\n")

    @classmethod
    def _parse_entry(cls, line: str) -> SessionEntry | None:
        """Parse a JSONL entry, supporting legacy message-only lines."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict) and "type" in data:
            entry_type = data.get("type")
            if entry_type == "message":
                entry = MessageEntry.model_validate(data)
                # Ensure message parent_id aligns with entry parent_id
                if entry.message.parent_id is None and entry.parent_id:
                    entry = entry.model_copy(
                        update={
                            "message": entry.message.model_copy(
                                update={"parent_id": entry.parent_id}
                            )
                        }
                    )
                return entry
            if entry_type == "model_change":
                return ModelChangeEntry.model_validate(data)
            return None

        # Legacy line: plain Message JSON
        message = Message.model_validate(data)
        return MessageEntry(
            id=message.id,
            parentId=message.parent_id,
            timestamp=message.timestamp,
            message=message,
        )

    @staticmethod
    def _serialize_entry(entry: SessionEntry) -> str:
        return entry.model_dump_json(by_alias=True, exclude_none=True)

    def _append_message_entry(self, message: Message, persist: bool = True) -> MessageEntry:
        """Append a message entry to session (optionally persist)."""
        if message.parent_id is None and self._leaf_id is not None:
            message = message.model_copy(update={"parent_id": self._leaf_id})

        entry = MessageEntry(
            id=message.id,
            parentId=message.parent_id,
            timestamp=message.timestamp,
            message=message,
        )
        self.entries.append(entry)
        self.messages.append(entry.message)
        self._leaf_id = entry.id
        if persist:
            with open(self.path, "a") as f:
                f.write(self._serialize_entry(entry) + "\n")
        return entry

    def _append_model_change_entry(
        self, provider: str, model: str, persist: bool = True
    ) -> ModelChangeEntry:
        """Append a model change entry (optionally persist)."""
        entry = ModelChangeEntry(
            id=uuid4().hex[:12],
            parentId=self._leaf_id,
            timestamp=datetime.now(UTC),
            provider=provider,
            modelId=model,
        )
        self.entries.append(entry)
        self._leaf_id = entry.id
        if persist:
            with open(self.path, "a") as f:
                f.write(self._serialize_entry(entry) + "\n")
        return entry

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
