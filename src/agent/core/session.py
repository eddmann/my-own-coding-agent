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

SESSION_VERSION = 1


class SessionMetadata(BaseModel):
    """Session header metadata."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session"] = "session"
    version: int = SESSION_VERSION
    id: str
    created_at: datetime = Field(alias="timestamp")
    cwd: str
    parent_session_id: str | None = Field(default=None, alias="parentSession")
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


class CompactionEntry(SessionEntryBase):
    """Session entry recording a compaction event."""

    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str | None = Field(default=None, alias="firstKeptEntryId")
    tokens_before: int | None = Field(default=None, alias="tokensBefore")
    tokens_after: int | None = Field(default=None, alias="tokensAfter")


class SessionStateEntry(SessionEntryBase):
    """Session entry recording the active leaf."""

    type: Literal["session_state"] = "session_state"
    leaf_id: str | None = Field(default=None, alias="leafId")


SessionEntry = MessageEntry | ModelChangeEntry | CompactionEntry | SessionStateEntry


class SessionContext(BaseModel):
    """Resolved session context (messages + last model selection)."""

    messages: list[Message]
    model: tuple[str | None, str | None] | None


class Session:
    """Manages conversation history with JSONL persistence."""

    __slots__ = ("path", "metadata", "messages", "entries", "_leaf_id", "_entries_by_id")

    def __init__(self, path: Path, metadata: SessionMetadata) -> None:
        self.path = path
        self.metadata = metadata
        self.messages: list[Message] = []
        self.entries: list[SessionEntry] = []
        self._leaf_id: str | None = None
        self._entries_by_id: dict[str, SessionEntry] = {}

    @classmethod
    def new(cls, session_dir: Path, provider: str | None = None, model: str | None = None) -> Self:
        """Create a new session."""
        session_id = uuid4().hex[:8]
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        path = session_dir / filename

        metadata = SessionMetadata(
            id=session_id,
            timestamp=datetime.now(UTC),
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
                    session._add_entry(entry)
                    if isinstance(entry, SessionStateEntry):
                        if entry.leaf_id:
                            session._leaf_id = entry.leaf_id
                    else:
                        session._leaf_id = entry.id
            if session._leaf_id and session._leaf_id not in session._entries_by_id:
                fallback = next(
                    (e for e in reversed(session.entries) if not isinstance(e, SessionStateEntry)),
                    None,
                )
                session._leaf_id = fallback.id if fallback else None
            session._rebuild_messages()
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
            timestamp=new_session.metadata.created_at,
            cwd=new_session.metadata.cwd,
            parentSession=self.metadata.id,
        )
        # Rewrite header with parent info
        new_session._write_header()

        prev_id: str | None = None
        for msg in self.messages[: idx + 1]:
            new_id = uuid4().hex[:12]
            new_msg = msg.model_copy(update={"id": new_id, "parent_id": prev_id})
            new_session.append(new_msg)
            prev_id = new_id

        return new_session

    def update_model_metadata(self, provider: str | None, model: str | None) -> None:
        """Update session metadata for provider/model and rewrite header."""
        self.metadata = self.metadata.model_copy(update={"provider": provider, "model": model})
        self._rewrite_file()

    def append_model_change(self, provider: str, model: str) -> None:
        """Record a model change as a session entry."""
        self._append_model_change_entry(provider=provider, model=model, persist=True)

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
    ) -> None:
        """Record a compaction event and rebuild in-memory messages."""
        entry = self._append_compaction_entry(
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            persist=True,
        )
        self._leaf_id = entry.id
        self._rebuild_messages()

    def set_leaf(self, leaf_id: str) -> None:
        """Set the active leaf (appends a session_state entry)."""
        if leaf_id not in self._entries_by_id:
            raise ValueError(f"Unknown leaf id: {leaf_id}")
        self._append_session_state_entry(leaf_id, persist=True)
        self._leaf_id = leaf_id
        self._rebuild_messages()

    def get_model_selection(self) -> tuple[str | None, str | None] | None:
        """Get the latest (provider, model) from session entries/messages."""
        provider = self.metadata.provider
        model = self.metadata.model

        for entry in self._branch_entries(self._leaf_id):
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
            f.write(self.metadata.model_dump_json(by_alias=True, exclude_none=True) + "\n")

    def _rewrite_file(self) -> None:
        """Rewrite header + all entries."""
        self._write_header()
        with open(self.path, "a") as f:
            for entry in self.entries:
                f.write(self._serialize_entry(entry) + "\n")

    @classmethod
    def _parse_entry(cls, line: str) -> SessionEntry | None:
        """Parse a JSONL entry."""
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
            if entry_type == "compaction":
                return CompactionEntry.model_validate(data)
            if entry_type == "session_state":
                return SessionStateEntry.model_validate(data)
            return None

        return None

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
        self._add_entry(entry)
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
        self._add_entry(entry)
        self._leaf_id = entry.id
        if persist:
            with open(self.path, "a") as f:
                f.write(self._serialize_entry(entry) + "\n")
        return entry

    def _append_compaction_entry(
        self,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        persist: bool = True,
    ) -> CompactionEntry:
        """Append a compaction entry (optionally persist)."""
        entry = CompactionEntry(
            id=uuid4().hex[:12],
            parentId=self._leaf_id,
            timestamp=datetime.now(UTC),
            summary=summary,
            firstKeptEntryId=first_kept_entry_id,
            tokensBefore=tokens_before,
            tokensAfter=tokens_after,
        )
        self._add_entry(entry)
        if persist:
            with open(self.path, "a") as f:
                f.write(self._serialize_entry(entry) + "\n")
        return entry

    def _append_session_state_entry(
        self,
        leaf_id: str | None,
        persist: bool = True,
    ) -> SessionStateEntry:
        """Append a session state entry (optionally persist)."""
        entry = SessionStateEntry(
            id=uuid4().hex[:12],
            parentId=self._leaf_id,
            timestamp=datetime.now(UTC),
            leafId=leaf_id,
        )
        self._add_entry(entry)
        if persist:
            with open(self.path, "a") as f:
                f.write(self._serialize_entry(entry) + "\n")
        return entry

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    @property
    def leaf_id(self) -> str | None:
        """Get the current leaf entry id."""
        return self._leaf_id

    def _add_entry(self, entry: SessionEntry) -> None:
        self.entries.append(entry)
        self._entries_by_id[entry.id] = entry

    def _branch_entries(self, leaf_id: str | None) -> list[SessionEntry]:
        if not leaf_id:
            return []
        branch: list[SessionEntry] = []
        current: str | None = leaf_id
        seen: set[str] = set()
        while current:
            if current in seen:
                break
            seen.add(current)
            entry = self._entries_by_id.get(current)
            if not entry:
                break
            branch.append(entry)
            current = entry.parent_id
        branch.reverse()
        return branch

    def _rebuild_messages(self) -> None:
        branch_entries = self._branch_entries(self._leaf_id)
        message_entries = [e for e in branch_entries if isinstance(e, MessageEntry)]
        compactions = [e for e in branch_entries if isinstance(e, CompactionEntry)]

        if not compactions:
            self.messages = [e.message for e in message_entries]
            return

        compaction = compactions[-1]

        messages: list[Message] = []
        seen_ids: set[str] = set()

        # Always include system messages from the active branch
        for msg_entry in message_entries:
            msg = msg_entry.message
            if msg.role == Role.SYSTEM and msg.id not in seen_ids:
                messages.append(msg)
                seen_ids.add(msg.id)

        summary_text = compaction.summary.strip()
        if summary_text:
            summary_msg = Message(
                role=Role.SYSTEM,
                content=f"[Previous conversation summary]\n{summary_text}",
            )
            messages.append(summary_msg)

        start_index: int | None = None
        if compaction.first_kept_entry_id:
            for idx, msg_entry in enumerate(message_entries):
                if msg_entry.message.id == compaction.first_kept_entry_id:
                    start_index = idx
                    break

        if start_index is None:
            after_compaction = False
            for branch_entry in branch_entries:
                if branch_entry is compaction:
                    after_compaction = True
                    continue
                if after_compaction and isinstance(branch_entry, MessageEntry):
                    msg = branch_entry.message
                    if msg.role != Role.SYSTEM and msg.id not in seen_ids:
                        messages.append(msg)
                        seen_ids.add(msg.id)
            self.messages = messages
            return

        for msg_entry in message_entries[start_index:]:
            msg = msg_entry.message
            if msg.role == Role.SYSTEM or msg.id in seen_ids:
                continue
            messages.append(msg)
            seen_ids.add(msg.id)

        self.messages = messages
