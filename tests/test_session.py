"""Tests for the session module."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.core.message import Message, Role
from agent.core.session import Session, SessionMetadata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestSession:
    """Tests for Session class."""

    def test_new_session(self, temp_dir: Path):
        """Test creating a new session."""
        session = Session.new(temp_dir)

        assert session.path.exists()
        assert session.metadata.id
        assert session.metadata.cwd == str(Path.cwd())
        assert len(session.messages) == 0

    def test_append_message(self, temp_dir: Path):
        """Test appending a message."""
        session = Session.new(temp_dir)

        msg = Message.user("Hello")
        session.append(msg)

        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"
        assert session.messages[0].role == Role.USER

    def test_load_session(self, temp_dir: Path):
        """Test loading an existing session."""
        # Create a session and add messages
        session = Session.new(temp_dir)
        session.append(Message.user("Hello"))
        session.append(Message.assistant("Hi there"))

        # Load the session
        loaded = Session.load(session.path)

        assert loaded.metadata.id == session.metadata.id
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].content == "Hi there"

    def test_list_sessions(self, temp_dir: Path):
        """Test listing sessions."""
        # Create multiple sessions
        Session.new(temp_dir)
        Session.new(temp_dir)
        Session.new(temp_dir)

        sessions = Session.list_sessions(temp_dir)

        assert len(sessions) == 3

    def test_get_latest(self, temp_dir: Path):
        """Test getting the latest session."""
        # Create sessions
        Session.new(temp_dir)
        latest = Session.new(temp_dir)
        latest.append(Message.user("Latest"))

        retrieved = Session.get_latest(temp_dir)

        assert retrieved is not None
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Latest"

    def test_fork_session(self, temp_dir: Path):
        """Test forking a session."""
        # Create original session
        original = Session.new(temp_dir)
        msg1 = Message.user("First")
        msg2 = Message.assistant("Second")
        msg3 = Message.user("Third")
        original.append(msg1)
        original.append(msg2)
        original.append(msg3)

        # Fork from second message
        forked = original.fork(msg2.id, temp_dir)

        assert forked.metadata.parent_session_id == original.metadata.id
        assert len(forked.messages) == 2  # First two messages
        assert forked.messages[0].content == "First"
        assert forked.messages[1].content == "Second"

        reloaded = Session.load(forked.path)
        assert len(reloaded.messages) == 2
        assert reloaded.messages[0].parent_id is None
        assert reloaded.messages[1].parent_id == reloaded.messages[0].id

    def test_set_leaf_moves_active_branch(self, temp_dir: Path):
        """Set leaf moves the active branch to the selected message."""
        session = Session.new(temp_dir)
        first = Message.user("First")
        second = Message.assistant("Second")
        third = Message.user("Third")
        session.append(first)
        session.append(second)
        session.append(third)

        session.set_leaf(second.id)

        assert [m.content for m in session.messages] == ["First", "Second"]

        loaded = Session.load(session.path)

        assert loaded.leaf_id == second.id
        assert [m.content for m in loaded.messages] == ["First", "Second"]

    def test_compaction_summary_rebuilds_context_from_first_kept(self, temp_dir: Path):
        """Compaction summary keeps messages from firstKeptEntryId onward."""
        session = Session.new(temp_dir)
        msg1 = Message.user("Old 1")
        msg2 = Message.assistant("Old 2")
        msg3 = Message.user("Recent 1")
        msg4 = Message.assistant("Recent 2")
        session.append(msg1)
        session.append(msg2)
        session.append(msg3)
        session.append(msg4)

        summary = "## Summary\n- condensed"
        TOKENS_BEFORE = 100

        session.append_compaction(summary, first_kept_entry_id=msg3.id, tokens_before=TOKENS_BEFORE)

        assert "Previous conversation summary" in session.messages[0].content
        assert [m.content for m in session.messages[1:]] == ["Recent 1", "Recent 2"]

        loaded = Session.load(session.path)

        assert "Previous conversation summary" in loaded.messages[0].content
        assert [m.content for m in loaded.messages[1:]] == ["Recent 1", "Recent 2"]


class TestSessionMetadata:
    """Tests for SessionMetadata class."""

    def test_metadata_creation(self):
        """Test creating session metadata."""
        metadata = SessionMetadata(
            id="test123",
            created_at=datetime.now(UTC),
            cwd="/home/user/project",
        )

        assert metadata.id == "test123"
        assert metadata.cwd == "/home/user/project"
        assert metadata.parent_session_id is None

    def test_metadata_with_parent(self):
        """Test metadata with parent session."""
        metadata = SessionMetadata(
            id="child",
            created_at=datetime.now(UTC),
            cwd="/home/user/project",
            parent_session_id="parent",
        )

        assert metadata.parent_session_id == "parent"
