"""Session modals for loading and forking sessions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, ListItem, ListView, Static, Tree

from agent.core.message import Role
from agent.core.session import (
    CompactionEntry,
    MessageEntry,
    ModelChangeEntry,
    Session,
    SessionEntry,
    SessionStateEntry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult


@dataclass(slots=True)
class SessionOption:
    label: str
    value: str


class SessionLoadModal(ModalScreen[None]):
    """Modal for selecting a session to load."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        session_dir: Path,
        on_load: Callable[[Path], None],
        limit: int = 50,
    ) -> None:
        super().__init__()
        self._session_dir = session_dir
        self._on_load = on_load
        self._options = self._build_options(limit)
        self._selected_path: str | None = self._options[0].value if self._options else None

    def compose(self) -> ComposeResult:
        with Container(id="session-modal"):
            yield Static("Load Session", id="session-title")
            with VerticalScroll(id="session-content"):
                if not self._options:
                    yield Static("No sessions found.", id="session-empty")
                else:
                    yield self._build_session_list()
                yield Button(
                    "Load",
                    id="session-load-btn",
                    variant="primary",
                    disabled=not self._options,
                )
            yield Static("Press [bold]ESC[/] to close", id="session-hint")

    async def on_mount(self) -> None:
        """Focus the list on mount."""
        with contextlib.suppress(Exception):
            self.query_one("#session-list", ListView).focus()

    def action_close(self) -> None:
        """Close the modal."""
        self.app.pop_screen()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle session selection changes."""
        if event.list_view.id == "session-list" and event.item and event.item.name:
            self._selected_path = event.item.name

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "session-load-btn":
            if self._selected_path:
                self._on_load(Path(self._selected_path))
            self.app.pop_screen()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle immediate selection via enter/click."""
        if event.list_view.id == "session-list" and event.item.name:
            self._selected_path = event.item.name
            self._on_load(Path(self._selected_path))
            self.app.pop_screen()

    def _build_options(self, limit: int) -> list[SessionOption]:
        options: list[SessionOption] = []
        for path in Session.list_sessions(self._session_dir, limit=limit):
            try:
                sess = Session.load(path)
                created = sess.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S")
                msg_count = len(sess.messages)
                cwd = sess.metadata.cwd
                label = f"{created} | id:{sess.metadata.id} | msgs:{msg_count} | cwd:{cwd}"
            except Exception:
                label = f"{path.name} | (error reading session)"
            options.append(SessionOption(label=label, value=str(path)))
        return options

    def _build_session_list(self) -> ListView:
        items: list[ListItem] = []
        values: list[str] = []
        for opt in self._options:
            items.append(ListItem(Static(opt.label), name=opt.value))
            values.append(opt.value)
        if values:
            if self._selected_path not in values:
                self._selected_path = values[0]
            initial_index = values.index(self._selected_path)
        else:
            initial_index = None
        return ListView(*items, id="session-list", initial_index=initial_index)


class SessionForkModal(ModalScreen[None]):
    """Modal for selecting a message to fork from."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        session: Session,
        on_fork: Callable[[str], None],
        limit: int = 200,
    ) -> None:
        super().__init__()
        self._session = session
        self._on_fork = on_fork
        self._options = self._build_message_options(limit)
        self._selected_id: str | None = self._default_message_id()
        if self._selected_id and self._options:
            valid = {opt.value for opt in self._options}
            if self._selected_id not in valid:
                self._selected_id = self._options[-1].value

    def compose(self) -> ComposeResult:
        with Container(id="fork-modal"):
            yield Static("Fork Session", id="fork-title")
            with VerticalScroll(id="fork-content"):
                yield Static(self._build_summary(), id="fork-summary")
                if not self._options:
                    yield Static("No messages available to fork.", id="fork-empty")
                else:
                    yield self._build_message_list()
                yield Button(
                    "Fork",
                    id="fork-btn",
                    variant="primary",
                    disabled=not self._options,
                )
            yield Static("Press [bold]ESC[/] to close", id="fork-hint")

    async def on_mount(self) -> None:
        """Focus the list on mount."""
        with contextlib.suppress(Exception):
            self.query_one("#fork-list", ListView).focus()

    def action_close(self) -> None:
        """Close the modal."""
        self.app.pop_screen()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle message selection changes."""
        if event.list_view.id == "fork-list" and event.item and event.item.name:
            self._selected_id = event.item.name

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "fork-btn":
            if self._selected_id:
                self._on_fork(self._selected_id)
            self.app.pop_screen()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle immediate selection via enter/click."""
        if event.list_view.id == "fork-list" and event.item.name:
            self._selected_id = event.item.name
            self._on_fork(self._selected_id)
            self.app.pop_screen()

    def _build_summary(self) -> str:
        msg_count = len(self._session.messages)
        return f"Session {self._session.metadata.id} | {msg_count} message(s)"

    def _default_message_id(self) -> str | None:
        for msg in reversed(self._session.messages):
            if msg.role == Role.ASSISTANT:
                return msg.id
        if self._session.messages:
            return self._session.messages[-1].id
        return None

    def _build_message_options(self, limit: int) -> list[SessionOption]:
        options: list[SessionOption] = []
        messages = self._session.messages
        start = max(0, len(messages) - limit)
        for idx, msg in enumerate(messages[start:], start=start):
            snippet = msg.content.replace("\n", " ").strip()
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            label = f"[{idx}] {msg.role.value.upper():<9} {msg.id} {snippet}"
            options.append(SessionOption(label=label, value=msg.id))
        return options

    def _build_message_list(self) -> ListView:
        items: list[ListItem] = []
        values: list[str] = []
        for opt in self._options:
            items.append(ListItem(Static(opt.label), name=opt.value))
            values.append(opt.value)
        if values:
            if self._selected_id not in values:
                self._selected_id = values[-1]
            initial_index = values.index(self._selected_id)
        else:
            initial_index = None
        return ListView(*items, id="fork-list", initial_index=initial_index)


class SessionTreeModal(ModalScreen[None]):
    """Modal for selecting a leaf entry in the session tree."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        session: Session,
        on_select: Callable[[str], None],
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._on_select = on_select
        self._limit = limit
        self._selected_id: str | None = self._default_leaf_id()
        self._tree_nodes: dict[str, Any] = {}
        self._tree: Tree[str] | None = None
        self._linear_ids: list[str] = []
        self._is_linear = self._is_linear_session()
        self._suppress_tree_select = False

    def compose(self) -> ComposeResult:
        with Container(id="tree-modal"):
            yield Static("Session Tree", id="tree-title")
            with VerticalScroll(id="tree-content"):
                yield Static(self._build_summary(), id="tree-summary")
                if not self._session.entries:
                    yield Static("No entries available.", id="tree-empty")
                elif self._is_linear:
                    yield self._build_linear_list()
                else:
                    yield self._build_tree()
                yield Button(
                    "Set Leaf",
                    id="tree-btn",
                    variant="primary",
                    disabled=not self._session.entries,
                )
            yield Static("Press [bold]ESC[/] to close", id="tree-hint")

    async def on_mount(self) -> None:
        """Focus the tree on mount."""
        if self._tree:
            self._tree.focus()
        else:
            with contextlib.suppress(Exception):
                self.query_one("#linear-list", ListView).focus()

    def action_close(self) -> None:
        """Close the modal."""
        self.app.pop_screen()

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """Handle immediate selection via enter/click."""
        if self._suppress_tree_select:
            return
        if event.node.data:
            self._selected_id = str(event.node.data)
            self._on_select(self._selected_id)
            self.app.pop_screen()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        """Handle tree highlight changes."""
        if event.node.data:
            self._selected_id = str(event.node.data)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle linear list highlight changes."""
        if event.list_view.id == "linear-list" and event.item and event.item.name:
            self._selected_id = event.item.name

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle immediate selection via enter/click."""
        if event.list_view.id == "linear-list" and event.item and event.item.name:
            self._selected_id = event.item.name
            self._on_select(self._selected_id)
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "tree-btn":
            if self._selected_id:
                self._on_select(self._selected_id)
            self.app.pop_screen()

    def _build_summary(self) -> str:
        leaf = self._session.leaf_id or "(unknown)"
        entry_count = len(
            [e for e in self._session.entries if not isinstance(e, SessionStateEntry)]
        )
        return f"Session {self._session.metadata.id} | entries: {entry_count} | leaf: {leaf}"

    def _default_leaf_id(self) -> str | None:
        return self._session.leaf_id

    def _is_linear_session(self) -> bool:
        children: dict[str | None, int] = {}
        for entry in self._session.entries:
            if isinstance(entry, SessionStateEntry):
                continue
            parent = entry.parent_id
            children[parent] = children.get(parent, 0) + 1
            if children[parent] > 1:
                return False
        return True

    def _build_linear_list(self) -> ListView:
        entries: list[SessionEntry] = [
            e for e in self._session.entries if not isinstance(e, SessionStateEntry)
        ]
        if self._limit is not None and len(entries) > self._limit:
            entries = entries[-self._limit :]
        items: list[ListItem] = []
        self._linear_ids = []
        for entry in entries:
            label = self._format_entry(entry)
            items.append(ListItem(Static(label), name=entry.id))
            self._linear_ids.append(entry.id)
        if self._linear_ids:
            if self._selected_id not in self._linear_ids:
                self._selected_id = self._linear_ids[-1]
            initial_index = self._linear_ids.index(self._selected_id)
        else:
            initial_index = None
        return ListView(*items, id="linear-list", initial_index=initial_index)

    def _build_tree(self) -> Tree[str]:
        tree: Tree[str] = Tree(f"Session {self._session.metadata.id}", id="tree-view")
        tree.show_root = False
        self._tree = tree

        entries: list[SessionEntry] = [
            e for e in self._session.entries if not isinstance(e, SessionStateEntry)
        ]
        if self._limit is not None and len(entries) > self._limit:
            entries = entries[-self._limit :]

        children: dict[str | None, list[SessionEntry]] = {}
        for entry in entries:
            children.setdefault(entry.parent_id, []).append(entry)

        def add_children(parent_node: Any, parent_id: str | None) -> None:
            for entry in children.get(parent_id, []):
                label = self._format_entry(entry)
                node = parent_node.add(label, data=entry.id)
                self._tree_nodes[entry.id] = node
                add_children(node, entry.id)

        add_children(tree.root, None)
        tree.root.expand()

        if self._session.leaf_id and self._session.leaf_id in self._tree_nodes:
            node = self._tree_nodes[self._session.leaf_id]
            current = node
            while current and current.parent:
                current.parent.expand()
                current = current.parent
            self._suppress_tree_select = True
            tree.select_node(node)
            tree.call_after_refresh(self._clear_tree_select_suppression)
            tree.scroll_to_node(node, animate=False)

        return tree

    def _clear_tree_select_suppression(self) -> None:
        self._suppress_tree_select = False

    def _format_entry(self, entry: SessionEntry) -> str:
        leaf_marker = " (leaf)" if entry.id == self._session.leaf_id else ""
        if isinstance(entry, MessageEntry):
            msg = entry.message
            snippet = msg.content.replace("\n", " ").strip()
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            return f"{msg.role.value.upper()} {entry.id} {snippet}{leaf_marker}"
        if isinstance(entry, ModelChangeEntry):
            return f"MODEL {entry.id} {entry.provider}/{entry.model_id}{leaf_marker}"
        if isinstance(entry, CompactionEntry):
            summary = entry.summary.replace("\n", " ").strip()
            if len(summary) > 60:
                summary = summary[:57] + "..."
            tokens = ""
            if entry.tokens_before is not None or entry.tokens_after is not None:
                tokens = f" {entry.tokens_before}->{entry.tokens_after}"
            return f"COMPACT {entry.id}{tokens} {summary}{leaf_marker}"
        if isinstance(entry, SessionStateEntry):
            return f"STATE {entry.id} leaf={entry.leaf_id}{leaf_marker}"
        return f"{entry.type} {entry.id}{leaf_marker}"
