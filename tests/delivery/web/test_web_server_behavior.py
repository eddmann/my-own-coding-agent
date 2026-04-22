"""Behavior tests for the web delivery server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from agent.config import Config
from agent.extensions import PresentedView
from agent.llm.events import DoneEvent, TextDeltaEvent, TextStartEvent
from agent.web.server import _WebClientConnection, create_app

if TYPE_CHECKING:
    from agent.runtime.message import Message

from agent.runtime.message import Message, Role
from agent.runtime.session import Session


class FakeProvider:
    """Minimal provider stub for web delivery tests."""

    name = "openai"

    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model

    def set_model(self, model: str) -> None:
        self.model = model

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: object | None = None,
    ):
        async def gen():
            yield TextStartEvent()
            yield TextDeltaEvent(delta="hello from web")
            yield DoneEvent()

        return gen()

    def count_tokens(self, text: str) -> int:
        return len(text)

    def count_messages_tokens(self, messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)

    def supports_thinking(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return [self.model, "gpt-5.3"]

    async def close(self) -> None:
        return None


def _make_config(temp_dir) -> Config:
    return Config(
        provider="openai",
        model="gpt-5.4",
        session_dir=temp_dir / "sessions",
    )


def _seed_session(temp_dir) -> Session:
    session = Session.new(temp_dir / "sessions", provider="openai", model="gpt-5.4")
    session.append(Message(role=Role.SYSTEM, content="system prompt"))
    session.append(Message(role=Role.USER, content="first user"))
    session.append(Message(role=Role.ASSISTANT, content="first reply"))
    session.append(Message(role=Role.USER, content="second user"))
    return session


def test_web_health_endpoint_reports_ok(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_root_serves_browser_shell(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Web Shell" in response.text
    assert "react.development.js" in response.text
    assert "@tailwindcss/browser@4" in response.text
    assert "daisyui@5" in response.text
    assert "Conversation path" in response.text


def test_web_models_endpoint_lists_provider_models(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client:
        response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5.4"
    assert payload["models"] == ["gpt-5.4", "gpt-5.3"]


def test_web_create_session_preserves_provider_and_model_metadata(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client:
        response = client.post("/api/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5.4"


def test_websocket_run_streams_state_and_text_chunks(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "state.snapshot"
        assert snapshot["model"]["name"] == "gpt-5.4"

        websocket.send_json(
            {
                "type": "run.start",
                "run_id": "run_1",
                "input": "hello",
            }
        )

        event_types: list[str] = []
        text_payloads: list[str] = []
        while "run.finished" not in event_types:
            message = websocket.receive_json()
            event_types.append(message["type"])
            if message["type"] == "chunk.text":
                text_payloads.append(message["delta"])

    assert "run.started" in event_types
    assert "chunk.text" in event_types
    assert "run.finished" in event_types
    assert text_payloads == ["hello from web"]


def test_websocket_session_fork_creates_child_session(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    source_session = _seed_session(temp_dir)
    source_message_id = source_session.messages[-1].id
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "session.load", "session_id": source_session.metadata.id})

        changed_session = None
        while changed_session is None:
            message = websocket.receive_json()
            if message["type"] == "session.changed":
                changed_session = message["session"]

        websocket.send_json({"type": "session.fork", "message_id": source_message_id})

        forked_session = None
        while forked_session is None:
            message = websocket.receive_json()
            if message["type"] == "session.changed":
                forked_session = message["session"]

        response = client.get(f"/api/sessions/{forked_session['id']}")

    assert forked_session["id"] != source_session.metadata.id
    assert forked_session["parent_session_id"] == source_session.metadata.id
    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["parent_session_id"] == source_session.metadata.id
    assert len(payload["entries"]) == len(source_session.messages)


def test_websocket_session_set_leaf_updates_active_branch(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    session = _seed_session(temp_dir)
    target_entry_id = session.messages[2].id
    app = create_app(config=_make_config(temp_dir), provider_factory=lambda config: FakeProvider())

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "session.load", "session_id": session.metadata.id})

        session_changed_count = 0
        while session_changed_count < 1:
            message = websocket.receive_json()
            if message["type"] == "session.changed":
                session_changed_count += 1

        websocket.send_json({"type": "session.set_leaf", "entry_id": target_entry_id})

        updated_session = None
        while updated_session is None:
            message = websocket.receive_json()
            if message["type"] == "session.changed":
                updated_session = message["session"]

        response = client.get(f"/api/sessions/{session.metadata.id}")

    assert updated_session["leaf_id"] == target_entry_id
    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["leaf_id"] == target_entry_id
    assert [entry["id"] for entry in payload["active_entries"]] == [
        session.messages[0].id,
        session.messages[1].id,
        target_entry_id,
    ]


def test_websocket_sender_ignores_runtime_error_after_disconnect(temp_dir):
    class ClosedWebSocket:
        def __init__(self) -> None:
            self.calls = 0

        async def send_json(self, payload: dict[str, Any]) -> None:
            self.calls += 1
            raise RuntimeError("websocket already closed")

    websocket = ClosedWebSocket()
    connection = _WebClientConnection(
        websocket,  # type: ignore[arg-type]
        config=_make_config(temp_dir),
        provider_factory=lambda config: FakeProvider(),
    )

    asyncio.run(connection.send_json({"type": "ui.status", "text": "plan mode"}))
    asyncio.run(connection.send_json({"type": "ui.status", "text": "ignored"}))

    assert websocket.calls == 1


def test_websocket_close_cancels_pending_presented_view(temp_dir):
    class ClosedWebSocket:
        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

    class PendingView(PresentedView[None]):
        def render(self) -> str:
            return "Pending"

        def controls(self) -> list[object]:
            return []

        def handle_action(self, action: str, value: str | None = None) -> None:
            return None

        def is_done(self) -> bool:
            return False

        def result(self) -> None:
            return None

    async def exercise() -> bool:
        connection = _WebClientConnection(
            ClosedWebSocket(),  # type: ignore[arg-type]
            config=_make_config(temp_dir),
            provider_factory=lambda config: FakeProvider(),
        )
        connection._active_run_task = asyncio.create_task(connection.bridge.present(PendingView()))
        await asyncio.sleep(0)
        await connection.close()
        return connection._active_run_task.done()

    assert asyncio.run(exercise()) is True
