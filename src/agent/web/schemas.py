"""HTTP and WebSocket payload models for web delivery."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class HealthResponse(BaseModel):
    """Basic health response."""

    status: Literal["ok"] = "ok"


class SessionSummary(BaseModel):
    """Web-facing session summary."""

    id: str
    path: str
    created_at: str
    cwd: str
    leaf_id: str | None = None
    parent_session_id: str | None = None
    provider: str | None = None
    model: str | None = None


class RunStartMessage(BaseModel):
    """Start one runtime invocation."""

    type: Literal["run.start"]
    input: str
    run_id: str | None = None


class RunCancelMessage(BaseModel):
    """Cancel the active runtime invocation."""

    type: Literal["run.cancel"]
    run_id: str | None = None


class SessionNewMessage(BaseModel):
    """Create and activate a new session."""

    type: Literal["session.new"]


class SessionLoadMessage(BaseModel):
    """Load a session by id."""

    type: Literal["session.load"]
    session_id: str


class SessionForkMessage(BaseModel):
    """Fork the active session from a message id."""

    type: Literal["session.fork"]
    message_id: str


class SessionSetLeafMessage(BaseModel):
    """Move the active session leaf to an entry id."""

    type: Literal["session.set_leaf"]
    entry_id: str


class ModelSetMessage(BaseModel):
    """Switch the active model."""

    type: Literal["model.set"]
    model: str


class ModelSetThinkingMessage(BaseModel):
    """Switch the active thinking level."""

    type: Literal["model.set_thinking"]
    level: Literal["off", "minimal", "low", "medium", "high"]


class UIResponseMessage(BaseModel):
    """Respond to an interactive extension UI request."""

    type: Literal["ui.response"]
    request_id: str
    value: str | bool | None = None


class UIViewActionMessage(BaseModel):
    """Trigger an action on a presented extension view."""

    type: Literal["ui.view_action"]
    view_id: str
    action: str
    value: str | None = None


type ClientMessage = Annotated[
    RunStartMessage
    | RunCancelMessage
    | SessionNewMessage
    | SessionLoadMessage
    | SessionForkMessage
    | SessionSetLeafMessage
    | ModelSetMessage
    | ModelSetThinkingMessage
    | UIResponseMessage
    | UIViewActionMessage,
    Field(discriminator="type"),
]

_CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(data: object) -> ClientMessage:
    """Validate one inbound WebSocket message."""
    return _CLIENT_MESSAGE_ADAPTER.validate_python(data)
