"""FastAPI server for the web delivery shell."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.config import Config
from agent.llm.factory import create_provider
from agent.runtime.chunk import (
    MessageChunk,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolCallStartChunk,
    ToolResultChunk,
)
from agent.runtime.session import Session
from agent.runtime.settings import ThinkingLevel
from agent.web.bridge import WebExtensionBridge
from agent.web.compose import WebRuntime, build_web_runtime
from agent.web.schemas import HealthResponse, parse_client_message
from agent.web.sessions import list_session_summaries, load_session_by_id, summarize_session

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent.llm.provider import LLMProvider


STATIC_DIR = Path(__file__).with_name("static")


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


@dataclass(slots=True)
class _AppState:
    """Shared application state for the web server."""

    config: Config
    provider_factory: Callable[[Config], LLMProvider]


class _WebClientConnection:
    """One active WebSocket client bound to a runtime stack."""

    __slots__ = (
        "_active_run_task",
        "_cancel_event",
        "_closed",
        "_config",
        "_provider_factory",
        "_runtime",
        "_send_lock",
        "_websocket",
        "bridge",
    )

    def __init__(
        self,
        websocket: WebSocket,
        *,
        config: Config,
        provider_factory: Callable[[Config], LLMProvider],
    ) -> None:
        self._websocket = websocket
        self._config = config
        self._provider_factory = provider_factory
        self._send_lock = asyncio.Lock()
        self.bridge = WebExtensionBridge(self.send_json)
        self._runtime: WebRuntime | None = None
        self._active_run_task: asyncio.Task[None] | None = None
        self._cancel_event: asyncio.Event | None = None
        self._closed = False

    async def open(self) -> None:
        """Initialize the runtime stack for the client."""
        provider = self._provider_factory(self._config)
        session = Session.get_latest(self._config.session_dir)
        if session is None:
            session = Session.new(
                self._config.session_dir,
                provider=provider.name,
                model=provider.model,
            )

        self._runtime = build_web_runtime(
            self._config,
            provider,
            session,
            extension_bridge=self.bridge,
        )

        errors = await self._runtime.extension_host.load_extensions()
        for error in errors:
            await self.bridge.notify(error, "error")

        await self._send_snapshot()

    async def close(self) -> None:
        """Tear down the runtime stack for the client."""
        self._closed = True
        await self.bridge.close()
        if self._active_run_task is not None:
            self._cancel_event = asyncio.Event()
            self._cancel_event.set()
            await asyncio.gather(self._active_run_task, return_exceptions=True)
        if self._runtime is not None:
            await self._runtime.agent.close()

    async def send_json(self, payload: dict[str, object]) -> None:
        """Send one JSON message safely to the client."""
        if self._closed:
            return
        async with self._send_lock:
            if self._closed:
                return
            try:
                await self._websocket.send_json(payload)
            except RuntimeError:
                self._closed = True

    async def handle_message(self, data: object) -> None:
        """Dispatch one inbound WebSocket message."""
        message = parse_client_message(data)
        runtime = self._require_runtime()

        match message.type:
            case "run.start":
                run_id = message.run_id or uuid4().hex
                await self._start_run(run_id=run_id, prompt=message.input)
            case "run.cancel":
                if self._cancel_event is not None:
                    self._cancel_event.set()
            case "session.new":
                await runtime.agent.new_session()
                await self.send_json(
                    {
                        "type": "session.changed",
                        "session": self._session_payload(runtime.agent.session),
                    }
                )
                await self._send_snapshot()
            case "session.load":
                session = load_session_by_id(self._config.session_dir, message.session_id)
                if session is None:
                    await self.send_json(
                        {
                            "type": "error",
                            "message": f"Unknown session id: {message.session_id}",
                        }
                    )
                    return
                await runtime.agent.load_session(session)
                await self.send_json(
                    {
                        "type": "session.changed",
                        "session": self._session_payload(runtime.agent.session),
                    }
                )
                await self._send_snapshot()
            case "session.fork":
                try:
                    session = runtime.agent.fork_session(message.message_id)
                    await runtime.agent.load_session(session)
                except ValueError as exc:
                    await self.send_json({"type": "error", "message": str(exc)})
                    return
                await self.send_json(
                    {
                        "type": "session.changed",
                        "session": self._session_payload(runtime.agent.session),
                    }
                )
                await self._send_snapshot()
            case "session.set_leaf":
                try:
                    runtime.agent.set_leaf(message.entry_id)
                except ValueError as exc:
                    await self.send_json({"type": "error", "message": str(exc)})
                    return
                await self.send_json(
                    {
                        "type": "session.changed",
                        "session": self._session_payload(runtime.agent.session),
                    }
                )
                await self._send_snapshot()
            case "model.set":
                try:
                    runtime.agent.set_model(message.model, source="web")
                except ValueError as exc:
                    await self.send_json({"type": "error", "message": str(exc)})
                    return
                await self.send_json(
                    {
                        "type": "model.changed",
                        "model": self._model_payload(runtime.agent),
                    }
                )
                await self._send_snapshot()
            case "model.set_thinking":
                runtime.agent.set_thinking_level(ThinkingLevel(message.level))
                await self.send_json(
                    {
                        "type": "model.changed",
                        "model": self._model_payload(runtime.agent),
                    }
                )
                await self._send_snapshot()
            case "ui.response":
                self.bridge.resolve_request(message.request_id, message.value)
            case "ui.view_action":
                await self.bridge.handle_view_action(
                    message.view_id,
                    message.action,
                    message.value,
                )

    async def _start_run(self, *, run_id: str, prompt: str) -> None:
        if self._active_run_task is not None and not self._active_run_task.done():
            await self.send_json(
                {
                    "type": "run.failed",
                    "run_id": run_id,
                    "error": "run already active",
                }
            )
            return

        self._cancel_event = asyncio.Event()
        self._active_run_task = asyncio.create_task(
            self._run_agent(run_id=run_id, prompt=prompt, cancel_event=self._cancel_event)
        )

    async def _run_agent(
        self,
        *,
        run_id: str,
        prompt: str,
        cancel_event: asyncio.Event,
    ) -> None:
        runtime = self._require_runtime()
        await self.send_json({"type": "run.started", "run_id": run_id})

        try:
            async for chunk in runtime.agent.run(prompt, cancel_event=cancel_event):
                await self.send_json(self._serialize_chunk(run_id, chunk))
        except Exception as exc:
            await self.send_json(
                {
                    "type": "run.failed",
                    "run_id": run_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            await self.send_json(
                {
                    "type": "run.finished",
                    "run_id": run_id,
                    "total_tokens": runtime.agent.total_tokens,
                }
            )
            await self._send_snapshot()

    async def _send_snapshot(self) -> None:
        runtime = self._require_runtime()
        await self.send_json(
            {
                "type": "state.snapshot",
                "session": self._session_payload(runtime.agent.session),
                "model": self._model_payload(runtime.agent),
                "runtime": {
                    "is_processing": runtime.agent.is_processing,
                    "total_tokens": runtime.agent.total_tokens,
                },
            }
        )

    def _serialize_chunk(self, run_id: str, chunk: object) -> dict[str, object]:
        match chunk:
            case TextDeltaChunk(payload=text):
                return {
                    "type": "chunk.text",
                    "run_id": run_id,
                    "delta": text,
                }
            case ThinkingDeltaChunk(payload=thinking):
                return {
                    "type": "chunk.thinking",
                    "run_id": run_id,
                    "delta": thinking.text,
                }
            case ToolCallStartChunk(payload=tool_call):
                return {
                    "type": "chunk.tool_start",
                    "run_id": run_id,
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                }
            case ToolCallChunk(payload=tool_call):
                return {
                    "type": "chunk.tool_call",
                    "run_id": run_id,
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "input": dict(tool_call.arguments),
                }
            case ToolResultChunk(payload=tool_result):
                return {
                    "type": "chunk.tool_result",
                    "run_id": run_id,
                    "tool_call_id": tool_result.tool_call_id,
                    "tool_name": tool_result.name,
                    "content": tool_result.result,
                    "is_error": False,
                }
            case MessageChunk(payload=message):
                return {
                    "type": "chunk.message",
                    "run_id": run_id,
                    "role": message.role.value,
                    "content": message.content,
                }
            case _:
                raise TypeError(f"Unsupported chunk type: {type(chunk).__name__}")

    def _require_runtime(self) -> WebRuntime:
        if self._runtime is None:
            raise RuntimeError("web runtime not initialized")
        return self._runtime

    def _session_payload(self, session: Session) -> dict[str, object]:
        summary = summarize_session(session)
        return cast("dict[str, object]", summary.model_dump(mode="json"))

    def _model_payload(self, agent: object) -> dict[str, object]:
        runtime_agent = cast("Any", agent)
        return {
            "name": runtime_agent.model_name,
            "thinking_level": runtime_agent.thinking_level.value,
            "supports_thinking": runtime_agent.supports_thinking(),
        }


def create_app(
    *,
    config: Config | None = None,
    provider_factory: Callable[[Config], LLMProvider] | None = None,
) -> FastAPI:
    """Create the FastAPI app for the web delivery shell."""
    resolved_config = config or Config.load()
    resolved_provider_factory = provider_factory or _create_llm_provider
    app = FastAPI(title="My Own Coding Agent Web")
    app.state.agent_web = _AppState(
        config=resolved_config,
        provider_factory=resolved_provider_factory,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/sessions")
    async def get_sessions(limit: int = 20) -> list[dict[str, object]]:
        state = cast("_AppState", app.state.agent_web)
        return [
            cast("dict[str, object]", summary.model_dump(mode="json"))
            for summary in list_session_summaries(state.config.session_dir, limit=limit)
        ]

    @app.post("/api/sessions")
    async def create_session() -> dict[str, object]:
        state = cast("_AppState", app.state.agent_web)
        provider = state.provider_factory(state.config)
        try:
            session = Session.new(
                state.config.session_dir,
                provider=provider.name,
                model=provider.model,
            )
            return cast("dict[str, object]", summarize_session(session).model_dump(mode="json"))
        finally:
            await provider.close()

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, object]:
        state = cast("_AppState", app.state.agent_web)
        session = load_session_by_id(state.config.session_dir, session_id)
        if session is None:
            return {"error": f"Unknown session id: {session_id}"}
        session_payload = cast(
            "dict[str, object]",
            summarize_session(session).model_dump(mode="json"),
        )
        return {
            "session": session_payload,
            "entries": [
                cast("dict[str, object]", entry.model_dump(mode="json", by_alias=True))
                for entry in session.entries
            ],
            "active_entries": [
                cast("dict[str, object]", entry.model_dump(mode="json", by_alias=True))
                for entry in session.active_entries
            ],
        }

    @app.get("/api/models")
    async def get_models() -> dict[str, object]:
        state = cast("_AppState", app.state.agent_web)
        provider = state.provider_factory(state.config)
        try:
            models = await provider.list_models()
            return {
                "provider": provider.name,
                "model": provider.model,
                "supports_thinking": provider.supports_thinking(),
                "models": models,
            }
        finally:
            await provider.close()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        state = cast("_AppState", app.state.agent_web)
        connection = _WebClientConnection(
            websocket,
            config=state.config,
            provider_factory=state.provider_factory,
        )
        await connection.open()
        try:
            while True:
                payload = await websocket.receive_json()
                await connection.handle_message(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await connection.close()

    return app


def run_web_server(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the local web delivery server."""
    uvicorn.run(create_app(config=config), host=host, port=port)
