from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agent.extensions import ExtensionAPI, PresentedView, ViewControl, WidgetView

if TYPE_CHECKING:
    from pathlib import Path

PLAN_TOOLS = ["read", "grep", "find", "ls"]


@dataclass(slots=True)
class PlanRecord:
    summary: str
    steps: list[str]
    risks: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_request: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> PlanRecord:
        updated_at_raw = payload.get("updated_at")
        updated_at = (
            datetime.fromisoformat(updated_at_raw)
            if isinstance(updated_at_raw, str)
            else datetime.now(UTC)
        )
        return cls(
            summary=str(payload.get("summary") or "").strip(),
            steps=_coerce_list(payload.get("steps")),
            risks=_coerce_list(payload.get("risks")),
            validation=_coerce_list(payload.get("validation")),
            notes=_coerce_list(payload.get("notes")),
            source_request=str(payload.get("source_request") or "").strip(),
            updated_at=updated_at,
        )


@dataclass(slots=True)
class PlanState:
    session_id: str
    current_plan: PlanRecord | None = None
    active: bool = False
    pending_request: str | None = None
    previous_tools: list[str] | None = None
    previous_thinking_level: str | None = None


class PlanRegistry:
    def __init__(self) -> None:
        self._states: dict[str, PlanState] = {}

    def get(self, session_id: str, cwd: Path) -> PlanState:
        state = self._states.get(session_id)
        if state is not None:
            return state

        state = PlanState(session_id=session_id)
        path = _plan_path(cwd, session_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                payload = None
            if isinstance(payload, dict):
                state.current_plan = PlanRecord.from_json(payload)
        self._states[session_id] = state
        return state

    def clear(self, session_id: str) -> None:
        self._states.pop(session_id, None)


REGISTRY = PlanRegistry()


class PlanModeWidget(WidgetView):
    def __init__(self, state: PlanState) -> None:
        self._state = state

    def render(self) -> str:
        status = "updating" if self._state.pending_request else "ready"
        lines = [f"Plan mode [{status}]"]
        if self._state.current_plan is not None:
            lines.append(self._state.current_plan.summary)
            lines.append(f"{len(self._state.current_plan.steps)} steps")
        else:
            lines.append("No plan yet")
        if self._state.pending_request:
            lines.append(f"Pending: {self._state.pending_request[:80]}")
        return "\n".join(lines)


class PlanView(PresentedView[None]):
    def __init__(self, state: PlanState, plan_path: Path) -> None:
        self._state = state
        self._plan_path = plan_path
        self._closed = False

    def render(self) -> str:
        plan = self._state.current_plan
        if plan is None:
            return "No plan available.\n\nPress ESC to close"

        lines = [
            "Plan",
            "",
            f"Summary: {plan.summary}",
            f"Updated: {plan.updated_at.isoformat(timespec='seconds')}",
            f"File: {self._plan_path}",
            "",
            "Steps",
            *[f"{index}. {step}" for index, step in enumerate(plan.steps, start=1)],
        ]
        if plan.risks:
            lines.extend(["", "Risks", *[f"- {item}" for item in plan.risks]])
        if plan.validation:
            lines.extend(["", "Validation", *[f"- {item}" for item in plan.validation]])
        if plan.notes:
            lines.extend(["", "Notes", *[f"- {item}" for item in plan.notes]])
        if plan.source_request:
            lines.extend(["", "Source Request", plan.source_request])
        lines.extend(["", "Press ESC to close"])
        return "\n".join(lines)

    def controls(self) -> list[ViewControl]:
        return [ViewControl(kind="button", name="close", label="Close", primary=True)]

    def handle_action(self, action: str, value: str | None = None) -> None:
        if action == "close":
            self._closed = True

    def is_done(self) -> bool:
        return self._closed

    def result(self) -> None:
        return None


def _coerce_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _plan_path(cwd: Path, session_id: str) -> Path:
    return cwd / ".agent" / "plans" / f"{session_id}.json"


def _persist_plan(cwd: Path, state: PlanState) -> Path | None:
    if state.current_plan is None:
        return None
    path = _plan_path(cwd, state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.current_plan.to_json(), indent=2) + "\n")
    return path


def _remove_plan_file(cwd: Path, session_id: str) -> None:
    path = _plan_path(cwd, session_id)
    if path.exists():
        path.unlink()


def _render_widget(ctx, state: PlanState) -> None:
    if ctx.ui is None:
        return
    if state.active:
        ctx.ui.set_widget("footer", PlanModeWidget(state))
    else:
        ctx.ui.set_widget("footer", None)


def _format_plan(plan: PlanRecord) -> str:
    lines = ["Plan", "", f"Summary: {plan.summary}", "", "Steps"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(plan.steps, start=1))
    if plan.risks:
        lines.extend(["", "Risks", *[f"- {item}" for item in plan.risks]])
    if plan.validation:
        lines.extend(["", "Validation", *[f"- {item}" for item in plan.validation]])
    if plan.notes:
        lines.extend(["", "Notes", *[f"- {item}" for item in plan.notes]])
    return "\n".join(lines)


def _help_text(state: PlanState) -> str:
    status = "active" if state.active else "inactive"
    return (
        f"Plan mode is {status}.\n"
        "Commands:\n"
        "- /plan on\n"
        "- /plan <request>\n"
        "- /plan show\n"
        "- /plan apply\n"
        "- /plan off\n"
        "- /plan clear"
    )


def _build_planning_prompt(state: PlanState, request: str, cwd: Path) -> str:
    current_plan = (
        json.dumps(state.current_plan.to_json(), indent=2)
        if state.current_plan is not None
        else "(none yet)"
    )
    return (
        "Planning mode is active. Do not implement code changes. Use only planning and read-only "
        "analysis. If you need repo context, inspect with the available read-only tools.\n\n"
        f"Working directory: {cwd}\n\n"
        f"Current plan:\n{current_plan}\n\n"
        f"Planning request:\n{request}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "summary": "one-paragraph plan summary",\n'
        '  "steps": ["ordered implementation step"],\n'
        '  "risks": ["optional risk"],\n'
        '  "validation": ["how to verify the change"],\n'
        '  "notes": ["optional note"]\n'
        "}\n"
    )


def _extract_json_payload(text: str) -> dict[str, object] | None:
    candidates = [text.strip()]
    fenced = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _fallback_plan(request: str, text: str) -> PlanRecord:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0] if lines else request
    steps = []
    for line in lines:
        if re.match(r"^\\d+\\.\\s+", line):
            steps.append(re.sub(r"^\\d+\\.\\s+", "", line))
        elif line.startswith("- "):
            steps.append(line[2:].strip())
    if not steps:
        steps = [request]
    return PlanRecord(summary=summary[:240], steps=steps, source_request=request)


def _parse_plan(request: str, text: str) -> PlanRecord:
    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        return _fallback_plan(request, text)

    summary = str(payload.get("summary") or request).strip()[:240] or request
    steps = _coerce_list(payload.get("steps")) or [request]
    return PlanRecord(
        summary=summary,
        steps=steps,
        risks=_coerce_list(payload.get("risks")),
        validation=_coerce_list(payload.get("validation")),
        notes=_coerce_list(payload.get("notes")),
        source_request=request,
    )


def _activate_mode(ctx, state: PlanState) -> str:
    if state.active:
        return "Plan mode is already active"

    state.active = True
    state.previous_tools = ctx.tools.active()
    state.previous_thinking_level = ctx.model.get_thinking_level()
    ctx.tools.set_active(PLAN_TOOLS)
    ctx.model.set_thinking_level("high")
    if ctx.ui is not None:
        ctx.ui.set_status("plan mode")
        ctx.ui.notify("Plan mode enabled", "info")
    _render_widget(ctx, state)
    return "Plan mode enabled"


def _deactivate_mode(ctx, state: PlanState) -> str:
    if not state.active:
        return "Plan mode is not active"

    if state.previous_tools is not None:
        ctx.tools.set_active(state.previous_tools)
    if state.previous_thinking_level is not None:
        ctx.model.set_thinking_level(state.previous_thinking_level)

    state.active = False
    state.pending_request = None
    state.previous_tools = None
    state.previous_thinking_level = None
    if ctx.ui is not None:
        ctx.ui.set_status(None)
        ctx.ui.notify("Plan mode disabled", "info")
    _render_widget(ctx, state)
    return "Plan mode disabled"


async def _queue_plan_request(ctx, state: PlanState, request: str) -> str:
    if not state.active:
        _activate_mode(ctx, state)

    state.pending_request = request.strip()
    _render_widget(ctx, state)
    if ctx.ui is not None:
        ctx.ui.notify("Queued planning request", "info")
    await ctx.runtime.send_user_message(_build_planning_prompt(state, request, ctx.cwd))
    return f"Queued planning request: {request}"


async def _show_plan(ctx, state: PlanState) -> str:
    plan = state.current_plan
    if plan is None:
        return "No plan available"

    path = _plan_path(ctx.cwd, state.session_id)
    if ctx.ui is not None:
        await ctx.ui.present(PlanView(state, path))
    return _format_plan(plan)


async def _apply_plan(ctx, state: PlanState) -> str:
    plan = state.current_plan
    if plan is None:
        return "No plan available"

    if ctx.ui is not None:
        confirmed = await ctx.ui.confirm("Apply the current plan to the main thread?")
        if not confirmed:
            return "Cancelled"

    if state.active:
        _deactivate_mode(ctx, state)

    await ctx.runtime.send_user_message(
        "Execute this approved plan.\n\n"
        f"{_format_plan(plan)}\n\n"
        "Follow the plan, adapt it to the codebase as needed, and report "
        "the implementation results."
    )
    return "Queued current plan into the main thread"


def _clear_plan(ctx, state: PlanState) -> str:
    state.current_plan = None
    state.pending_request = None
    _remove_plan_file(ctx.cwd, state.session_id)
    if not state.active:
        _render_widget(ctx, state)
    return "Cleared current plan"


async def _plan_command(args: str, ctx) -> str:
    state = REGISTRY.get(ctx.session.id, ctx.cwd)
    stripped = args.strip()
    if not stripped:
        if state.current_plan is not None:
            return await _show_plan(ctx, state)
        return _help_text(state)

    command, _, remainder = stripped.partition(" ")
    lowered = command.lower()

    if lowered == "on":
        return _activate_mode(ctx, state)
    if lowered == "off":
        return _deactivate_mode(ctx, state)
    if lowered == "show":
        return await _show_plan(ctx, state)
    if lowered == "apply":
        return await _apply_plan(ctx, state)
    if lowered == "clear":
        return _clear_plan(ctx, state)

    return await _queue_plan_request(ctx, state, stripped)


def _on_turn_end(event, ctx) -> None:
    state = REGISTRY.get(ctx.session.id, ctx.cwd)
    if not state.pending_request:
        return
    if event.message is None or event.message.role.value != "assistant":
        return

    request = state.pending_request
    state.current_plan = _parse_plan(request, event.message.content)
    state.pending_request = None
    _persist_plan(ctx.cwd, state)

    if ctx.ui is not None:
        ctx.ui.notify("Plan updated", "info")
    _render_widget(ctx, state)


def _on_session_end(event, ctx) -> None:
    state = REGISTRY.get(ctx.session.id, ctx.cwd)
    state.pending_request = None
    if ctx.ui is not None:
        ctx.ui.set_status(None)
        ctx.ui.set_widget("footer", None)


def setup(api: ExtensionAPI) -> None:
    api.register_command("plan", _plan_command)
    api.on("turn_end", _on_turn_end)
    api.on("session_end", _on_session_end)
