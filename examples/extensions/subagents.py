from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.extensions import ExtensionAPI, PresentedView, ViewControl, WidgetView
from agent.llm import factory as llm_factory
from agent.runtime.agent import Agent
from agent.runtime.message import Role
from agent.runtime.settings import AgentSettings, ThinkingLevel

if TYPE_CHECKING:
    from agent.config.runtime import ProviderConfig


@dataclass(slots=True, frozen=True)
class SubagentProfile:
    name: str
    description: str
    instructions: str
    thinking_level: ThinkingLevel
    active_tools: tuple[str, ...]
    write_enabled: bool = False
    model: str | None = None


@dataclass(slots=True)
class SubagentRun:
    id: str
    session_id: str
    profile: SubagentProfile
    task: str
    status: str = "queued"
    summary: str = ""
    details: str = ""
    findings: list[str] = field(default_factory=list)
    recommended_next_step: str | None = None
    error: str | None = None
    files_changed: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    raw_output: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task_handle: asyncio.Task[None] | None = None


PROFILES: dict[str, SubagentProfile] = {
    "researcher": SubagentProfile(
        name="researcher",
        description="Read-only context gathering and option synthesis.",
        instructions=(
            "You are a focused research subagent. Gather the most relevant context for the task, "
            "identify constraints and options, and return a concise but useful summary."
        ),
        thinking_level=ThinkingLevel.MEDIUM,
        active_tools=("read", "grep", "find", "ls"),
    ),
    "reviewer": SubagentProfile(
        name="reviewer",
        description="Read-only bug, regression, and testing review.",
        instructions=(
            "You are a review subagent. Prioritize bugs, regressions, missing tests, and risky "
            "assumptions. Order findings by severity and use concrete file "
            "references when possible."
        ),
        thinking_level=ThinkingLevel.HIGH,
        active_tools=("read", "grep", "find", "ls"),
    ),
    "implementer": SubagentProfile(
        name="implementer",
        description="Write-enabled scoped implementation pass.",
        instructions=(
            "You are an implementation subagent. Make the smallest effective change for the task, "
            "verify it when practical, and report exactly what changed."
        ),
        thinking_level=ThinkingLevel.MEDIUM,
        active_tools=("read", "grep", "find", "ls", "edit", "write", "bash"),
        write_enabled=True,
    ),
}


class SubagentRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, list[SubagentRun]] = {}

    def add(self, run: SubagentRun) -> None:
        self._runs.setdefault(run.session_id, []).append(run)

    def list(self, session_id: str) -> list[SubagentRun]:
        return list(self._runs.get(session_id, []))

    def get(self, session_id: str, run_id: str) -> SubagentRun | None:
        for run in self._runs.get(session_id, []):
            if run.id == run_id:
                return run
        return None

    def cancel_session(self, session_id: str) -> None:
        for run in self._runs.get(session_id, []):
            _cancel_run(run)


REGISTRY = SubagentRegistry()


def _visible_widget_runs(session_id: str) -> list[SubagentRun]:
    return [run for run in REGISTRY.list(session_id) if run.status != "applied"]


def _sync_subagent_widget(ctx) -> None:
    if ctx.ui is None:
        return
    if _visible_widget_runs(ctx.session.id):
        ctx.ui.set_widget("right_panel", SubagentWidget(ctx.session.id))
    else:
        ctx.ui.set_widget("right_panel", None)


class SubagentWidget(WidgetView):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def render(self) -> str:
        runs = _visible_widget_runs(self._session_id)
        recent = runs[-6:]
        lines = ["Subagents"]
        for run in reversed(recent):
            status = run.status
            summary = run.summary or run.task
            lines.append(f"- {run.id} [{status}] {run.profile.name}: {summary}")
        return "\n".join(lines)


class SubagentResultView(PresentedView[None]):
    def __init__(self, run: SubagentRun) -> None:
        self._run = run
        self._closed = False

    def render(self) -> str:
        parts = [
            f"Subagent {self._run.id}",
            f"Profile: {self._run.profile.name}",
            f"Status: {self._run.status}",
            f"Task: {self._run.task}",
            "",
        ]
        if self._run.summary:
            parts.extend(["Summary", self._run.summary, ""])
        if self._run.findings:
            parts.extend(["Findings", *[f"- {item}" for item in self._run.findings], ""])
        if self._run.files_changed:
            parts.extend(["Files Changed", *[f"- {path}" for path in self._run.files_changed], ""])
        if self._run.commands_run:
            parts.extend(["Commands Run", *[f"- {cmd}" for cmd in self._run.commands_run], ""])
        if self._run.recommended_next_step:
            parts.extend(["Recommended Next Step", self._run.recommended_next_step, ""])
        if self._run.error:
            parts.extend(["Error", self._run.error, ""])
        if self._run.details:
            parts.extend(["Details", self._run.details])
        parts.append("")
        parts.append("Press ESC to close")
        return "\n".join(parts).strip()

    def controls(self) -> list[ViewControl]:
        return [ViewControl(kind="button", name="close", label="Close", primary=True)]

    def handle_action(self, action: str, value: str | None = None) -> None:
        if action == "close":
            self._closed = True

    def is_done(self) -> bool:
        return self._closed

    def result(self) -> None:
        return None


def _provider_overrides(raw: object) -> dict[str, ProviderConfig]:
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, ProviderConfig] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            continue
        base_url = values.get("base_url")
        if not isinstance(base_url, str):
            continue
        overrides[name] = SimpleNamespace(
            base_url=base_url,
            model=values.get("model"),
            api_key=values.get("api_key"),
        )
    return overrides


def _build_child_settings(ctx, profile: SubagentProfile) -> AgentSettings:
    return AgentSettings(
        context_max_tokens=ctx.config["context_max_tokens"],
        max_output_tokens=ctx.config["max_output_tokens"],
        temperature=ctx.config["temperature"],
        thinking_level=profile.thinking_level,
        session_dir=Path(ctx.config["session_dir"]),
        skills_dirs=list(ctx.config["skills_dirs"]),
        extensions=[],
        prompt_template_dirs=list(ctx.config["prompt_template_dirs"]),
        context_file_paths=list(ctx.config["context_file_paths"]),
        custom_system_prompt=ctx.config["custom_system_prompt"],
        append_system_prompt=ctx.config["append_system_prompt"],
    )


def _build_prompt(profile: SubagentProfile, task: str, cwd: Path) -> str:
    return (
        f"{profile.instructions}\n\n"
        "Work only on the delegated scope below.\n"
        f"Working directory: {cwd}\n\n"
        f"Task:\n{task}\n\n"
        "Return a JSON object only, with this exact shape:\n"
        "{\n"
        '  "summary": "short summary",\n'
        '  "details": "full details",\n'
        '  "findings": ["optional point"],\n'
        '  "recommended_next_step": "optional next step"\n'
        "}\n"
    )


def _parse_structured_result(text: str, fallback: str) -> tuple[str, str, list[str], str | None]:
    stripped = text.strip()
    payload = _extract_json_payload(stripped)
    if not isinstance(payload, dict):
        return _fallback_result(stripped, fallback)

    summary = str(payload.get("summary") or fallback).strip()[:240] or fallback
    details = str(payload.get("details") or "").strip()
    findings_raw = payload.get("findings")
    findings = []
    if isinstance(findings_raw, list):
        findings = [str(item).strip() for item in findings_raw if str(item).strip()]
    next_step = payload.get("recommended_next_step")
    recommended_next_step = str(next_step).strip() if next_step else None
    return summary, details, findings, recommended_next_step


def _extract_json_payload(text: str) -> dict[str, object] | None:
    candidates = [text]
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    tag_match = re.search(r"<subagent-result>\s*(\{.*?\})\s*</subagent-result>", text, re.DOTALL)
    if tag_match:
        candidates.insert(0, tag_match.group(1).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _fallback_result(text: str, fallback: str) -> tuple[str, str, list[str], str | None]:
    if not text:
        return fallback, "", [], None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[0][:240] if lines else fallback
    details = text
    findings = [line[2:].strip() for line in lines if line.startswith("- ")]
    return summary or fallback, details, findings, None


def _record_tool_call(run: SubagentRun, name: str, arguments: dict[str, object]) -> None:
    if name == "bash":
        command = arguments.get("command")
        if isinstance(command, str) and command and command not in run.commands_run:
            run.commands_run.append(command)
        return

    if name in {"edit", "write"}:
        path = arguments.get("path")
        if isinstance(path, str) and path and path not in run.files_changed:
            run.files_changed.append(path)


def _cancel_run(run: SubagentRun) -> None:
    if run.status in {"completed", "failed", "cancelled"}:
        return
    run.cancel_event.set()
    run.status = "cancelled"
    run.finished_at = datetime.now(UTC)
    if run.task_handle is not None:
        run.task_handle.cancel()


def _format_run(run: SubagentRun) -> str:
    summary = run.summary or run.task
    return f"{run.id} [{run.status}] {run.profile.name}: {summary}"


def _format_run_details(run: SubagentRun) -> str:
    parts = [
        f"Subagent {run.id}",
        f"Profile: {run.profile.name}",
        f"Status: {run.status}",
        f"Task: {run.task}",
    ]
    if run.summary:
        parts.extend(["", "Summary", run.summary])
    if run.findings:
        parts.extend(["", "Findings", *[f"- {item}" for item in run.findings]])
    if run.files_changed:
        parts.extend(["", "Files Changed", *[f"- {path}" for path in run.files_changed]])
    if run.commands_run:
        parts.extend(["", "Commands Run", *[f"- {cmd}" for cmd in run.commands_run]])
    if run.recommended_next_step:
        parts.extend(["", "Recommended Next Step", run.recommended_next_step])
    if run.error:
        parts.extend(["", "Error", run.error])
    if run.details:
        parts.extend(["", "Details", run.details])
    return "\n".join(parts)


async def _execute_subagent(run: SubagentRun, ctx, ui) -> None:
    profile = run.profile
    run.status = "running"
    run.started_at = datetime.now(UTC)
    _sync_subagent_widget(ctx)

    provider = llm_factory.create_provider(
        provider=ctx.config["provider"],
        model=profile.model or ctx.model.get(),
        api_key=ctx.config.get("api_key"),
        base_url=ctx.config.get("base_url"),
        temperature=ctx.config["temperature"],
        max_output_tokens=ctx.config["max_output_tokens"],
        provider_overrides=_provider_overrides(ctx.config.get("provider_overrides")),
    )
    child = Agent(
        _build_child_settings(ctx, profile),
        provider,
        cwd=ctx.cwd,
    )
    child.tools.set_active_tools(list(profile.active_tools))
    child.refresh_system_prompt()

    try:
        async for chunk in child.run(
            _build_prompt(profile, run.task, ctx.cwd), cancel_event=run.cancel_event
        ):
            if chunk.type == "tool_call":
                _record_tool_call(run, chunk.payload.name, chunk.payload.arguments)

        assistant_messages = [
            message
            for message in child.session.messages
            if message.role == Role.ASSISTANT and message.content
        ]
        if assistant_messages:
            run.raw_output = assistant_messages[-1].content.strip()
        run.summary, run.details, run.findings, run.recommended_next_step = (
            _parse_structured_result(
                run.raw_output,
                fallback=run.task,
            )
        )
        if run.status != "cancelled":
            run.status = "completed"
            _sync_subagent_widget(ctx)
            if ui is not None:
                ui.notify(f"Subagent {run.id} finished", "info")
    except asyncio.CancelledError:
        run.status = "cancelled"
        run.error = None
        _sync_subagent_widget(ctx)
        raise
    except Exception as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        _sync_subagent_widget(ctx)
        if ui is not None:
            ui.notify(f"Subagent {run.id} failed: {exc}", "error")
    finally:
        run.finished_at = datetime.now(UTC)
        _sync_subagent_widget(ctx)
        with contextlib.suppress(Exception):
            await child.close()


async def _launch_subagent(ctx, profile: SubagentProfile, task: str) -> SubagentRun:
    run = SubagentRun(
        id=uuid4().hex[:8],
        session_id=ctx.session.id,
        profile=profile,
        task=task,
    )
    REGISTRY.add(run)

    if ctx.ui is not None:
        _sync_subagent_widget(ctx)
        ctx.ui.notify(f"Launching {profile.name} subagent {run.id}", "info")

    run.task_handle = asyncio.create_task(_execute_subagent(run, ctx, ctx.ui))
    return run


async def _pick_run(ctx, purpose: str) -> SubagentRun | None:
    runs = REGISTRY.list(ctx.session.id)
    if not runs:
        return None
    if ctx.ui is None:
        return runs[-1]
    options = [f"{run.id} {run.profile.name} [{run.status}]" for run in runs]
    selected = await ctx.ui.select(f"Select subagent to {purpose}", options)
    if not selected:
        return None
    run_id = selected.split()[0]
    return REGISTRY.get(ctx.session.id, run_id)


def _parse_profile_and_task(args: str) -> tuple[str | None, str]:
    stripped = args.strip()
    if not stripped:
        return None, ""
    parts = stripped.split(maxsplit=1)
    profile = parts[0].strip().lower()
    task = parts[1].strip() if len(parts) > 1 else ""
    return profile, task


async def _interactive_launch_request(ctx) -> tuple[str, str] | None:
    if ctx.ui is None:
        return None

    options = [f"{profile.name} - {profile.description}" for profile in PROFILES.values()]
    selected = await ctx.ui.select("Choose subagent profile", options)
    if not selected:
        return None

    profile_name = selected.split(" - ", 1)[0].strip().lower()
    profile = PROFILES[profile_name]
    task = await ctx.ui.input(f"Task for {profile.name}")
    if task is None:
        return None
    task = task.strip()
    if not task:
        return None

    if profile.write_enabled:
        confirmed = await ctx.ui.confirm(
            f"{profile.name} can modify files with: {', '.join(profile.active_tools)}. Continue?"
        )
        if not confirmed:
            return None

    return profile_name, task


async def _subagent_command(args: str, ctx) -> str:
    profile_name, task = _parse_profile_and_task(args)

    if not profile_name:
        if ctx.ui is None:
            return "Usage: /subagent <researcher|reviewer|implementer> <task>"
        launch_request = await _interactive_launch_request(ctx)
        if launch_request is None:
            return "Cancelled"
        profile_name, task = launch_request

    profile = PROFILES.get(profile_name)
    if profile is None:
        available = ", ".join(sorted(PROFILES))
        return f"Unknown subagent profile: {profile_name}. Available: {available}"

    if not task:
        return f"Usage: /subagent {profile.name} <task>"

    if profile.write_enabled and ctx.ui is not None:
        confirmed = await ctx.ui.confirm(
            f"{profile.name} can modify files with: {', '.join(profile.active_tools)}. Continue?"
        )
        if not confirmed:
            return "Cancelled"

    run = await _launch_subagent(ctx, profile, task)
    return f"Launched subagent {run.id} [{profile.name}] {task}"


def _subagents_command(args: str, ctx) -> str:
    runs = REGISTRY.list(ctx.session.id)
    if not runs:
        return "No subagents"
    return "\n".join(_format_run(run) for run in reversed(runs))


async def _subagent_show_command(args: str, ctx) -> str:
    run_id = args.strip()
    run = REGISTRY.get(ctx.session.id, run_id) if run_id else await _pick_run(ctx, "inspect")
    if run is None:
        return "Subagent not found"
    if ctx.ui is not None:
        await ctx.ui.present(SubagentResultView(run))
    return _format_run_details(run)


async def _subagent_apply_command(args: str, ctx) -> str:
    parts = args.split(maxsplit=1)
    run_id = parts[0].strip() if parts else ""
    mode = parts[1].strip().lower() if len(parts) > 1 else ""
    run = REGISTRY.get(ctx.session.id, run_id) if run_id else await _pick_run(ctx, "apply")
    if run is None:
        return "Subagent not found"
    if run.status != "completed":
        return f"Subagent {run.id} is {run.status}; only completed runs can be applied"

    if mode not in {"summary", "full"}:
        if ctx.ui is not None:
            selected = await ctx.ui.select("Apply summary or full result?", ["summary", "full"])
            if not selected:
                return "Cancelled"
            mode = selected
        else:
            mode = "summary"

    body = run.summary if mode == "summary" else _format_run_details(run)
    await ctx.runtime.send_user_message(
        f"[Subagent {run.id} | {run.profile.name} | {mode}]\nTask: {run.task}\n\n{body}"
    )
    run.status = "applied"
    _sync_subagent_widget(ctx)
    return f"Queued {mode} result from subagent {run.id} into the main thread"


async def _subagent_cancel_command(args: str, ctx) -> str:
    run_id = args.strip()
    run = REGISTRY.get(ctx.session.id, run_id) if run_id else await _pick_run(ctx, "cancel")
    if run is None:
        return "Subagent not found"
    if run.status in {"completed", "failed", "cancelled"}:
        return f"Subagent {run.id} is already {run.status}"
    _cancel_run(run)
    _sync_subagent_widget(ctx)
    return f"Cancelled subagent {run.id}"


def _on_session_end(event, ctx) -> None:
    REGISTRY.cancel_session(ctx.session.id)
    if ctx.ui is not None:
        ctx.ui.set_widget("right_panel", None)


def setup(api: ExtensionAPI) -> None:
    api.register_command("subagent", _subagent_command)
    api.register_command("subagents", _subagents_command)
    api.register_command("subagent-show", _subagent_show_command)
    api.register_command("subagent-apply", _subagent_apply_command)
    api.register_command("subagent-cancel", _subagent_cancel_command)
    api.on("session_end", _on_session_end)
