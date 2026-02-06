"""Main Agent class - orchestrates LLM, tools, and session."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agent.core.config import (
    Config,
    ThinkingLevel,
    clamp_thinking_level,
    get_available_thinking_levels,
)
from agent.core.context import ContextManager
from agent.core.context_loader import load_all_context
from agent.core.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ContextCompactionEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSelectEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agent.core.message import (
    Message,
    Role,
    ThinkingContent,
    ToolCall,
    ToolCallStart,
    ToolResult,
)
from agent.core.prompt_builder import ContextFile, SystemPromptOptions, build_system_prompt
from agent.core.session import Session
from agent.extensions.api import ExtensionAPI
from agent.extensions.loader import ExtensionLoader
from agent.extensions.runner import ExtensionRunner
from agent.extensions.types import ToolCallEvent, ToolResultEvent
from agent.llm.events import StreamOptions, ToolCallBlock
from agent.prompts.loader import PromptTemplateLoader
from agent.prompts.parser import ParsedCommand, expand_template, parse_command
from agent.skills.loader import SkillLoader
from agent.tools.bash import BashTool
from agent.tools.edit import EditTool
from agent.tools.find import FindTool
from agent.tools.grep import GrepTool
from agent.tools.ls import LsTool
from agent.tools.read import ReadTool
from agent.tools.registry import ToolRegistry
from agent.tools.write import WriteTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from agent.llm.provider import LLMProvider
    from agent.tools.base import BaseTool


class Agent:
    """Main agent orchestrating LLM, tools, and session."""

    __slots__ = (
        "config",
        "provider",
        "tools",
        "session",
        "context",
        "skill_loader",
        "template_loader",
        "_total_tokens",
        "_listeners",
        "_extension_api",
        "_extension_runner",
        "_in_loop",
        "_cwd",
        "_context_files",
    )

    def __init__(
        self,
        config: Config,
        session: Session | None = None,
        cwd: Path | None = None,
        skill_loader: SkillLoader | None = None,
        template_loader: PromptTemplateLoader | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration
            session: Optional existing session to resume
            cwd: Working directory (defaults to Path.cwd())
            skill_loader: Optional skill loader (created if not provided)
            template_loader: Optional template loader (created if not provided)
        """
        self.config = config
        self._cwd = cwd or Path.cwd()
        self.provider = self._create_provider(config)
        self.tools = ToolRegistry()
        self.session = session or Session.new(
            config.session_dir, provider=config.provider, model=config.model
        )
        self.context = ContextManager(self.provider, config.context_max_tokens)
        self._total_tokens = 0
        self._listeners: set[Callable[[AgentEvent], Any]] = set()
        self._extension_api = ExtensionAPI()
        self._extension_runner = ExtensionRunner(self._extension_api, self)
        self._in_loop = False
        self._context_files: list[ContextFile] = []

        # Use provided loaders or create with default directories + config paths
        self.skill_loader = skill_loader or SkillLoader.with_defaults(
            extra_dirs=config.skills_dirs,
            cwd=self._cwd,
        )

        # Use provided loader or create with default directories + config paths
        self.template_loader = template_loader or PromptTemplateLoader.with_defaults(
            extra_dirs=config.prompt_template_dirs,
            cwd=self._cwd,
        )

        # Register built-in tools
        tools: list[BaseTool[Any]] = [
            ReadTool(),
            WriteTool(),
            EditTool(),
            BashTool(),
            GrepTool(),
            FindTool(),
            LsTool(),
        ]
        for tool in tools:
            self.tools.register(tool)

        # Restore model selection from session (if any)
        self._restore_model_from_session()

        # Add system prompt if this is a new session
        if not self.session.messages:
            self._init_system_prompt()

    def _create_provider(self, config: Config) -> LLMProvider:
        """Create LLM provider from config."""
        prov_config = config.get_provider_config()

        # Use Anthropic provider for anthropic provider type
        if config.provider == "anthropic":
            from agent.llm.anthropic import AnthropicProvider

            return AnthropicProvider(
                api_key=prov_config.api_key or "",
                model=prov_config.model,
                max_tokens=config.max_output_tokens,
            )

        # Use native OpenAI provider for openai provider type
        if config.provider == "openai":
            from agent.llm.openai import OpenAIProvider

            return OpenAIProvider(
                api_key=prov_config.api_key or "",
                model=prov_config.model,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
            )

        # Default to OpenAI-compatible for others (ollama, openrouter, groq, etc.)
        from agent.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url=prov_config.base_url,
            api_key=prov_config.api_key or "",
            model=prov_config.model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    def _init_system_prompt(self) -> None:
        """Initialize system prompt with context using the prompt builder."""
        # Load context files (AGENTS.md, CLAUDE.md from project and ancestors)
        context_files = load_all_context(
            cwd=self._cwd,
            explicit_paths=self.config.context_file_paths,
            include_ancestors=True,
        )
        # Store for later access via /context command
        self._context_files = context_files

        # Get invocable skills (those not marked as disable_model_invocation)
        skills = self.skill_loader.get_invocable_skills()

        # Get list of registered tool names
        tool_names = self.tools.list_tools()

        # Build system prompt options
        options = SystemPromptOptions(
            custom_prompt=self.config.custom_system_prompt,
            selected_tools=tool_names,
            append_system_prompt=self.config.append_system_prompt,
            cwd=self._cwd,
            context_files=context_files,
            skills=skills,
        )

        # Build the system prompt
        system_content = build_system_prompt(options)

        self.session.append(Message(role=Role.SYSTEM, content=system_content))

    def new_session(self) -> None:
        """Start a fresh session and reinitialize system prompt."""
        self.session = Session.new(
            self.config.session_dir, provider=self.config.provider, model=self.config.model
        )
        self._total_tokens = 0
        self._init_system_prompt()

    def load_session(self, session: Session) -> None:
        """Switch to an existing session and restore model selection."""
        self.session = session
        self._restore_model_from_session()
        if not self.session.messages:
            self._init_system_prompt()
        self._total_tokens = self.context.current_tokens(self.session.messages)

    def set_model(self, model: str, source: str = "set") -> None:
        """Switch to a new model and persist selection."""
        if not model or model == self.provider.model:
            return

        previous_model = self.provider.model
        self.provider.model = model
        self.config.model = model

        # Reset encoder for providers that cache per-model encoders
        if hasattr(self.provider, "_encoder"):
            cast("Any", self.provider)._encoder = None

        # Clamp thinking level to model capabilities
        available = get_available_thinking_levels(model)
        self.config.thinking_level = clamp_thinking_level(self.config.thinking_level, available)

        # Persist selection in session entries
        self.session.append_model_change(self.config.provider, model)

        # Emit model selection event
        self._emit_model_select(
            provider=self.config.provider,
            model=model,
            previous_provider=self.config.provider,
            previous_model=previous_model,
            source=source,
        )

    @property
    def total_tokens(self) -> int:
        """Get total tokens used in current context."""
        return self._total_tokens

    @property
    def context_files(self) -> list[ContextFile]:
        """Get loaded context files (AGENTS.md, CLAUDE.md, etc.)."""
        return self._context_files

    @property
    def extension_api(self) -> ExtensionAPI:
        """Get the extension API for registering handlers."""
        return self._extension_api

    def subscribe(self, listener: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        """Subscribe to agent events.

        Args:
            listener: Function to call with each event

        Returns:
            Unsubscribe function
        """
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    async def _emit(self, event: AgentEvent) -> None:
        """Emit an event to all listeners and extensions."""
        # Notify listeners
        for listener in self._listeners:
            try:
                result = listener(event)
                # Handle async listeners
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass  # Don't let listener errors break the agent

        # Also emit to extensions
        await self._extension_runner.emit_agent_event(event)

    def _emit_model_select(
        self,
        provider: str,
        model: str,
        previous_provider: str | None,
        previous_model: str | None,
        source: str,
    ) -> None:
        """Emit a model_select event if possible."""
        if previous_model == model and previous_provider == provider:
            return

        event = ModelSelectEvent(
            provider=provider,
            model=model,
            previous_provider=previous_provider,
            previous_model=previous_model,
            source=source,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(self._emit(event))

    def _restore_model_from_session(self) -> None:
        """Restore model selection from session entries (if available)."""
        selection = self.session.get_model_selection()
        if not selection:
            return

        provider, model = selection
        if not model:
            return
        if provider and provider != self.config.provider:
            return

        previous_model = self.config.model

        self.config.model = model
        self.provider.model = model

        # Reset encoder for providers that cache per-model encoders
        if hasattr(self.provider, "_encoder"):
            cast("Any", self.provider)._encoder = None

        # Clamp thinking level to model capabilities
        available = get_available_thinking_levels(model)
        self.config.thinking_level = clamp_thinking_level(self.config.thinking_level, available)

        # Emit model selection event (restore)
        self._emit_model_select(
            provider=provider or self.config.provider,
            model=model,
            previous_provider=self.config.provider,
            previous_model=previous_model,
            source="restore",
        )

    async def load_extensions(self, paths: list[Path] | None = None) -> list[str]:
        """Load extensions from config or specified paths.

        Args:
            paths: Optional list of paths to load (defaults to config.extensions)

        Returns:
            List of error messages (empty if all succeeded)
        """
        extension_paths = paths or self.config.extensions
        if not extension_paths:
            return []

        errors = await ExtensionLoader.load_multiple(extension_paths, self._extension_api)

        # Register extension tools
        for tool in self._extension_api.get_tools().values():
            self.tools.register(tool)

        return errors

    async def run(
        self, user_input: str, cancel_event: asyncio.Event | None = None
    ) -> AsyncIterator[str | ToolCall | ToolCallStart | ToolResult | ThinkingContent | Message]:
        """Process user input and yield response chunks, tool calls, or tool results.

        Args:
            user_input: The user's message
            cancel_event: Optional event to signal cancellation

        Yields:
            String tokens, ToolCall/ToolCallStart/ToolCallDelta objects,
            ToolResult objects, or ThinkingContent
        """
        # Check for input blocking/transformation via extensions
        input_result = await self._extension_runner.emit_input(user_input)
        if input_result and input_result.block:
            return  # Input was blocked by extension

        # Use transformed input if provided
        if input_result and input_result.text:
            user_input = input_result.text

        # Expand prompt templates (e.g., /template-name arg1 arg2)
        user_input, command, used_template = self._expand_template(user_input)

        # Add user message
        self.session.append(Message(role=Role.USER, content=user_input))

        # Check compaction
        if self.context.needs_compaction(self.session.messages):
            original_tokens = self.context.current_tokens(self.session.messages)
            compaction = await self.context.compact(self.session.messages)
            self.session.append_compaction(
                compaction.summary,
                compaction.first_kept_id,
                tokens_before=original_tokens,
            )
            compacted_tokens = self.context.current_tokens(self.session.messages)
            await self._emit(
                ContextCompactionEvent(
                    original_tokens=original_tokens, compacted_tokens=compacted_tokens
                )
            )

        # Emit agent start
        await self._emit(AgentStartEvent())

        # Handle extension slash command (only if no template matched)
        if command and not used_template:
            cmd_result = await self._extension_runner.execute_command(
                command.template_name, command.raw_args
            )
            if cmd_result is not None:
                output = cmd_result.strip()
                if not output:
                    output = "(no output)"
                command_msg = Message.system(f"[Command /{command.template_name}]\n{output}")
                self.session.append(command_msg)
                await self._emit(MessageEndEvent(message=command_msg))
                yield command_msg
                await self._emit(AgentEndEvent(messages=list(self.session.messages)))
                return

        # Run agent loop (may have multiple turns for tool use)
        async for chunk in self._agent_loop(cancel_event=cancel_event):
            yield chunk

        # Emit agent end
        await self._emit(AgentEndEvent(messages=list(self.session.messages)))

    def _expand_template(self, user_input: str) -> tuple[str, ParsedCommand | None, bool]:
        """Expand prompt template if input is a slash command.

        Format: /template-name arg1 arg2 "arg with spaces" ...

        Args:
            user_input: User input text

        Returns:
            Tuple of (expanded content, parsed command or None, template_used)
        """
        # Expand skill commands: $skill-name [args]
        stripped = user_input.strip()
        if stripped.startswith("$"):
            skill_text = stripped[1:]
            if not skill_text:
                return user_input, None, False

            parts = skill_text.split(None, 1)
            skill_name = parts[0]
            raw_args = parts[1] if len(parts) > 1 else ""

            skill = self.skill_loader.get(skill_name)
            if not skill:
                return user_input, None, False

            try:
                body = skill.read_body()
            except Exception:
                return user_input, None, False

            skill_block = (
                f'<skill name="{skill.name}" location="{skill.readme_path}">\n'
                f"References are relative to {skill.base_dir}.\n\n"
                f"{body}\n"
                "</skill>"
            )
            args = raw_args.strip()
            expanded = f"{skill_block}\n\n{args}" if args else skill_block
            return expanded, None, True

        command = parse_command(user_input)
        if not command:
            return user_input, None, False

        # Look up template by name
        template = self.template_loader.get(command.template_name)
        if not template:
            # Prefer extension commands over skills
            if command.template_name in self._extension_api.get_commands():
                return user_input, command, False

            # Not a known template or skill, return original input
            return user_input, command, False

        # Expand the template with arguments
        return expand_template(template.content, command), command, True

    def _build_stream_options(self, cancel_event: asyncio.Event | None = None) -> StreamOptions:
        """Build StreamOptions from config."""
        thinking_level: str | None = None
        if self.config.thinking_level and self.config.thinking_level != ThinkingLevel.OFF:
            thinking_level = self.config.thinking_level.value

        return StreamOptions(
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            thinking_level=thinking_level,
            cancel_event=cancel_event,
        )

    async def _agent_loop(
        self,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[str | ToolCall | ToolCallStart | ToolResult | ThinkingContent | Message]:
        """Run the agent loop with tool execution."""
        max_iterations = 25  # Prevent infinite loops
        self._in_loop = True

        try:
            for turn in range(max_iterations):
                # Check cancellation at start of each turn
                if cancel_event and cancel_event.is_set():
                    break

                await self._emit(TurnStartEvent(turn_number=turn))

                response_content = ""
                thinking_content = ""
                provider_metadata: dict[str, Any] = {}
                tool_calls: list[ToolCall] = []
                thinking_started = False
                message_started = False
                stream_error: str | None = None
                stream_aborted = False

                # Allow extensions to modify context before LLM call
                messages_for_llm = await self._extension_runner.emit_context(
                    list(self.session.messages)
                )

                # Build stream options with cancel_event
                options = self._build_stream_options(cancel_event=cancel_event)

                # Get completion stream
                stream = self.provider.stream(
                    messages_for_llm,
                    tools=self.tools.get_schemas(),
                    options=options,
                )

                # Consume events from stream
                async for event in stream:
                    match event.type:
                        case "text_start":
                            if not message_started:
                                await self._emit(MessageStartEvent())
                                message_started = True

                        case "text_delta":
                            if not message_started:
                                await self._emit(MessageStartEvent())
                                message_started = True
                            response_content += event.delta
                            await self._emit(MessageUpdateEvent(delta=event.delta))
                            yield event.delta

                        case "thinking_start":
                            if not thinking_started:
                                await self._emit(ThinkingStartEvent())
                                thinking_started = True

                        case "thinking_delta":
                            if not thinking_started:
                                await self._emit(ThinkingStartEvent())
                                thinking_started = True
                            thinking_content += event.delta
                            await self._emit(ThinkingDeltaEvent(delta=event.delta))
                            yield ThinkingContent(text=event.delta)

                        case "toolcall_start":
                            # Yield early notification that tool call is starting
                            yield ToolCallStart(
                                id=event.tool_id,
                                name=event.tool_name,
                            )

                        case "toolcall_end":
                            # Convert ToolCallBlock to ToolCall
                            tc_block: ToolCallBlock = event.tool_call
                            tc = ToolCall(
                                id=tc_block.id,
                                name=tc_block.name,
                                arguments=tc_block.arguments,
                            )
                            tool_calls.append(tc)
                            yield tc

                        case "assistant_metadata":
                            if hasattr(event, "metadata") and isinstance(event.metadata, dict):
                                for key, value in event.metadata.items():
                                    if isinstance(value, dict) and isinstance(
                                        provider_metadata.get(key), dict
                                    ):
                                        provider_metadata[key].update(value)
                                    else:
                                        provider_metadata[key] = value

                        case "error":
                            stream_error = event.message.error_message or "LLM stream error"
                            stream_aborted = event.stop_reason == "aborted"
                            break
                        case "done":
                            # Stream complete
                            pass

                if stream_error:
                    if not stream_aborted:
                        error_msg = Message.system(f"[LLM stream error]\n{stream_error}")
                        self.session.append(error_msg)
                        await self._emit(MessageEndEvent(message=error_msg))
                        yield error_msg
                        await self._emit(TurnEndEvent(message=None, tool_results=[]))
                    break

                # End thinking if started
                if thinking_started:
                    await self._emit(ThinkingEndEvent(content=thinking_content))

                # Build message with thinking
                thinking_obj = (
                    ThinkingContent(text=thinking_content or "") if thinking_content else None
                )
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls if tool_calls else None,
                    thinking=thinking_obj,
                    provider_metadata=provider_metadata or None,
                    provider=self.config.provider,
                    model=getattr(self.provider, "model", None),
                )

                # Save assistant response
                self.session.append(assistant_msg)
                await self._emit(MessageEndEvent(message=assistant_msg))

                # If no tool calls, we're done
                if not tool_calls:
                    await self._emit(TurnEndEvent(message=assistant_msg, tool_results=[]))
                    break

                # Execute tools and add results
                tool_results: list[ToolResult] = []
                for tool_call in tool_calls:
                    # Check cancellation before each tool
                    if cancel_event and cancel_event.is_set():
                        break

                    await self._emit(
                        ToolExecutionStartEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            args=tool_call.arguments,
                        )
                    )

                    # Check if extension wants to block
                    block_result = await self._extension_runner.emit_tool_call(
                        ToolCallEvent(
                            tool_name=tool_call.name,
                            tool_call_id=tool_call.id,
                            input=tool_call.arguments,
                        )
                    )

                    if block_result and block_result.block:
                        result = f"Tool blocked: {block_result.reason or 'blocked by extension'}"
                        is_error = True
                    else:
                        exec_result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        result = exec_result.content
                        is_error = exec_result.is_error

                    # Allow extension to modify result
                    mod = await self._extension_runner.emit_tool_result(
                        ToolResultEvent(
                            tool_name=tool_call.name,
                            tool_call_id=tool_call.id,
                            content=result,
                            is_error=is_error,
                        )
                    )
                    if mod:
                        if mod.content is not None:
                            result = mod.content
                        if mod.is_error is not None:
                            is_error = mod.is_error

                    await self._emit(
                        ToolExecutionEndEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=result,
                            is_error=is_error,
                        )
                    )

                    self.session.append(
                        Message(
                            role=Role.TOOL,
                            content=result,
                            tool_call_id=tool_call.id,
                        )
                    )

                    tr = ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        result=result,
                    )
                    tool_results.append(tr)
                    yield tr

                await self._emit(TurnEndEvent(message=assistant_msg, tool_results=tool_results))

                # Check cancellation after tool execution
                if cancel_event and cancel_event.is_set():
                    break

            # Update token count
            self._total_tokens = self.context.current_tokens(self.session.messages)
        finally:
            self._in_loop = False

    async def compact(self) -> None:
        """Manually trigger context compaction."""
        original_tokens = self.context.current_tokens(self.session.messages)
        compaction = await self.context.compact(self.session.messages)
        self.session.append_compaction(
            compaction.summary,
            compaction.first_kept_id,
            tokens_before=original_tokens,
        )
        self._total_tokens = self.context.current_tokens(self.session.messages)

    async def close(self) -> None:
        """Clean up resources."""
        await self.provider.close()
