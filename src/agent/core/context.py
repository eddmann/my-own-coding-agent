"""Context window management and compaction."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.core.message import Message, Role
from agent.llm.events import StreamOptions

if TYPE_CHECKING:
    from agent.llm.provider import LLMProvider


@dataclass(slots=True)
class CompactionResult:
    summary: str
    first_kept_id: str | None


@dataclass(slots=True)
class ContextManager:
    """Manages context window and compaction."""

    provider: LLMProvider
    max_tokens: int
    reserve_tokens: int = 8192  # Reserve for response and tools
    keep_recent: int = 10  # Always keep last N messages

    def current_tokens(self, messages: list[Message]) -> int:
        """Calculate total tokens in messages."""
        return self.provider.count_messages_tokens(messages)

    def needs_compaction(self, messages: list[Message]) -> bool:
        """Check if compaction is needed."""
        if len(messages) <= self.keep_recent + 1:
            return False
        available = self.max_tokens - self.reserve_tokens
        return self.current_tokens(messages) > available * 0.8

    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Summarize older messages, keep recent ones.

        Strategy:
        1. Always keep system messages
        2. Always keep the most recent messages
        3. Summarize the middle section
        Returns a CompactionResult with the summary and first kept message id.
        """
        if len(messages) <= self.keep_recent + 1:
            return CompactionResult(summary="", first_kept_id=None)

        # Separate system messages from conversation
        other_msgs = [m for m in messages if m.role != Role.SYSTEM]

        if len(other_msgs) <= self.keep_recent:
            first_kept_id = other_msgs[0].id if other_msgs else None
            return CompactionResult(summary="", first_kept_id=first_kept_id)

        # Split into old and recent
        old_msgs = other_msgs[: -self.keep_recent]
        recent_msgs = other_msgs[-self.keep_recent :]

        # Generate summary of old messages
        summary = await self._generate_summary(old_msgs)

        first_kept_id = recent_msgs[0].id if recent_msgs else None
        return CompactionResult(summary=summary, first_kept_id=first_kept_id)

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Generate a summary of messages using the LLM."""
        # Build a prompt for summarization
        conversation_text = self._format_conversation(messages)

        summary_prompt = Message.user(
            "Summarize the following conversation concisely.\n"
            "Output markdown with these headings in order:\n"
            "1) Summary\n"
            "2) Decisions\n"
            "3) Files Read\n"
            "4) Files Modified\n"
            "5) Commands Run\n"
            "6) Tools Used\n"
            "7) Open TODOs\n"
            "8) Risks/Concerns\n\n"
            "Rules:\n"
            "- Do NOT include system prompt text or policies.\n"
            "- Do NOT include secrets (API keys, tokens, passwords). Redact as [REDACTED].\n"
            "- Keep bullets short and actionable.\n\n"
            f"Conversation:\n{conversation_text}"
        )

        summary_response = ""
        error_message: str | None = None

        try:
            stream = self.provider.stream(
                [summary_prompt],
                tools=None,
                options=StreamOptions(
                    temperature=0.2,
                    max_tokens=1024,
                    tool_choice="none",
                ),
            )
            async for event in stream:
                if event.type == "text_delta":
                    summary_response += event.delta
                elif event.type == "error":
                    error_message = event.message.error_message or "summary generation failed"
                    break
        except Exception as e:
            error_message = str(e)

        summary_response = summary_response.strip()
        if not summary_response:
            return self._fallback_summary(messages, error_message)

        return summary_response

    def _fallback_summary(self, messages: list[Message], error: str | None) -> str:
        """Fallback summary when the model fails."""
        files_read: set[str] = set()
        files_modified: set[str] = set()
        commands: list[str] = []
        tools_used: set[str] = set()
        searches: list[str] = []

        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.add(tc.name)
                    if tc.name == "read":
                        if path := tc.arguments.get("path"):
                            files_read.add(self._redact(str(path)))
                    elif tc.name in ("write", "edit"):
                        if path := tc.arguments.get("path"):
                            files_modified.add(self._redact(str(path)))
                    elif tc.name == "bash":
                        if cmd := tc.arguments.get("command"):
                            commands.append(self._redact(str(cmd)))
                    elif tc.name in ("grep", "find"):
                        pattern = tc.arguments.get("pattern")
                        path = tc.arguments.get("path")
                        if pattern or path:
                            searches.append(
                                self._redact(
                                    " ".join(s for s in [str(pattern or ""), str(path or "")] if s)
                                )
                            )

        # Collect recent user requests for context
        recent_user: list[str] = []
        for msg in reversed(messages):
            if msg.role == Role.USER:
                recent_user.append(self._redact(msg.content.strip()))
            if len(recent_user) >= 3:
                break

        lines = []
        if error:
            lines.append(f"[Summary unavailable: {self._redact(error)}]")
            lines.append("")

        lines.append("## Summary")
        lines.append(f"- Messages summarized: {len(messages)}")
        if tools_used:
            lines.append(f"- Tools used: {', '.join(sorted(tools_used))}")
        if searches:
            lines.append(f"- Searches: {len(searches)}")
        lines.append("")

        lines.append("## Decisions")
        lines.append("- (not available)")
        lines.append("")

        lines.append("## Files Read")
        if files_read:
            for path in sorted(files_read):
                lines.append(f"- {path}")
        else:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Files Modified")
        if files_modified:
            for path in sorted(files_modified):
                lines.append(f"- {path}")
        else:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Commands Run")
        if commands:
            for cmd in commands[:10]:
                lines.append(f"- {cmd}")
        else:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Tools Used")
        if tools_used:
            for name in sorted(tools_used):
                lines.append(f"- {name}")
        else:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Open TODOs")
        lines.append("- (not available)")
        lines.append("")

        lines.append("## Risks/Concerns")
        lines.append("- (not available)")
        lines.append("")

        if recent_user:
            lines.append("## Recent User Requests")
            for item in reversed(recent_user):
                if item:
                    lines.append(f"- {item[:300]}")

        return "\n".join(lines)

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format messages as readable text for summarization."""
        parts = []
        for msg in messages:
            role_name = msg.role.value.upper()
            content = self._redact(msg.content[:2000])  # Truncate long messages
            if msg.tool_calls:
                tool_names = [tc.name for tc in msg.tool_calls]
                content += f" [Called tools: {', '.join(tool_names)}]"
            parts.append(f"{role_name}: {content}")
        return "\n\n".join(parts)

    def _redact(self, text: str) -> str:
        """Redact likely secrets from text."""
        if not text:
            return text

        patterns = [
            r"sk-[A-Za-z0-9]{10,}",
            r"AIza[0-9A-Za-z\-_]{10,}",
            r"(?i)api[_-]?key\s*[:=]\s*\S+",
            r"(?i)token\s*[:=]\s*\S+",
            r"(?i)password\s*[:=]\s*\S+",
        ]

        redacted = text
        for pattern in patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted)
        return redacted
