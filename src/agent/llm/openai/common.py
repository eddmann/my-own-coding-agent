"""Shared helpers for OpenAI providers (chat + responses)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
import tiktoken

from agent.llm.events import Cost, StopReason, Usage
from agent.llm.pricing import get_pricing

if TYPE_CHECKING:
    from agent.core.message import Message

OPENAI_API_URL = "https://api.openai.com"


class OpenAIError(Exception):
    """Error from the OpenAI API with parsed message."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def parse_api_error(response: httpx.Response) -> str:
    """Extract error message from API response."""
    try:
        data = response.json()
        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                return str(error.get("message", error))
            return str(error)
        return response.text or f"HTTP {response.status_code}"
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def map_stop_reason(openai_reason: str | None) -> StopReason:
    """Map OpenAI finish reason to our StopReason type."""
    mapping: dict[str | None, StopReason] = {
        "stop": "stop",
        "tool_calls": "tool_use",
        "length": "length",
        "content_filter": "stop",
        None: "stop",
    }
    return mapping.get(openai_reason, "stop")


@dataclass(slots=True)
class ModelCapabilities:
    """Model-specific capabilities and parameter support."""

    supports_temperature: bool = True
    token_param: str = "max_tokens"  # or "max_completion_tokens"
    is_reasoning: bool = False
    fixed_reasoning_effort: str | None = None
    max_output_tokens: int = 16384  # Model's max completion tokens


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Get model-specific capabilities based on model name."""
    model_lower = model.lower()

    # gpt-5-pro - only high reasoning, no temperature
    if model_lower.startswith("gpt-5-pro"):
        return ModelCapabilities(
            supports_temperature=False,
            token_param="max_completion_tokens",
            is_reasoning=True,
            fixed_reasoning_effort="high",
        )

    # o1/o3 + gpt-5* - no temperature, use max_completion_tokens
    if model_lower.startswith(("o1", "o3", "gpt-5")):
        return ModelCapabilities(
            supports_temperature=False,
            token_param="max_completion_tokens",
            is_reasoning=True,
        )

    # Default: gpt-4o, gpt-4, gpt-3.5, etc.
    return ModelCapabilities(
        supports_temperature=True,
        token_param="max_tokens",
        is_reasoning=False,
    )


def uses_responses_api(model: str) -> bool:
    """Return True if the model should use the Responses API."""
    return "codex" in model.lower()


def map_reasoning_effort(model: str, thinking_level: str | None) -> str | None:
    """Map generic thinking levels to OpenAI reasoning effort values."""
    if not thinking_level:
        return None

    if thinking_level in {"minimal", "low", "medium", "high"}:
        effort_map = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        return effort_map[thinking_level]

    if thinking_level != "xhigh":
        return None

    # xhigh is only available for select models (e.g., GPT-5.2/5.3 families).
    from agent.llm.models import supports_xhigh

    return "xhigh" if supports_xhigh(model, provider="openai") else "high"


def normalize_fc_id(value: str) -> str:
    """Normalize function call IDs for Responses API (must start with 'fc')."""
    if not value:
        return value
    if value.startswith("fc"):
        return value
    return f"fc_{value}"


def apply_pricing(usage: Usage, model: str) -> None:
    """Populate cost info on a Usage object if pricing is known."""
    pricing = get_pricing(model, "openai")
    if not pricing:
        return

    usage.cost = Cost(
        input=(usage.input / 1_000_000) * pricing.input,
        output=(usage.output / 1_000_000) * pricing.output,
        cache_read=(usage.cache_read / 1_000_000) * pricing.cache_read,
        cache_write=(usage.cache_write / 1_000_000) * pricing.cache_write,
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


def parse_streaming_json(text: str) -> dict[str, object]:
    """Parse partial JSON during streaming, returning best-effort dict."""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        # Try progressively shorter suffixes to salvage a valid object
        for i in range(len(text) - 1, 1, -1):
            try:
                parsed = json.loads(text[:i])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}


@dataclass(slots=True)
class OpenAIBase:
    """Shared base for OpenAI providers."""

    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _encoder: tiktoken.Encoding | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.http_client is not None:
            self._client = self.http_client

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Get or create the token encoder."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _get_capabilities(self) -> ModelCapabilities:
        """Get capabilities for the current model."""
        return get_model_capabilities(self.model)

    def set_model(self, model: str) -> None:
        """Update model and clear model-scoped caches."""
        from agent.llm.models import is_model_valid_for_provider

        if not model or model == self.model:
            return
        if not is_model_valid_for_provider(model, "openai"):
            raise ValueError(f"Model '{model}' is not valid for provider 'openai'")
        self.model = model
        self._encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoder.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += 4  # Overhead per message
            total += self.count_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.count_tokens(tc.name)
                    total += self.count_tokens(json.dumps(tc.arguments))
        return total

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning.

        Uses model registry for known models, falls back to
        capability detection for unknown models.
        """
        from agent.llm.models import get_model_info, supports_reasoning

        # Check registry first
        info = get_model_info(self.model)
        if info is not None:
            return info.reasoning

        # Fallback to name-based detection
        return supports_reasoning(self.model, provider="openai")

    async def list_models(self) -> list[str]:
        """Return known OpenAI models from the local registry."""
        from agent.llm.models import MODELS

        models = [model_id for model_id, info in MODELS.items() if info.provider == "openai"]
        return sorted(models)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
