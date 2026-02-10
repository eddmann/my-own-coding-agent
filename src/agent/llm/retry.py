"""Retry logic with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 32.0
    jitter: bool = True


def parse_retry_after(response: httpx.Response) -> float | None:
    """Parse Retry-After header (seconds or HTTP-date).

    Args:
        response: HTTP response to check

    Returns:
        Seconds to wait, or None if header not present/parseable
    """
    retry_after = response.headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


async def with_retry(
    coro_func: Callable[[], Any],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
    """Execute async function with exponential backoff.

    Args:
        coro_func: Async callable to execute (must be callable, not coroutine)
        config: Retry configuration (defaults to RetryConfig())
        on_retry: Optional callback(attempt, exception, delay) called before retry

    Returns:
        Result from successful execution

    Raises:
        The last exception if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()

    delay = config.initial_delay
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e

            # Check if retriable
            if not _is_retriable(e):
                raise

            # Last attempt - don't retry
            if attempt == config.max_retries:
                raise

            # Calculate delay
            wait_time = _get_retry_delay(e, delay, config)

            if on_retry:
                on_retry(attempt, e, wait_time)

            await asyncio.sleep(wait_time)
            delay = min(delay * 2, config.max_delay)

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


def _is_retriable(e: Exception) -> bool:
    """Check if exception is retriable.

    Args:
        e: Exception to check

    Returns:
        True if the request should be retried
    """
    # Timeout errors
    if isinstance(e, asyncio.TimeoutError):
        return True
    if isinstance(e, httpx.TimeoutException):
        return True
    if isinstance(e, httpx.ConnectError):
        return True

    # Check for retriable status codes in our custom errors
    status_code = getattr(e, "status_code", None)
    if isinstance(status_code, int):
        return status_code in RETRIABLE_STATUS_CODES

    return False


def _get_retry_delay(e: Exception, base_delay: float, config: RetryConfig) -> float:
    """Get delay for retry, respecting Retry-After if available.

    Args:
        e: Exception that triggered retry
        base_delay: Current exponential backoff delay
        config: Retry configuration

    Returns:
        Seconds to wait before retry
    """
    # Check for Retry-After from rate limit
    response = getattr(e, "response", None)
    if isinstance(response, httpx.Response):
        retry_after = parse_retry_after(response)
        if retry_after:
            return retry_after

    delay = min(base_delay, config.max_delay)
    if config.jitter:
        delay = delay * (0.5 + random.random())
    return delay
