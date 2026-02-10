"""Event stream for async iteration with final result."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from agent.llm.events import ErrorEvent, PartialMessage, StreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

T = TypeVar("T")
R = TypeVar("R")


class EventStream[T, R]:
    """Async event stream with final result promise.

    Designed to follow a simple event-stream pattern.
    Allows consumers to iterate over events while also awaiting final result.
    """

    __slots__ = (
        "_queue",
        "_waiting",
        "_done",
        "_result_future",
        "_is_complete",
        "_extract_result",
    )

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        """Initialize the event stream.

        Args:
            is_complete: Predicate to check if an event signals completion
            extract_result: Function to extract final result from completion event
        """
        self._queue: list[T] = []
        self._waiting: list[asyncio.Future[T | None]] = []
        self._done = False
        self._result_future: asyncio.Future[R] | None = None
        self._is_complete = is_complete
        self._extract_result = extract_result

    def _get_result_future(self) -> asyncio.Future[R]:
        """Get or create the result future using the running loop."""
        if self._result_future is None:
            loop = asyncio.get_running_loop()
            self._result_future = loop.create_future()
        return self._result_future

    def push(self, event: T) -> None:
        """Push event to consumers.

        If consumers are waiting, delivers directly.
        Otherwise, queues for later consumption.
        """
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            future = self._get_result_future()
            if not future.done():
                future.set_result(self._extract_result(event))

        if self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result(event)
        else:
            self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        """End the stream.

        Args:
            result: Optional result to set if not already set by completion event
        """
        self._done = True
        future = self._get_result_future()
        if result is not None and not future.done():
            future.set_result(result)
        for waiter in self._waiting:
            if not waiter.done():
                waiter.set_result(None)
        self._waiting.clear()

    async def result(self) -> R:
        """Await final result."""
        return await self._get_result_future()

    @property
    def is_done(self) -> bool:
        """Check if stream is complete."""
        return self._done

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self._queue:
            return self._queue.pop(0)
        if self._done:
            raise StopAsyncIteration
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T | None] = loop.create_future()
        self._waiting.append(future)
        event = await future
        if event is None:
            raise StopAsyncIteration
        return event


class AssistantMessageEventStream(EventStream[StreamEvent, PartialMessage]):
    """Typed event stream for assistant messages.

    Convenience wrapper that pre-configures completion detection
    for done/error events. Supports early abort via abort() method.
    """

    __slots__ = ("_aborted", "_task")

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda e: e.type in ("done", "error"),
            extract_result=lambda e: e.message,  # type: ignore[union-attr]
        )
        self._aborted = False
        self._task: asyncio.Task[None] | None = None  # Holds reference to streaming task

    def attach_task(self, task: asyncio.Task[None]) -> None:
        """Attach the producer task to keep a strong reference."""
        self._task = task

    def abort(self, reason: str = "aborted") -> None:
        """Abort the stream early.

        Args:
            reason: Human-readable reason for the abort
        """
        if self._done:
            return
        self._aborted = True
        # Create error message with abort info
        msg = PartialMessage(stop_reason="aborted", error_message=reason)
        self.push(ErrorEvent(stop_reason="aborted", message=msg))
        self.end(msg)

    @property
    def is_aborted(self) -> bool:
        """Check if the stream was aborted."""
        return self._aborted
