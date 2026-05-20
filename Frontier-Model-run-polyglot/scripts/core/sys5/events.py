"""
System 5.9 — Async Event Bus.

Decoupled pub/sub for telemetry, circuit-breaker signals,
tool-health advisories, and prompt-sanitizer notifications.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List
from enum import Enum, auto

logger = logging.getLogger("sys5.events")


# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------

class EventType(Enum):
    """All observable events in the System 5.9 lifecycle."""

    TELEMETRY        = "telemetry"
    ERROR            = "error"
    TRAINING_STEP    = "training_step"
    MODEL_EVENT      = "model_event"
    SYSTEM           = "system"
    CIRCUIT_BREAKER  = "circuit_breaker"
    TOOL_DEGRADED    = "tool_degraded"
    PROMPT_SANITIZED = "prompt_sanitized"


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------

class Event:
    """Immutable event payload delivered to subscribers."""

    __slots__ = ("type", "data", "timestamp")

    def __init__(self, event_type: EventType, data: Any = None) -> None:
        self.type = event_type
        self.data = data if data is not None else {}
        self.timestamp: float = time.time()

    def __repr__(self) -> str:
        return f"<Event {self.type.value} t={self.timestamp:.3f}>"


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class AsyncEventBus:
    """Fire-and-forget async event bus with sync fallback emission."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable]] = {
            t: [] for t in EventType
        }

    # -- subscribe ----------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Register *callback* for *event_type* (sync or async callable)."""
        self._subscribers[event_type].append(callback)

    # -- emit (async) -------------------------------------------------------

    async def emit(self, event_type: EventType, data: Any = None) -> None:
        """Emit an event to all subscribers (awaits coroutines)."""
        event = Event(event_type, data)
        for cb in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Subscriber error for %s: %s", event_type.value, exc)

    # -- emit_sync ----------------------------------------------------------

    def emit_sync(self, event_type: EventType, data: Any = None) -> None:
        """
        Best-effort synchronous emit.

        Schedules the async *emit* on the running loop when available;
        otherwise invokes sync callbacks directly and skips async ones.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event_type, data))
        except RuntimeError:
            # No running loop — call sync callbacks only
            event = Event(event_type, data)
            for cb in self._subscribers[event_type]:
                if not asyncio.iscoroutinefunction(cb):
                    try:
                        cb(event)
                    except Exception as exc:
                        logger.error(
                            "Sync subscriber error for %s: %s",
                            event_type.value, exc,
                        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

event_bus = AsyncEventBus()
