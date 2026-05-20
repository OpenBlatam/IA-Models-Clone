"""
System 5.9 — Telemetry Service.

High-fidelity span-based profiling with error counting,
unclosed-span reaping, and event-bus integration.
"""

import time
import functools
import logging
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager

from .events import event_bus, EventType

logger = logging.getLogger("sys5.telemetry")


# ---------------------------------------------------------------------------
# Span record
# ---------------------------------------------------------------------------

@dataclass
class SpanRecord:
    """A single profiled operation."""
    name: str
    phase: str
    start: float
    end: float = 0.0
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000 if self.end else 0.0

    @property
    def is_closed(self) -> bool:
        return self.end > 0.0

    def __repr__(self) -> str:
        state = "✓" if self.is_closed else "…"
        return f"<Span {state} {self.name} {self.duration_ms:.1f}ms [{self.status}]>"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TelemetryService:
    """Collects span records and rolling counters for the training system."""

    _EWMA_ALPHA = 0.05  # exponential-weighted moving average weight

    def __init__(self) -> None:
        self._spans: List[SpanRecord] = []
        self._counters: Dict[str, int] = {
            "total_calls": 0,
            "errors": 0,
            "circuit_breaker_activations": 0,
            "sanitized_prompts": 0,
            "degraded_tool_calls": 0,
            "training_steps": 0,
        }
        self._avg_latency_ms: float = 0.0

    # -- context managers ---------------------------------------------------

    @asynccontextmanager
    async def span(self, name: str, **meta: Any):
        """Async context manager — always closes the span."""
        record = self._open(name, meta)
        try:
            yield record
        except Exception as exc:
            self._mark_error(record, exc)
            raise
        finally:
            self._close(record, meta)
            try:
                await event_bus.emit(EventType.TELEMETRY, {
                    "name": name,
                    "duration_ms": round(record.duration_ms, 2),
                    "status": record.status,
                    **meta,
                })
            except Exception as e:
                self._counters["errors"] += 1
                logger.warning("Event emit failed for span '%s': %s", name, e)

    @contextmanager
    def sync_span(self, name: str, **meta: Any):
        """Synchronous context manager — always closes the span."""
        record = self._open(name, meta)
        try:
            yield record
        except Exception as exc:
            self._mark_error(record, exc)
            raise
        finally:
            self._close(record, meta)

    # -- counters -----------------------------------------------------------

    def increment(self, name: str, amount: int = 1) -> None:
        """Bump a named counter (creates it if missing)."""
        self._counters[name] = self._counters.get(name, 0) + amount

    # -- reaper -------------------------------------------------------------

    def reap_unclosed(self, max_age: float = 60.0) -> int:
        """Force-close spans older than *max_age* seconds."""
        now = time.perf_counter()
        reaped = 0
        for s in self._spans:
            if not s.is_closed and (now - s.start) > max_age:
                s.end = now
                s.status = "timeout"
                s.metadata["reaped"] = True
                s.metadata["age_s"] = round(now - s.start, 2)
                reaped += 1
                logger.warning("Reaped span '%s' (age %.1fs)", s.name, now - s.start)
        if reaped:
            self._counters["errors"] += reaped
        return reaped

    # -- reporting ----------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        unclosed = sum(1 for s in self._spans if not s.is_closed)
        return {
            "total_calls": self._counters["total_calls"],
            "total_errors": self._counters["errors"],
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "span_count": len(self._spans),
            "unclosed_spans": unclosed,
            "counters": dict(self._counters),
        }

    def __repr__(self) -> str:
        s = self.get_summary()
        return (
            f"<TelemetryService calls={s['total_calls']} "
            f"errors={s['total_errors']} "
            f"avg={s['avg_latency_ms']:.1f}ms>"
        )

    # -- internals ----------------------------------------------------------

    def _open(self, name: str, meta: dict) -> SpanRecord:
        return SpanRecord(
            name=name,
            phase=meta.get("phase", "General"),
            start=time.perf_counter(),
            metadata=dict(meta),
        )

    def _mark_error(self, record: SpanRecord, exc: Exception) -> None:
        record.status = "error"
        record.metadata["error"] = str(exc)
        self._counters["errors"] += 1
        logger.error("Span '%s' failed: %s", record.name, exc)

    def _close(self, record: SpanRecord, meta: dict) -> None:
        record.end = time.perf_counter()
        self._spans.append(record)
        self._counters["total_calls"] += 1
        self._avg_latency_ms = (
            self._avg_latency_ms * (1 - self._EWMA_ALPHA)
            + record.duration_ms * self._EWMA_ALPHA
        )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def tracked(phase: str = "General"):
    """
    Decorator that wraps sync/async functions in a telemetry span.

    Usage::

        @tracked("Inference")
        async def do_inference(self, prompt): ...
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async(*args, **kwargs):
                telem = _resolve_telemetry()
                if telem is None:
                    return await func(*args, **kwargs)
                component = _component_name(args)
                async with telem.span(func.__name__, phase=phase, component=component):
                    return await func(*args, **kwargs)
            return _async
        else:
            @functools.wraps(func)
            def _sync(*args, **kwargs):
                telem = _resolve_telemetry()
                if telem is None:
                    return func(*args, **kwargs)
                component = _component_name(args)
                with telem.sync_span(func.__name__, phase=phase, component=component):
                    return func(*args, **kwargs)
            return _sync
    return decorator


def _resolve_telemetry() -> Optional[TelemetryService]:
    from .registry import registry
    return registry.get("TelemetryService")


def _component_name(args: tuple) -> str:
    if args and hasattr(args[0], "__class__"):
        return args[0].__class__.__name__
    return "unknown"
