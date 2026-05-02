"""
Frontier Model Polyglot — System 5.0+ Telemetry.
High-fidelity profiling for large-scale model training.
"""
import time
import functools
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from .events import event_bus, EventType

@dataclass
class SpanRecord:
    """A profiled operation span."""
    name: str
    phase: str
    start: float
    end: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000 if self.end else 0.0

class TelemetryService:
    """Performance tracking for the Frontier Model system."""
    
    def __init__(self):
        self._spans: List[SpanRecord] = []
        self._counters: Dict[str, int] = {
            "total_calls": 0,
            "errors": 0,
            "training_steps": 0
        }
        self._avg_latency_ms = 0.0
        self.log = logging.getLogger("TelemetryService")

    @asynccontextmanager
    async def span(self, name: str, **metadata):
        """Async context manager for profiling operations."""
        record = SpanRecord(
            name=name,
            phase=metadata.get("phase", "General"),
            start=time.perf_counter(),
            metadata=metadata,
        )
        try:
            yield record
        finally:
            record.end = time.perf_counter()
            self._spans.append(record)
            self._counters["total_calls"] += 1
            
            # Update rolling average
            self._avg_latency_ms = (self._avg_latency_ms * 0.95) + (record.duration_ms * 0.05)
            
            # Emit telemetry event
            try:
                await event_bus.emit(EventType.TELEMETRY, {
                    "name": name,
                    "duration_ms": round(record.duration_ms, 2),
                    **metadata
                })
            except: pass

    def get_summary(self) -> Dict[str, Any]:
        count = len(self._spans)
        start_idx = max(0, count - 10)
        recent = self._spans[start_idx:]
        
        return {
            "total_calls": self._counters["total_calls"],
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "recent_spans": len(recent)
        }

def tracked(phase: str = "General"):
    """
    Decorator for System 5.0+ operation tracking.
    Supports both synchronous and asynchronous functions.
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from .registry import registry
                telem = registry.get("TelemetryService")
                if not telem:
                    return await func(*args, **kwargs)

                component = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else "unknown"
                async with telem.span(func.__name__, phase=phase, component=component):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from .registry import registry
                telem = registry.get("TelemetryService")
                if not telem:
                    return func(*args, **kwargs)

                component = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else "unknown"
                
                # For sync, we use a different approach since span is an async context manager
                # We'll just emit events manually or use a sync span if we had one
                # But since the event bus is async, we'll fire and forget
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = (time.perf_counter() - start) * 1000
                    if event_bus:
                        try:
                            # Fire and forget telemetry event
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(event_bus.emit(EventType.TELEMETRY, {
                                    "name": func.__name__,
                                    "duration_ms": round(duration, 2),
                                    "phase": phase,
                                    "component": component
                                }))
                        except: pass
            return sync_wrapper
    return decorator
