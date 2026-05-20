"""
Telemetry System for Document Analyzer
========================================

Advanced telemetry for monitoring and analytics.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class TelemetryEvent:
    """Telemetry event"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "properties": self.properties,
            "metrics": self.metrics,
            "user_id": self.user_id,
            "session_id": self.session_id
        }

class TelemetryCollector:
    """Advanced telemetry collector"""
    
    def __init__(self, max_events: int = 10000, flush_interval: float = 60.0):
        self.max_events = max_events
        self.flush_interval = flush_interval
        self.events: deque = deque(maxlen=max_events)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.is_flushing = False
        logger.info(f"TelemetryCollector initialized. Max events: {max_events}, Flush interval: {flush_interval}s")
    
    def track_event(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Track an event"""
        event = TelemetryEvent(
            event_type=event_type,
            properties=properties or {},
            metrics=metrics or {},
            user_id=user_id,
            session_id=session_id
        )
        
        self.events.append(event)
        self.counters[f"events.{event_type}"] += 1
        
        logger.debug(f"Tracked event: {event_type}")
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Record a histogram value"""
        self.histograms[name].append(value)
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics"""
        return {
            "total_events": len(self.events),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0
                }
                for name, values in self.histograms.items()
            },
            "event_types": {
                event_type: self.counters.get(f"events.{event_type}", 0)
                for event_type in set(e.event_type for e in self.events)
            }
        }
    
    async def flush(self, handler: Optional[Callable] = None):
        """Flush events to handler"""
        if self.is_flushing or len(self.events) == 0:
            return
        
        self.is_flushing = True
        try:
            events_to_flush = list(self.events)
            self.events.clear()
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(events_to_flush)
                else:
                    handler(events_to_flush)
            else:
                # Default: log events
                logger.info(f"Flushed {len(events_to_flush)} telemetry events")
        finally:
            self.is_flushing = False
    
    async def start_auto_flush(self, handler: Optional[Callable] = None):
        """Start automatic flushing"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush(handler)

class TelemetryTracker:
    """Context manager for tracking operations"""
    
    def __init__(
        self,
        collector: TelemetryCollector,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.collector = collector
        self.event_type = event_type
        self.properties = properties or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        metrics = {
            "duration": duration,
            "success": exc_type is None
        }
        
        if exc_type:
            self.properties["error"] = str(exc_val)
        
        self.collector.track_event(
            event_type=self.event_type,
            properties=self.properties,
            metrics=metrics
        )
        
        self.collector.record_histogram(f"{self.event_type}.duration", duration)
        
        return False

# Global instance
telemetry_collector = TelemetryCollector()
















