"""
🚀 TruthGPT SOTA Log Aggregator - System 5.9 Gold Standard
Centralized telemetry and event collection.
"""

import logging
import time
from collections import deque
from typing import List, Dict, Any
from pydantic import BaseModel

class LogEntry(BaseModel):
    timestamp: float
    layer: str
    level: str
    message: str

class LogAggregator:
    """
    Industrial Log Collector.
    Aggregates events from all 16 layers for real-time dashboard monitoring.
    """

    def __init__(self, max_history: int = 1000):
        self.history = deque(maxlen=max_history)
        self.stats = {"INFO": 0, "WARNING": 0, "ERROR": 0}

    def log(self, layer: str, message: str, level: str = "INFO"):
        """Record a system event."""
        entry = LogEntry(
            timestamp=time.time(),
            layer=layer,
            level=level,
            message=message
        )
        self.history.append(entry)
        self.stats[level] = self.stats.get(level, 0) + 1
        
        # Also print to standard logging
        getattr(logging, level.lower())(f"[{layer}] {message}")

    def get_recent(self, limit: int = 5) -> List[LogEntry]:
        """Get the last N events for the dashboard."""
        return list(self.history)[-limit:]

    def get_system_pulse(self) -> Dict[str, Any]:
        """Return a health summary based on recent logs."""
        return {
            "total_events": len(self.history),
            "errors": self.stats.get("ERROR", 0),
            "status": "Healthy" if self.stats.get("ERROR", 0) == 0 else "Degraded"
        }

# Global Aggregator
log_aggregator = LogAggregator()
