"""
Telemetry System
================

Advanced telemetry system for collecting and analyzing system data.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class TelemetryType(Enum):
    """Telemetry types."""
    METRIC = "metric"
    EVENT = "event"
    LOG = "log"
    TRACE = "trace"
    SPAN = "span"


@dataclass
class TelemetryData:
    """Telemetry data point."""
    type: TelemetryType
    name: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """Telemetry data collector."""
    
    def __init__(self, max_buffer_size: int = 10000):
        """
        Initialize telemetry collector.
        
        Args:
            max_buffer_size: Maximum buffer size
        """
        self.max_buffer_size = max_buffer_size
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.handlers: List[Callable] = []
        self.enabled = True
    
    def collect(self, data: TelemetryData):
        """
        Collect telemetry data.
        
        Args:
            data: Telemetry data point
        """
        if not self.enabled:
            return
        
        self.buffer.append(data)
        
        # Call handlers
        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Telemetry handler failed: {e}")
    
    def add_handler(self, handler: Callable):
        """
        Add telemetry handler.
        
        Args:
            handler: Handler function
        """
        self.handlers.append(handler)
    
    def get_recent_data(
        self,
        data_type: Optional[TelemetryType] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[TelemetryData]:
        """
        Get recent telemetry data.
        
        Args:
            data_type: Optional filter by type
            since: Optional filter by timestamp
            limit: Optional limit results
            
        Returns:
            List of telemetry data
        """
        results = list(self.buffer)
        
        # Filter by type
        if data_type:
            results = [d for d in results if d.type == data_type]
        
        # Filter by timestamp
        if since:
            results = [d for d in results if d.timestamp >= since]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda d: d.timestamp, reverse=True)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get telemetry statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self.buffer)
        by_type = {}
        
        for data in self.buffer:
            type_name = data.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total": total,
            "by_type": by_type,
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.max_buffer_size,
            "handlers": len(self.handlers),
            "enabled": self.enabled
        }
    
    def clear(self):
        """Clear telemetry buffer."""
        self.buffer.clear()


class TelemetryManager:
    """Manager for multiple telemetry collectors."""
    
    def __init__(self):
        """Initialize telemetry manager."""
        self.collectors: Dict[str, TelemetryCollector] = {}
        self.default_collector = TelemetryCollector()
    
    def get_collector(self, name: str) -> TelemetryCollector:
        """
        Get or create telemetry collector.
        
        Args:
            name: Collector name
            
        Returns:
            Telemetry collector
        """
        if name not in self.collectors:
            self.collectors[name] = TelemetryCollector()
        return self.collectors[name]
    
    def collect_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        collector: Optional[str] = None
    ):
        """
        Collect metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            collector: Optional collector name
        """
        data = TelemetryData(
            type=TelemetryType.METRIC,
            name=name,
            value=value,
            tags=tags or {}
        )
        
        if collector:
            self.get_collector(collector).collect(data)
        else:
            self.default_collector.collect(data)
    
    def collect_event(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        collector: Optional[str] = None
    ):
        """
        Collect event.
        
        Args:
            name: Event name
            metadata: Optional metadata
            tags: Optional tags
            collector: Optional collector name
        """
        data = TelemetryData(
            type=TelemetryType.EVENT,
            name=name,
            value=1,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        if collector:
            self.get_collector(collector).collect(data)
        else:
            self.default_collector.collect(data)
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all collectors.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "default": self.default_collector.get_statistics()
        }
        
        for name, collector in self.collectors.items():
            stats[name] = collector.get_statistics()
        
        return stats

