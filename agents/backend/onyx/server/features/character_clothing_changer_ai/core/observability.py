"""
Observability System
====================

Comprehensive observability system for monitoring and debugging.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    service: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "service": self.service,
            "context": self.context,
            "tags": self.tags,
            "duration_ms": self.duration_ms
        }


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class Span:
    """Observability span."""
    name: str
    service: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[LogEntry] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "service": self.service,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": [log.to_dict() for log in self.logs],
            "metrics": [metric.to_dict() for metric in self.metrics]
        }


class ObservabilityManager:
    """Comprehensive observability manager."""
    
    def __init__(self, service_name: str):
        """
        Initialize observability manager.
        
        Args:
            service_name: Service name
        """
        self.service_name = service_name
        self.logs: List[LogEntry] = []
        self.metrics: List[Metric] = []
        self.spans: List[Span] = []
        self.max_entries = 10000  # Keep last 10000 entries
    
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Log an entry.
        
        Args:
            level: Log level
            message: Log message
            context: Optional context data
            tags: Optional tags
            duration_ms: Optional duration
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            service=self.service_name,
            context=context or {},
            tags=tags or [],
            duration_ms=duration_ms
        )
        
        self.logs.append(entry)
        
        # Limit logs
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]
        
        # Also log to standard logger
        log_func = getattr(logger, level.value, logger.info)
        log_func(f"[{self.service_name}] {message}", extra=context or {})
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            unit: Optional unit
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        self.metrics.append(metric)
        
        # Limit metrics
        if len(self.metrics) > self.max_entries:
            self.metrics = self.metrics[-self.max_entries:]
    
    def start_span(
        self,
        name: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a span.
        
        Args:
            name: Span name
            tags: Optional tags
            
        Returns:
            Span object
        """
        span = Span(
            name=name,
            service=self.service_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        self.spans.append(span)
        return span
    
    def end_span(self, span: Span):
        """
        End a span.
        
        Args:
            span: Span to end
        """
        span.end_time = time.time()
    
    @contextmanager
    def span(self, name: str, tags: Optional[Dict[str, Any]] = None):
        """
        Context manager for span.
        
        Args:
            name: Span name
            tags: Optional tags
        """
        span = self.start_span(name, tags)
        try:
            yield span
        finally:
            self.end_span(span)
    
    def get_recent_logs(
        self,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Get recent logs.
        
        Args:
            level: Optional log level filter
            limit: Number of logs to return
            
        Returns:
            List of log entries
        """
        logs = self.logs
        if level:
            logs = [log for log in logs if log.level == level]
        return logs[-limit:]
    
    def get_recent_metrics(
        self,
        name: Optional[str] = None,
        limit: int = 100
    ) -> List[Metric]:
        """
        Get recent metrics.
        
        Args:
            name: Optional metric name filter
            limit: Number of metrics to return
            
        Returns:
            List of metrics
        """
        metrics = self.metrics
        if name:
            metrics = [m for m in metrics if m.name == name]
        return metrics[-limit:]
    
    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """
        Get recent spans.
        
        Args:
            limit: Number of spans to return
            
        Returns:
            List of spans
        """
        return self.spans[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get observability summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "service": self.service_name,
            "total_logs": len(self.logs),
            "total_metrics": len(self.metrics),
            "total_spans": len(self.spans),
            "recent_logs": len(self.get_recent_logs()),
            "recent_metrics": len(self.get_recent_metrics()),
            "recent_spans": len(self.get_recent_spans())
        }

