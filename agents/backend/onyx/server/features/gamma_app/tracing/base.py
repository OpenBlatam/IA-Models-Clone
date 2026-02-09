"""
Tracing Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


@dataclass
class Span:
    """Span in trace"""
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class TraceContext:
    """Trace context"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


class Trace:
    """Trace definition"""
    
    def __init__(self, trace_id: str, name: str):
        self.trace_id = trace_id
        self.name = name
        self.spans: list = []
        self.created_at = datetime.utcnow()


class TracingBase(ABC):
    """Base interface for tracing"""
    
    @abstractmethod
    async def start_trace(self, name: str) -> Trace:
        """Start trace"""
        pass
    
    @abstractmethod
    async def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None
    ) -> Span:
        """Start span"""
        pass
    
    @abstractmethod
    async def end_span(self, span: Span) -> bool:
        """End span"""
        pass
    
    @abstractmethod
    async def log_event(
        self,
        span: Span,
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log event"""
        pass

