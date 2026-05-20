"""
Tracing Service Implementation
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from uuid import uuid4

from .base import TracingBase, Trace, Span, TraceContext

logger = logging.getLogger(__name__)


class TracingService(TracingBase):
    """Tracing service implementation"""
    
    def __init__(self, config_service=None):
        """Initialize tracing service"""
        self.config_service = config_service
        self._traces: dict = {}
        self._spans: dict = {}
        self._enabled = True
    
    async def start_trace(self, name: str) -> Trace:
        """Start trace"""
        try:
            trace_id = str(uuid4())
            trace = Trace(trace_id=trace_id, name=name)
            self._traces[trace_id] = trace
            return trace
            
        except Exception as e:
            logger.error(f"Error starting trace: {e}")
            raise
    
    async def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None
    ) -> Span:
        """Start span"""
        try:
            span_id = str(uuid4())
            span = Span(
                span_id=span_id,
                trace_id=trace_id,
                name=name,
                start_time=datetime.utcnow()
            )
            
            self._spans[span_id] = span
            
            trace = self._traces.get(trace_id)
            if trace:
                trace.spans.append(span)
            
            return span
            
        except Exception as e:
            logger.error(f"Error starting span: {e}")
            raise
    
    async def end_span(self, span: Span) -> bool:
        """End span"""
        try:
            span.end_time = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Error ending span: {e}")
            return False
    
    async def log_event(
        self,
        span: Span,
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log event"""
        try:
            if "events" not in span.attributes:
                span.attributes["events"] = []
            
            span.attributes["events"].append({
                "name": event_name,
                "attributes": attributes or {},
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False

