"""
Distributed Tracing - Sistema de tracing distribuido
=====================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SpanKind(str, Enum):
    """Tipos de spans"""
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class Span:
    """Span de tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None


class DistributedTracing:
    """Sistema de tracing distribuido"""
    
    def __init__(self):
        self.traces: Dict[str, List[Span]] = {}
        self.active_spans: Dict[str, Span] = {}
    
    def start_trace(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """Inicia un nuevo trace"""
        trace_id = str(uuid4())
        span_id = str(uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            name=name,
            kind=SpanKind.SERVER,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        
        self.traces[trace_id] = [span]
        self.active_spans[span_id] = span
        
        logger.debug(f"Trace iniciado: {trace_id} - {name}")
        return trace_id
    
    def start_span(self, trace_id: str, name: str, 
                   parent_span_id: Optional[str] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   attributes: Optional[Dict[str, Any]] = None) -> str:
        """Inicia un nuevo span"""
        span_id = str(uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=datetime.now(),
            attributes=attributes or {}
        )
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span
        
        logger.debug(f"Span iniciado: {span_id} - {name}")
        return span_id
    
    def end_span(self, span_id: str, status: str = "ok", error: Optional[str] = None):
        """Termina un span"""
        span = self.active_spans.get(span_id)
        if not span:
            logger.warning(f"Span no encontrado: {span_id}")
            return
        
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error = error
        
        del self.active_spans[span_id]
        logger.debug(f"Span terminado: {span_id} - Duración: {span.duration_ms:.2f}ms")
    
    def add_event(self, span_id: str, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Agrega un evento a un span"""
        span = self.active_spans.get(span_id)
        if not span:
            # Buscar en traces completados
            for spans in self.traces.values():
                for s in spans:
                    if s.span_id == span_id:
                        span = s
                        break
                if span:
                    break
        
        if span:
            span.events.append({
                "name": event_name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {}
            })
    
    def add_attribute(self, span_id: str, key: str, value: Any):
        """Agrega un atributo a un span"""
        span = self.active_spans.get(span_id)
        if span:
            span.attributes[key] = value
    
    def get_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """Obtiene un trace completo"""
        spans = self.traces.get(trace_id)
        if not spans:
            return None
        
        return [self._span_to_dict(span) for span in spans]
    
    def _span_to_dict(self, span: Span) -> Dict[str, Any]:
        """Convierte span a diccionario"""
        return {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "kind": span.kind.value,
            "start_time": span.start_time.isoformat(),
            "end_time": span.end_time.isoformat() if span.end_time else None,
            "duration_ms": span.duration_ms,
            "attributes": span.attributes,
            "events": span.events,
            "status": span.status,
            "error": span.error
        }
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene resumen de un trace"""
        spans = self.traces.get(trace_id)
        if not spans:
            return None
        
        total_duration = sum(s.duration_ms or 0 for s in spans)
        root_span = next((s for s in spans if s.parent_span_id is None), None)
        
        return {
            "trace_id": trace_id,
            "name": root_span.name if root_span else "Unknown",
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "root_span": root_span.name if root_span else None,
            "status": "ok" if all(s.status == "ok" for s in spans) else "error"
        }


class TraceContext:
    """Context manager para tracing"""
    
    def __init__(self, tracing: DistributedTracing, trace_id: str, name: str,
                 attributes: Optional[Dict[str, Any]] = None):
        self.tracing = tracing
        self.trace_id = trace_id
        self.name = name
        self.attributes = attributes
        self.span_id: Optional[str] = None
    
    def __enter__(self):
        self.span_id = self.tracing.start_span(
            self.trace_id, self.name, attributes=self.attributes
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "ok"
        error = str(exc_val) if exc_val else None
        self.tracing.end_span(self.span_id, status, error)




