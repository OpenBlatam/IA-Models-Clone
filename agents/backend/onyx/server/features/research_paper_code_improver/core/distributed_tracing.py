"""
Distributed Tracing - Trazabilidad distribuida de requests
===========================================================
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Tipos de span"""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Estados de span"""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class Span:
    """Span individual"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links
        }
    
    def end(self, status: SpanStatus = SpanStatus.OK):
        """Finaliza el span"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status


@dataclass
class Trace:
    """Trace completo"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    root_span: Optional[Span] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.spans],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "root_span": self.root_span.to_dict() if self.root_span else None
        }


class DistributedTracing:
    """Sistema de trazabilidad distribuida"""
    
    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}  # span_id -> span
        self.trace_context: Dict[str, str] = {}  # request_id -> trace_id
    
    def start_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Inicia un nuevo trace"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            attributes=attributes or {}
        )
        
        # Crear trace si no existe
        if trace_id not in self.traces:
            trace = Trace(trace_id=trace_id)
            trace.root_span = span
            self.traces[trace_id] = trace
        
        trace = self.traces[trace_id]
        trace.spans.append(span)
        self.active_spans[span_id] = span
        
        return span
    
    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Inicia un nuevo span"""
        if trace_id is None and parent_span_id:
            parent_span = self.active_spans.get(parent_span_id)
            if parent_span:
                trace_id = parent_span.trace_id
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            attributes=attributes or {}
        )
        
        # Agregar al trace
        if trace_id not in self.traces:
            self.traces[trace_id] = Trace(trace_id=trace_id)
        
        trace = self.traces[trace_id]
        trace.spans.append(span)
        self.active_spans[span_id] = span
        
        return span
    
    def end_span(self, span_id: str, status: SpanStatus = SpanStatus.OK):
        """Finaliza un span"""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.end(status)
        
        # Remover de activos
        del self.active_spans[span_id]
        
        # Actualizar trace end_time si es el último span
        trace = self.traces.get(span.trace_id)
        if trace:
            active_spans_in_trace = [
                s for s in trace.spans
                if s.span_id in self.active_spans
            ]
            if not active_spans_in_trace:
                trace.end_time = datetime.now()
    
    def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Agrega un evento a un span"""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def set_attribute(self, span_id: str, key: str, value: Any):
        """Establece un atributo en un span"""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.attributes[key] = value
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Obtiene un trace completo"""
        return self.traces.get(trace_id)
    
    def get_trace_tree(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el árbol de spans de un trace"""
        trace = self.get_trace(trace_id)
        if not trace:
            return None
        
        # Construir árbol
        spans_by_id = {span.span_id: span for span in trace.spans}
        root_spans = [span for span in trace.spans if not span.parent_span_id]
        
        def build_tree(span: Span) -> Dict[str, Any]:
            children = [
                build_tree(spans_by_id[child.span_id])
                for child in trace.spans
                if child.parent_span_id == span.span_id
            ]
            
            return {
                "span": span.to_dict(),
                "children": children
            }
        
        if root_spans:
            return build_tree(root_spans[0])
        
        return None
    
    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        status: Optional[SpanStatus] = None,
        limit: int = 100
    ) -> List[Trace]:
        """Busca traces con filtros"""
        results = []
        
        for trace in self.traces.values():
            # Filtrar por duración
            if trace.end_time and trace.start_time:
                duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
                if min_duration_ms and duration_ms < min_duration_ms:
                    continue
                if max_duration_ms and duration_ms > max_duration_ms:
                    continue
            
            # Filtrar por spans
            matching_spans = trace.spans
            if service_name:
                matching_spans = [
                    s for s in matching_spans
                    if s.attributes.get("service.name") == service_name
                ]
            if operation_name:
                matching_spans = [
                    s for s in matching_spans
                    if s.name == operation_name
                ]
            if status:
                matching_spans = [
                    s for s in matching_spans
                    if s.status == status
                ]
            
            if matching_spans:
                results.append(trace)
        
        return results[:limit]




