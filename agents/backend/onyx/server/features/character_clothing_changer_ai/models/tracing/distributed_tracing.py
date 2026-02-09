"""
Distributed Tracing System
===========================
Sistema de tracing distribuido para microservicios
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class SpanKind(Enum):
    """Tipos de span"""
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Estados de span"""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class Span:
    """Span de tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = None
    events: List[Dict[str, Any]] = None
    links: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []
        if self.links is None:
            self.links = []
    
    @property
    def duration(self) -> Optional[float]:
        """Duración del span"""
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class Trace:
    """Trace completo"""
    trace_id: str
    spans: List[Span]
    start_time: float
    end_time: Optional[float] = None
    service_name: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Duración del trace"""
        if self.end_time:
            return self.end_time - self.start_time
        return None


class DistributedTracing:
    """
    Sistema de tracing distribuido
    """
    
    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        self.trace_context: Dict[str, str] = {}  # request_id -> trace_id
    
    def start_trace(
        self,
        name: str,
        service_name: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> str:
        """
        Iniciar trace
        
        Args:
            name: Nombre del trace
            service_name: Nombre del servicio
            trace_id: ID del trace (opcional, se genera si no se proporciona)
        
        Returns:
            trace_id
        """
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        trace = Trace(
            trace_id=trace_id,
            spans=[],
            start_time=time.time(),
            service_name=service_name
        )
        
        self.traces[trace_id] = trace
        return trace_id
    
    def start_span(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Iniciar span
        
        Args:
            trace_id: ID del trace
            name: Nombre del span
            kind: Tipo de span
            parent_span_id: ID del span padre
            attributes: Atributos del span
        
        Returns:
            span_id
        """
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        if trace_id in self.traces:
            self.traces[trace_id].spans.append(span)
        
        self.active_spans[span_id] = span
        return span_id
    
    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Finalizar span
        
        Args:
            span_id: ID del span
            status: Estado final
            attributes: Atributos adicionales
        """
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end_time = time.time()
            span.status = status
            
            if attributes:
                span.attributes.update(attributes)
            
            del self.active_spans[span_id]
    
    def add_span_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Agregar evento a span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.events.append({
                'name': name,
                'timestamp': time.time(),
                'attributes': attributes or {}
            })
    
    def add_span_attribute(
        self,
        span_id: str,
        key: str,
        value: Any
    ):
        """Agregar atributo a span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].attributes[key] = value
    
    def end_trace(self, trace_id: str):
        """Finalizar trace"""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace.end_time = time.time()
            
            # Finalizar spans activos
            active_span_ids = [
                span_id for span_id, span in self.active_spans.items()
                if span.trace_id == trace_id
            ]
            for span_id in active_span_ids:
                self.end_span(span_id, SpanStatus.UNSET)
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Obtener trace"""
        return self.traces.get(trace_id)
    
    def get_trace_tree(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Obtener árbol de spans del trace"""
        trace = self.get_trace(trace_id)
        if not trace:
            return None
        
        # Construir árbol
        span_map = {span.span_id: span for span in trace.spans}
        root_spans = [span for span in trace.spans if not span.parent_span_id]
        
        def build_tree(span: Span) -> Dict[str, Any]:
            children = [
                build_tree(child_span)
                for child_span in trace.spans
                if child_span.parent_span_id == span.span_id
            ]
            
            return {
                'span_id': span.span_id,
                'name': span.name,
                'kind': span.kind.value,
                'duration': span.duration,
                'status': span.status.value,
                'attributes': span.attributes,
                'children': children
            }
        
        if root_spans:
            return {
                'trace_id': trace_id,
                'service_name': trace.service_name,
                'duration': trace.duration,
                'root_spans': [build_tree(span) for span in root_spans]
            }
        
        return None
    
    def search_traces(
        self,
        service_name: Optional[str] = None,
        span_name: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        status: Optional[SpanStatus] = None
    ) -> List[Trace]:
        """Buscar traces"""
        results = []
        
        for trace in self.traces.values():
            if service_name and trace.service_name != service_name:
                continue
            
            if span_name:
                if not any(span.name == span_name for span in trace.spans):
                    continue
            
            if min_duration and trace.duration and trace.duration < min_duration:
                continue
            
            if max_duration and trace.duration and trace.duration > max_duration:
                continue
            
            if status:
                if not any(span.status == status for span in trace.spans):
                    continue
            
            results.append(trace)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de tracing"""
        total_spans = sum(len(trace.spans) for trace in self.traces.values())
        total_duration = sum(
            trace.duration or 0
            for trace in self.traces.values()
            if trace.duration
        )
        avg_duration = (
            total_duration / len(self.traces)
            if self.traces else 0
        )
        
        status_counts = {}
        for trace in self.traces.values():
            for span in trace.spans:
                status = span.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_traces': len(self.traces),
            'total_spans': total_spans,
            'active_spans': len(self.active_spans),
            'average_trace_duration': avg_duration,
            'status_counts': status_counts
        }


# Instancia global
distributed_tracing = DistributedTracing()

