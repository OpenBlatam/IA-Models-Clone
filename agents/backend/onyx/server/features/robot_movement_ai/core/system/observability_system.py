"""
Observability System
====================

Sistema de observabilidad para monitoreo completo.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Nivel de observabilidad."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Trace:
    """Trace."""
    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Span:
    """Span."""
    span_id: str
    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservabilitySystem:
    """
    Sistema de observabilidad.
    
    Gestiona traces, spans y métricas de observabilidad.
    """
    
    def __init__(self):
        """Inicializar sistema de observabilidad."""
        self.traces: Dict[str, Trace] = {}
        self.spans: Dict[str, Span] = {}
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.max_traces = 10000
        self.max_spans = 50000
    
    def start_trace(
        self,
        trace_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Iniciar trace.
        
        Args:
            trace_id: ID único del trace
            operation: Nombre de la operación
            metadata: Metadata adicional
            
        Returns:
            Trace iniciado
        """
        trace = Trace(
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            status="started",
            metadata=metadata or {}
        )
        
        self.traces[trace_id] = trace
        
        # Limitar tamaño
        if len(self.traces) > self.max_traces:
            oldest_trace = min(self.traces.values(), key=lambda t: t.start_time)
            del self.traces[oldest_trace.trace_id]
        
        logger.debug(f"Started trace: {operation} ({trace_id})")
        
        return trace
    
    def end_trace(
        self,
        trace_id: str,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Trace]:
        """
        Finalizar trace.
        
        Args:
            trace_id: ID del trace
            status: Estado final
            metadata: Metadata adicional
            
        Returns:
            Trace finalizado o None
        """
        if trace_id not in self.traces:
            return None
        
        trace = self.traces[trace_id]
        trace.end_time = time.time()
        trace.duration = trace.end_time - trace.start_time
        trace.status = status
        
        if metadata:
            trace.metadata.update(metadata)
        
        logger.debug(f"Ended trace: {trace.operation} ({trace_id}) - {trace.duration:.3f}s")
        
        return trace
    
    def start_span(
        self,
        span_id: str,
        trace_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Iniciar span.
        
        Args:
            span_id: ID único del span
            trace_id: ID del trace padre
            operation: Nombre de la operación
            metadata: Metadata adicional
            
        Returns:
            Span iniciado
        """
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.spans[span_id] = span
        
        # Agregar al trace
        if trace_id in self.traces:
            self.traces[trace_id].spans.append({
                "span_id": span_id,
                "operation": operation,
                "start_time": span.start_time
            })
        
        # Limitar tamaño
        if len(self.spans) > self.max_spans:
            oldest_span = min(self.spans.values(), key=lambda s: s.start_time)
            del self.spans[oldest_span.span_id]
        
        return span
    
    def end_span(
        self,
        span_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        """
        Finalizar span.
        
        Args:
            span_id: ID del span
            metadata: Metadata adicional
            
        Returns:
            Span finalizado o None
        """
        if span_id not in self.spans:
            return None
        
        span = self.spans[span_id]
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        
        if metadata:
            span.metadata.update(metadata)
        
        return span
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Registrar métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            labels: Labels adicionales
        """
        key = metric_name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{metric_name}[{label_str}]"
        
        self.metrics[key].append(value)
        
        # Limitar tamaño
        if len(self.metrics[key]) > 1000:
            self.metrics[key] = self.metrics[key][-1000:]
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Obtener trace por ID."""
        return self.traces.get(trace_id)
    
    def get_traces(
        self,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[Trace]:
        """
        Obtener traces.
        
        Args:
            operation: Filtrar por operación
            limit: Límite de resultados
            
        Returns:
            Lista de traces
        """
        traces = list(self.traces.values())
        
        if operation:
            traces = [t for t in traces if t.operation == operation]
        
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de observabilidad."""
        completed_traces = [t for t in self.traces.values() if t.duration is not None]
        
        if completed_traces:
            avg_duration = sum(t.duration for t in completed_traces) / len(completed_traces)
            max_duration = max(t.duration for t in completed_traces)
        else:
            avg_duration = 0.0
            max_duration = 0.0
        
        return {
            "total_traces": len(self.traces),
            "completed_traces": len(completed_traces),
            "total_spans": len(self.spans),
            "total_metrics": len(self.metrics),
            "average_trace_duration": avg_duration,
            "max_trace_duration": max_duration
        }


# Instancia global
_observability_system: Optional[ObservabilitySystem] = None


def get_observability_system() -> ObservabilitySystem:
    """Obtener instancia global del sistema de observabilidad."""
    global _observability_system
    if _observability_system is None:
        _observability_system = ObservabilitySystem()
    return _observability_system


# Importar defaultdict
from collections import defaultdict






