"""
Advanced Observability - Observabilidad Avanzada
================================================

Sistema de observabilidad avanzada con tracing distribuido, métricas detalladas y logging estructurado.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import time

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Span de tracing."""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "started"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    parent_span_id: Optional[str] = None


@dataclass
class ObservabilityMetric:
    """Métrica de observabilidad."""
    metric_name: str
    value: float
    metric_type: str  # counter, gauge, histogram
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedObservability:
    """Sistema de observabilidad avanzada."""
    
    def __init__(self, max_traces: int = 10000, max_spans_per_trace: int = 1000):
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.max_traces = max_traces
        self.max_spans_per_trace = max_spans_per_trace
        self._lock = asyncio.Lock()
    
    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> str:
        """Iniciar un nuevo trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            status="started",
        )
        
        self.active_spans[span.span_id] = span
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        
        # Limitar tamaño
        if len(self.traces) > self.max_traces:
            oldest_trace = min(self.traces.keys())
            del self.traces[oldest_trace]
        
        return span.span_id
    
    def start_span(
        self,
        operation_name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Iniciar un nuevo span."""
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            status="started",
            parent_span_id=parent_span_id,
            tags=tags or {},
        )
        
        self.active_spans[span.span_id] = span
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        spans = self.traces[trace_id]
        if len(spans) < self.max_spans_per_trace:
            spans.append(span)
        
        return span.span_id
    
    def end_span(self, span_id: str, status: str = "success", tags: Optional[Dict[str, Any]] = None):
        """Finalizar un span."""
        span = self.active_spans.get(span_id)
        if not span:
            return
        
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if tags:
            span.tags.update(tags)
        
        del self.active_spans[span_id]
    
    def add_span_log(self, span_id: str, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):
        """Agregar log a un span."""
        span = self.active_spans.get(span_id)
        if not span:
            # Buscar en traces completados
            for trace_spans in self.traces.values():
                for s in trace_spans:
                    if s.span_id == span_id:
                        span = s
                        break
                if span:
                    break
        
        if span:
            span.logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "data": data or {},
            })
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Registrar métrica."""
        metric = ObservabilityMetric(
            metric_name=metric_name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
        )
        
        self.metrics[metric_name].append(metric)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Obtener trace completo."""
        spans = self.traces.get(trace_id)
        if not spans:
            return None
        
        return {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "operation_name": s.operation_name,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "tags": s.tags,
                    "logs": s.logs,
                    "parent_span_id": s.parent_span_id,
                }
                for s in spans
            ],
            "total_spans": len(spans),
            "total_duration_ms": sum(s.duration_ms or 0 for s in spans),
        }
    
    def get_traces(
        self,
        operation_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener traces."""
        traces_list = []
        
        for trace_id, spans in self.traces.items():
            if operation_name:
                if not any(s.operation_name == operation_name for s in spans):
                    continue
            
            total_duration = sum(s.duration_ms or 0 for s in spans)
            
            traces_list.append({
                "trace_id": trace_id,
                "operation_name": spans[0].operation_name if spans else "unknown",
                "total_spans": len(spans),
                "total_duration_ms": total_duration,
                "start_time": min(s.start_time for s in spans).isoformat() if spans else None,
            })
        
        # Ordenar por tiempo más reciente
        traces_list.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        return traces_list[:limit]
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Obtener métricas."""
        if metric_name:
            metrics_list = list(self.metrics.get(metric_name, deque()))[-limit:]
            return {
                metric_name: [
                    {
                        "value": m.value,
                        "metric_type": m.metric_type,
                        "labels": m.labels,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in metrics_list
                ]
            }
        
        result = {}
        for name, metrics_queue in self.metrics.items():
            metrics_list = list(metrics_queue)[-limit:]
            result[name] = [
                {
                    "value": m.value,
                    "metric_type": m.metric_type,
                    "labels": m.labels,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in metrics_list
            ]
        
        return result
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Obtener resumen de observabilidad."""
        total_traces = len(self.traces)
        total_spans = sum(len(spans) for spans in self.traces.values())
        active_spans = len(self.active_spans)
        
        # Métricas por tipo
        metrics_by_type: Dict[str, int] = defaultdict(int)
        for metrics_queue in self.metrics.values():
            for metric in metrics_queue:
                metrics_by_type[metric.metric_type] += 1
        
        return {
            "total_traces": total_traces,
            "total_spans": total_spans,
            "active_spans": active_spans,
            "metrics_count": sum(len(m) for m in self.metrics.values()),
            "metrics_by_type": dict(metrics_by_type),
        }
















