"""
Routing Observability Optimizations
====================================

Optimizaciones de observabilidad avanzada.
Incluye: Distributed tracing, OpenTelemetry, Log aggregation, etc.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from collections import deque
import threading

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available, distributed tracing disabled")


class TraceContext:
    """Contexto de traza distribuida."""
    
    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        """
        Inicializar contexto de traza.
        
        Args:
            trace_id: ID de traza
            span_id: ID de span
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.parent_span_id: Optional[str] = None
        self.start_time = time.time()
        self.tags: Dict[str, Any] = {}
    
    def add_tag(self, key: str, value: Any):
        """Agregar tag a la traza."""
        self.tags[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'start_time': self.start_time,
            'tags': self.tags
        }


class DistributedTracer:
    """Tracer distribuido."""
    
    def __init__(self):
        """Inicializar tracer."""
        self.traces: Dict[str, TraceContext] = {}
        self.spans: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        if OPENTELEMETRY_AVAILABLE:
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None
    
    def start_trace(self, operation: str, parent_trace: Optional[TraceContext] = None) -> TraceContext:
        """
        Iniciar traza.
        
        Args:
            operation: Nombre de la operación
            parent_trace: Traza padre (opcional)
        
        Returns:
            Contexto de traza
        """
        trace_context = TraceContext()
        
        if parent_trace:
            trace_context.trace_id = parent_trace.trace_id
            trace_context.parent_span_id = parent_trace.span_id
        
        trace_context.add_tag('operation', operation)
        
        with self.lock:
            self.traces[trace_context.trace_id] = trace_context
        
        if self.tracer:
            span = self.tracer.start_span(operation)
            trace_context.span_id = format(span.get_span_context().span_id, '016x')
        
        return trace_context
    
    def end_trace(self, trace_context: TraceContext, status: str = "success"):
        """Finalizar traza."""
        duration = time.time() - trace_context.start_time
        
        with self.lock:
            span_data = {
                'trace_id': trace_context.trace_id,
                'span_id': trace_context.span_id,
                'operation': trace_context.tags.get('operation'),
                'duration': duration,
                'status': status,
                'tags': trace_context.tags,
                'timestamp': time.time()
            }
            self.spans.append(span_data)
            
            # Mantener solo últimas 1000 spans
            if len(self.spans) > 1000:
                self.spans = self.spans[-1000:]
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Obtener traza por ID."""
        with self.lock:
            return self.traces.get(trace_id)
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de trazas."""
        with self.lock:
            if not self.spans:
                return {}
            
            durations = [s['duration'] for s in self.spans]
            operations = {}
            
            for span in self.spans:
                op = span.get('operation', 'unknown')
                if op not in operations:
                    operations[op] = []
                operations[op].append(span['duration'])
            
            op_stats = {}
            for op, op_durations in operations.items():
                op_stats[op] = {
                    'count': len(op_durations),
                    'avg_duration': sum(op_durations) / len(op_durations),
                    'min_duration': min(op_durations),
                    'max_duration': max(op_durations)
                }
            
            return {
                'total_traces': len(self.traces),
                'total_spans': len(self.spans),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'operations': op_stats
            }


class LogAggregator:
    """Agregador de logs."""
    
    def __init__(self, max_logs: int = 10000):
        """
        Inicializar agregador.
        
        Args:
            max_logs: Máximo de logs a mantener
        """
        self.logs: deque = deque(maxlen=max_logs)
        self.lock = threading.Lock()
    
    def add_log(self, level: str, message: str, **kwargs):
        """Agregar log."""
        log_entry = {
            'level': level,
            'message': message,
            'timestamp': time.time(),
            **kwargs
        }
        
        with self.lock:
            self.logs.append(log_entry)
    
    def get_logs(
        self,
        level: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Obtener logs filtrados."""
        with self.lock:
            logs = list(self.logs)
        
        if level:
            logs = [l for l in logs if l['level'] == level]
        
        if start_time:
            logs = [l for l in logs if l['timestamp'] >= start_time]
        
        if end_time:
            logs = [l for l in logs if l['timestamp'] <= end_time]
        
        return logs
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de logs."""
        with self.lock:
            if not self.logs:
                return {}
            
            level_counts = {}
            for log in self.logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            return {
                'total_logs': len(self.logs),
                'level_counts': level_counts
            }


class ObservabilityOptimizer:
    """Optimizador completo de observabilidad."""
    
    def __init__(self):
        """Inicializar optimizador de observabilidad."""
        self.tracer = DistributedTracer()
        self.log_aggregator = LogAggregator()
    
    def trace_operation(self, operation: str, func: Any, *args, **kwargs):
        """
        Trazar operación.
        
        Args:
            operation: Nombre de la operación
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Resultado de la función
        """
        trace_context = self.tracer.start_trace(operation)
        
        try:
            result = func(*args, **kwargs)
            self.tracer.end_trace(trace_context, "success")
            return result
        except Exception as e:
            trace_context.add_tag('error', str(e))
            self.tracer.end_trace(trace_context, "error")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'tracing_stats': self.tracer.get_trace_statistics(),
            'log_stats': self.log_aggregator.get_log_statistics(),
            'opentelemetry_available': OPENTELEMETRY_AVAILABLE
        }

