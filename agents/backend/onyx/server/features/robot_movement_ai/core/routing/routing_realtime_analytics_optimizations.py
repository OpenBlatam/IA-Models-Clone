"""
Routing Real-Time Analytics Optimizations
==========================================

Optimizaciones para analytics en tiempo real.
Incluye: Stream processing, Real-time metrics, Event processing, etc.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Procesador de streams."""
    
    def __init__(self, window_size: int = 1000):
        """
        Inicializar procesador.
        
        Args:
            window_size: Tamaño de ventana deslizante
        """
        self.window_size = window_size
        self.stream: deque = deque(maxlen=window_size)
        self.processors: List[Callable] = []
        self.lock = threading.Lock()
    
    def add_event(self, event: Dict[str, Any]):
        """Agregar evento al stream."""
        with self.lock:
            event['timestamp'] = time.time()
            self.stream.append(event)
            
            # Procesar con todos los procesadores
            for processor in self.processors:
                try:
                    processor(event)
                except Exception as e:
                    logger.error(f"Error in stream processor: {e}")
    
    def register_processor(self, processor: Callable):
        """Registrar procesador."""
        with self.lock:
            self.processors.append(processor)
    
    def get_window(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Obtener ventana de eventos."""
        with self.lock:
            window_size = size or self.window_size
            return list(self.stream)[-window_size:]


class RealTimeMetrics:
    """Métricas en tiempo real."""
    
    def __init__(self):
        """Inicializar métricas."""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregations: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()
    
    def record(self, metric_name: str, value: float):
        """Registrar métrica."""
        with self.lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': time.time()
            })
            self._update_aggregations(metric_name)
    
    def _update_aggregations(self, metric_name: str):
        """Actualizar agregaciones."""
        values = [m['value'] for m in self.metrics[metric_name]]
        if values:
            self.aggregations[metric_name] = {
                'count': len(values),
                'sum': sum(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'last': values[-1]
            }
    
    def get_metric(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Obtener métrica."""
        with self.lock:
            return self.aggregations.get(metric_name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Obtener todas las métricas."""
        with self.lock:
            return dict(self.aggregations)


class EventProcessor:
    """Procesador de eventos."""
    
    def __init__(self):
        """Inicializar procesador."""
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        self.lock = threading.Lock()
    
    def register_handler(self, event_type: str, handler: Callable):
        """Registrar handler de evento."""
        with self.lock:
            self.event_handlers[event_type].append(handler)
    
    def process_event(self, event: Dict[str, Any]):
        """Procesar evento."""
        event_type = event.get('type', 'unknown')
        
        with self.lock:
            self.event_history.append({
                **event,
                'timestamp': time.time()
            })
        
        # Ejecutar handlers
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de eventos."""
        with self.lock:
            event_counts = defaultdict(int)
            for event in self.event_history:
                event_type = event.get('type', 'unknown')
                event_counts[event_type] += 1
            
            return {
                'total_events': len(self.event_history),
                'event_counts': dict(event_counts),
                'num_handlers': sum(len(handlers) for handlers in self.event_handlers.values())
            }


class RealTimeAnalyticsOptimizer:
    """Optimizador completo de analytics en tiempo real."""
    
    def __init__(self):
        """Inicializar optimizador."""
        self.stream_processor = StreamProcessor()
        self.real_time_metrics = RealTimeMetrics()
        self.event_processor = EventProcessor()
    
    def process_stream_event(self, event: Dict[str, Any]):
        """Procesar evento de stream."""
        self.stream_processor.add_event(event)
        self.event_processor.process_event(event)
    
    def record_metric(self, metric_name: str, value: float):
        """Registrar métrica."""
        self.real_time_metrics.record(metric_name, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'stream_stats': {
                'window_size': self.stream_processor.window_size,
                'events_in_window': len(self.stream_processor.stream),
                'num_processors': len(self.stream_processor.processors)
            },
            'metrics': self.real_time_metrics.get_all_metrics(),
            'event_stats': self.event_processor.get_event_stats()
        }

