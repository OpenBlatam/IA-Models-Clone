"""
Routing Monitoring and Analytics Optimizations
===============================================

Optimizaciones de monitoreo y analytics avanzados.
Incluye: Metrics collection, Performance analytics, Alerting, etc.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus not available, metrics disabled")


@dataclass
class Metric:
    """Métrica individual."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Colector de métricas."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Inicializar colector de métricas.
        
        Args:
            max_metrics: Máximo de métricas a mantener
        """
        self.max_metrics = max_metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.lock = threading.Lock()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        """
        metric = Metric(name=name, value=value, tags=tags or {})
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: str, window_seconds: Optional[float] = None) -> List[Metric]:
        """
        Obtener métricas.
        
        Args:
            name: Nombre de la métrica
            window_seconds: Ventana de tiempo (None = todas)
        
        Returns:
            Lista de métricas
        """
        with self.lock:
            metrics = list(self.metrics[name])
        
        if window_seconds:
            current_time = time.time()
            metrics = [m for m in metrics if (current_time - m.timestamp) <= window_seconds]
        
        return metrics
    
    def get_statistics(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Obtener estadísticas de una métrica."""
        metrics = self.get_metrics(name, window_seconds)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'sum': sum(values)
        }


class PerformanceAnalytics:
    """Analytics de rendimiento."""
    
    def __init__(self):
        """Inicializar analytics."""
        self.metrics_collector = MetricsCollector()
        self.prometheus_metrics = {}
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Inicializar métricas de Prometheus."""
        self.prometheus_metrics = {
            'route_requests_total': Counter('route_requests_total', 'Total route requests'),
            'route_duration_seconds': Histogram('route_duration_seconds', 'Route calculation duration'),
            'cache_hits_total': Counter('cache_hits_total', 'Total cache hits'),
            'cache_misses_total': Counter('cache_misses_total', 'Total cache misses'),
            'active_routes': Gauge('active_routes', 'Number of active routes')
        }
    
    def record_route_request(self, duration: float, cached: bool = False):
        """Registrar request de ruta."""
        self.metrics_collector.record('route_requests', 1.0)
        self.metrics_collector.record('route_duration', duration)
        
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['route_requests_total'].inc()
            self.prometheus_metrics['route_duration_seconds'].observe(duration)
            
            if cached:
                self.prometheus_metrics['cache_hits_total'].inc()
            else:
                self.prometheus_metrics['cache_misses_total'].inc()
    
    def record_active_routes(self, count: int):
        """Registrar número de rutas activas."""
        self.metrics_collector.record('active_routes', float(count))
        
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['active_routes'].set(count)
    
    def get_analytics(self, window_seconds: float = 3600) -> Dict[str, Any]:
        """Obtener analytics."""
        return {
            'route_requests': self.metrics_collector.get_statistics('route_requests', window_seconds),
            'route_duration': self.metrics_collector.get_statistics('route_duration', window_seconds),
            'active_routes': self.metrics_collector.get_statistics('active_routes', window_seconds)
        }


class AlertManager:
    """Gestor de alertas."""
    
    def __init__(self):
        """Inicializar gestor de alertas."""
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Callable] = []
        self.lock = threading.Lock()
    
    def add_alert_rule(self, rule: Callable):
        """
        Agregar regla de alerta.
        
        Args:
            rule: Función que retorna (should_alert, message)
        """
        self.alert_rules.append(rule)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """
        Verificar alertas.
        
        Args:
            metrics: Métricas actuales
        """
        with self.lock:
            for rule in self.alert_rules:
                try:
                    should_alert, message = rule(metrics)
                    if should_alert:
                        alert = {
                            'timestamp': time.time(),
                            'message': message,
                            'metrics': metrics
                        }
                        self.alerts.append(alert)
                        logger.warning(f"ALERT: {message}")
                except Exception as e:
                    logger.debug(f"Alert rule error: {e}")
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener alertas recientes."""
        with self.lock:
            return self.alerts[-limit:]


class MonitoringOptimizer:
    """Optimizador completo de monitoreo."""
    
    def __init__(self, enable_prometheus: bool = True):
        """
        Inicializar optimizador de monitoreo.
        
        Args:
            enable_prometheus: Habilitar métricas de Prometheus
        """
        self.analytics = PerformanceAnalytics()
        self.alert_manager = AlertManager()
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Agregar reglas de alerta por defecto
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Configurar alertas por defecto."""
        def high_latency_alert(metrics):
            route_duration = metrics.get('route_duration', {})
            mean_duration = route_duration.get('mean', 0)
            if mean_duration > 1.0:  # Más de 1 segundo
                return True, f"High route calculation latency: {mean_duration:.2f}s"
            return False, None
        
        def low_cache_hit_rate(metrics):
            # Esta regla se puede mejorar con métricas de cache
            return False, None
        
        self.alert_manager.add_alert_rule(high_latency_alert)
        self.alert_manager.add_alert_rule(low_cache_hit_rate)
    
    def record_route_metrics(self, duration: float, cached: bool = False):
        """Registrar métricas de ruta."""
        self.analytics.record_route_request(duration, cached)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        analytics = self.analytics.get_analytics()
        alerts = self.alert_manager.get_alerts()
        
        return {
            'analytics': analytics,
            'recent_alerts': len(alerts),
            'prometheus_enabled': self.enable_prometheus
        }

