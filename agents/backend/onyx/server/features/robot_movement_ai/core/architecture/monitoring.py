"""
Sistema de monitoreo y métricas para Robot Movement AI v2.0
Integración con Prometheus para observabilidad completa
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Stubs para cuando Prometheus no está disponible
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
    def generate_latest():
        return b""

@dataclass
class MetricsCollector:
    """Recolector de métricas para la aplicación"""
    
    # Contadores
    robot_movements_total: Counter = field(default=None)
    robot_commands_total: Counter = field(default=None)
    robot_errors_total: Counter = field(default=None)
    api_requests_total: Counter = field(default=None)
    
    # Histogramas (distribuciones)
    movement_duration: Histogram = field(default=None)
    command_duration: Histogram = field(default=None)
    api_request_duration: Histogram = field(default=None)
    
    # Gauges (valores actuales)
    active_robots: Gauge = field(default=None)
    active_connections: Gauge = field(default=None)
    circuit_breaker_state: Gauge = field(default=None)
    
    # Métricas internas (fallback si Prometheus no está disponible)
    _internal_metrics: Dict = field(default_factory=lambda: defaultdict(float))
    
    def __post_init__(self):
        """Inicializar métricas de Prometheus si está disponible"""
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            self._init_internal_metrics()
    
    def _init_prometheus_metrics(self):
        """Inicializar métricas de Prometheus"""
        self.robot_movements_total = Counter(
            'robot_movements_total',
            'Total number of robot movements',
            ['robot_id', 'status']
        )
        
        self.robot_commands_total = Counter(
            'robot_commands_total',
            'Total number of robot commands',
            ['command_type', 'status']
        )
        
        self.robot_errors_total = Counter(
            'robot_errors_total',
            'Total number of robot errors',
            ['error_type', 'robot_id']
        )
        
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.movement_duration = Histogram(
            'movement_duration_seconds',
            'Duration of robot movements',
            ['robot_id'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.command_duration = Histogram(
            'command_duration_seconds',
            'Duration of robot commands',
            ['command_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'Duration of API requests',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )
        
        self.active_robots = Gauge(
            'active_robots',
            'Number of active robots',
            ['status']
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service_name']
        )
    
    def _init_internal_metrics(self):
        """Inicializar métricas internas como fallback"""
        self._internal_metrics = defaultdict(float)
    
    def record_movement(self, robot_id: str, duration: float, status: str = "success"):
        """Registrar un movimiento de robot"""
        if self.robot_movements_total:
            self.robot_movements_total.labels(robot_id=robot_id, status=status).inc()
        if self.movement_duration:
            self.movement_duration.labels(robot_id=robot_id).observe(duration)
        
        # Fallback interno
        key = f"movements.{robot_id}.{status}"
        self._internal_metrics[key] += 1
    
    def record_command(self, command_type: str, duration: float, status: str = "success"):
        """Registrar un comando de robot"""
        if self.robot_commands_total:
            self.robot_commands_total.labels(command_type=command_type, status=status).inc()
        if self.command_duration:
            self.command_duration.labels(command_type=command_type).observe(duration)
        
        # Fallback interno
        key = f"commands.{command_type}.{status}"
        self._internal_metrics[key] += 1
    
    def record_error(self, error_type: str, robot_id: Optional[str] = None):
        """Registrar un error"""
        if self.robot_errors_total:
            self.robot_errors_total.labels(
                error_type=error_type,
                robot_id=robot_id or "unknown"
            ).inc()
        
        # Fallback interno
        key = f"errors.{error_type}"
        self._internal_metrics[key] += 1
    
    def record_api_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """Registrar una petición API"""
        if self.api_requests_total:
            self.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
        if self.api_request_duration:
            self.api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        # Fallback interno
        key = f"api.{method}.{endpoint}.{status_code}"
        self._internal_metrics[key] += 1
    
    def set_active_robots(self, count: int, status: str = "connected"):
        """Establecer número de robots activos"""
        if self.active_robots:
            self.active_robots.labels(status=status).set(count)
        
        # Fallback interno
        self._internal_metrics[f"active_robots.{status}"] = count
    
    def set_active_connections(self, count: int):
        """Establecer número de conexiones activas"""
        if self.active_connections:
            self.active_connections.set(count)
        
        # Fallback interno
        self._internal_metrics["active_connections"] = count
    
    def set_circuit_breaker_state(self, service_name: str, state: int):
        """Establecer estado del circuit breaker (0=closed, 1=open, 2=half-open)"""
        if self.circuit_breaker_state:
            self.circuit_breaker_state.labels(service_name=service_name).set(state)
        
        # Fallback interno
        self._internal_metrics[f"circuit_breaker.{service_name}"] = state
    
    def get_metrics(self) -> bytes:
        """Obtener métricas en formato Prometheus"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY)
        else:
            # Retornar métricas internas como texto simple
            lines = []
            for key, value in self._internal_metrics.items():
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {value}")
            return "\n".join(lines).encode()
    
    def get_metrics_dict(self) -> Dict:
        """Obtener métricas como diccionario"""
        return dict(self._internal_metrics)


# Instancia global del recolector de métricas
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtener instancia global del recolector de métricas"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_movement(robot_id: str, duration: float, status: str = "success"):
    """Helper para registrar movimiento"""
    get_metrics_collector().record_movement(robot_id, duration, status)


def record_command(command_type: str, duration: float, status: str = "success"):
    """Helper para registrar comando"""
    get_metrics_collector().record_command(command_type, duration, status)


def record_error(error_type: str, robot_id: Optional[str] = None):
    """Helper para registrar error"""
    get_metrics_collector().record_error(error_type, robot_id)


def record_api_request(method: str, endpoint: str, duration: float, status_code: int):
    """Helper para registrar petición API"""
    get_metrics_collector().record_api_request(method, endpoint, duration, status_code)




