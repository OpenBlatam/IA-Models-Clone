"""
Metrics Service - Servicio de métricas y observabilidad.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from config.logging_config import get_logger

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("Prometheus no disponible, usando métricas en memoria")

logger = get_logger(__name__)


class MetricsService:
    """
    Servicio de métricas y observabilidad con mejoras.
    
    Attributes:
        use_prometheus: Si está usando Prometheus
        metrics: Diccionario de métricas en memoria
        timers: Diccionario de timers activos
    """

    def __init__(self, use_prometheus: bool = True):
        """
        Inicializar servicio de métricas con validaciones.

        Args:
            use_prometheus: Si usar Prometheus (requiere prometheus-client)
            
        Raises:
            ValueError: Si hay error al inicializar Prometheus
        """
        if not isinstance(use_prometheus, bool):
            raise ValueError(f"use_prometheus debe ser un booleano, recibido: {type(use_prometheus)}")
        
        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        self.metrics: Dict[str, Any] = defaultdict(dict)
        self.timers: Dict[str, float] = {}

        try:
            if self.use_prometheus:
                self._init_prometheus_metrics()
                logger.info("✅ MetricsService inicializado con Prometheus")
            else:
                logger.info("📊 MetricsService inicializado con métricas en memoria (Prometheus no disponible)")
        except Exception as e:
            logger.error(f"Error al inicializar métricas de Prometheus: {e}", exc_info=True)
            self.use_prometheus = False
            logger.warning("Fallando a métricas en memoria debido a error en Prometheus")

    def _init_prometheus_metrics(self) -> None:
        """Inicializar métricas de Prometheus."""
        self.task_counter = Counter(
            "github_agent_tasks_total",
            "Total de tareas procesadas",
            ["status", "type"]
        )
        self.task_duration = Histogram(
            "github_agent_task_duration_seconds",
            "Duración de procesamiento de tareas",
            ["type"]
        )
        self.api_requests = Counter(
            "github_agent_api_requests_total",
            "Total de requests a GitHub API",
            ["method", "status"]
        )
        self.api_duration = Histogram(
            "github_agent_api_duration_seconds",
            "Duración de requests a GitHub API",
            ["method"]
        )
        self.cache_operations = Counter(
            "github_agent_cache_operations_total",
            "Operaciones de caché",
            ["operation", "result"]
        )
        self.active_tasks = Gauge(
            "github_agent_active_tasks",
            "Tareas activas actualmente"
        )
        self.error_counter = Counter(
            "github_agent_errors_total",
            "Total de errores",
            ["error_type"]
        )

    def record_task(
        self,
        task_type: str,
        status: str,
        duration: Optional[float] = None
    ) -> None:
        """
        Registrar una tarea procesada con validaciones.

        Args:
            task_type: Tipo de tarea (create_file, update_file, etc.) - debe ser string no vacío
            status: Estado (completed, failed, etc.) - debe ser string no vacío
            duration: Duración en segundos (opcional, debe ser >= 0 si se proporciona)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not task_type or not isinstance(task_type, str) or not task_type.strip():
            raise ValueError(f"task_type debe ser un string no vacío, recibido: {task_type}")
        
        if not status or not isinstance(status, str) or not status.strip():
            raise ValueError(f"status debe ser un string no vacío, recibido: {status}")
        
        if duration is not None:
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValueError(f"duration debe ser un número no negativo, recibido: {duration}")
        
        task_type = task_type.strip()
        status = status.strip()
        
        try:
            if self.use_prometheus:
                self.task_counter.labels(status=status, type=task_type).inc()
                if duration is not None:
                    self.task_duration.labels(type=task_type).observe(duration)
            else:
                key = f"tasks.{task_type}.{status}"
                self.metrics[key] = self.metrics.get(key, 0) + 1
                if duration is not None:
                    duration_key = f"tasks.{task_type}.duration"
                    if duration_key not in self.metrics:
                        self.metrics[duration_key] = []
                    self.metrics[duration_key].append(duration)
            
            logger.debug(f"Métrica de tarea registrada: {task_type} - {status} (duration: {duration}s)")
        except Exception as e:
            logger.error(f"Error al registrar métrica de tarea: {e}", exc_info=True)
            raise

    def record_api_request(
        self,
        method: str,
        status: str,
        duration: Optional[float] = None
    ) -> None:
        """
        Registrar un request a la API de GitHub.

        Args:
            method: Método HTTP o nombre de operación
            status: Estado (success, error, etc.)
            duration: Duración en segundos (opcional)
        """
        if self.use_prometheus:
            self.api_requests.labels(method=method, status=status).inc()
            if duration is not None:
                self.api_duration.labels(method=method).observe(duration)
        else:
            key = f"api.{method}.{status}"
            self.metrics[key] = self.metrics.get(key, 0) + 1
            if duration is not None:
                duration_key = f"api.{method}.duration"
                if duration_key not in self.metrics:
                    self.metrics[duration_key] = []
                self.metrics[duration_key].append(duration)

    def record_error(self, error_type: str) -> None:
        """
        Registrar un error.

        Args:
            error_type: Tipo de error
        """
        if self.use_prometheus:
            self.error_counter.labels(error_type=error_type).inc()
        else:
            key = f"errors.{error_type}"
            self.metrics[key] = self.metrics.get(key, 0) + 1

    def record_cache_operation(
        self,
        operation: str,
        result: str
    ) -> None:
        """
        Registrar una operación de caché.

        Args:
            operation: Operación (get, set, delete)
            result: Resultado (hit, miss, success, error)
        """
        if self.use_prometheus:
            self.cache_operations.labels(
                operation=operation,
                result=result
            ).inc()
        else:
            key = f"cache.{operation}.{result}"
            self.metrics[key] = self.metrics.get(key, 0) + 1

    def start_timer(self, name: str) -> None:
        """
        Iniciar un timer con validaciones.

        Args:
            name: Nombre del timer (debe ser string no vacío)
            
        Raises:
            ValueError: Si name es inválido o el timer ya existe
        """
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        name = name.strip()
        
        if name in self.timers:
            logger.warning(f"Timer '{name}' ya existe. Sobrescribiendo...")
        
        self.timers[name] = time.time()
        logger.debug(f"Timer iniciado: {name}")

    def stop_timer(self, name: str) -> Optional[float]:
        """
        Detener un timer y obtener la duración con validaciones.

        Args:
            name: Nombre del timer (debe ser string no vacío)

        Returns:
            Duración en segundos o None si el timer no existe
            
        Raises:
            ValueError: Si name es inválido
        """
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        name = name.strip()
        
        if name not in self.timers:
            logger.warning(f"Timer '{name}' no existe")
            return None
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        logger.debug(f"Timer detenido: {name} (duración: {duration:.3f}s)")
        return duration

    def set_active_tasks(self, count: int) -> None:
        """
        Establecer número de tareas activas con validaciones.

        Args:
            count: Número de tareas activas (debe ser entero >= 0)
            
        Raises:
            ValueError: Si count es inválido
        """
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"count debe ser un entero no negativo, recibido: {count}")
        
        try:
            if self.use_prometheus:
                self.active_tasks.set(count)
            else:
                self.metrics["active_tasks"] = count
            logger.debug(f"Tareas activas establecidas: {count}")
        except Exception as e:
            logger.error(f"Error al establecer tareas activas: {e}", exc_info=True)
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener todas las métricas.

        Returns:
            Diccionario con todas las métricas
        """
        if self.use_prometheus:
            # Prometheus expone métricas vía HTTP, retornamos resumen
            return {
                "prometheus_enabled": True,
                "metrics_available_at": "/metrics"
            }
        else:
            # Calcular estadísticas agregadas
            result = {
                "timestamp": datetime.now().isoformat(),
                "metrics": dict(self.metrics),
                "active_timers": len(self.timers)
            }

            # Calcular promedios de duración
            for key, values in self.metrics.items():
                if key.endswith(".duration") and isinstance(values, list):
                    if values:
                        result[f"{key}.avg"] = sum(values) / len(values)
                        result[f"{key}.min"] = min(values)
                        result[f"{key}.max"] = max(values)
                        result[f"{key}.count"] = len(values)

            return result

    def reset(self) -> None:
        """Resetear todas las métricas."""
        self.metrics.clear()
        self.timers.clear()
        logger.info("Métricas reseteadas")

