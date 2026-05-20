"""
Servicio de monitoreo avanzado para métricas y alertas.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from config.logging_config import get_logger
from config.di_setup import get_service

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Tipos de métricas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Severidad de alertas."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Metric:
    """
    Representa una métrica con validaciones.
    
    Attributes:
        name: Nombre de la métrica
        metric_type: Tipo de métrica
        value: Valor de la métrica
        labels: Etiquetas adicionales
        timestamp: Timestamp de la métrica
    """
    
    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        value: float = 0.0,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Inicializar métrica con validaciones.
        
        Args:
            name: Nombre de la métrica (debe ser string no vacío)
            metric_type: Tipo de métrica (debe ser MetricType)
            value: Valor de la métrica (default: 0.0)
            labels: Etiquetas adicionales (opcional)
            timestamp: Timestamp de la métrica (opcional)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        if not isinstance(metric_type, MetricType):
            raise ValueError(f"metric_type debe ser un MetricType, recibido: {type(metric_type)}")
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"value debe ser un número, recibido: {type(value)}")
        
        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(f"labels debe ser un diccionario, recibido: {type(labels)}")
            # Validar que todos los valores sean strings
            for k, v in labels.items():
                if not isinstance(v, str):
                    raise ValueError(f"Todos los valores de labels deben ser strings, recibido: {type(v)} para '{k}'")
        
        self.name = name.strip()
        self.metric_type = metric_type
        self.value = float(value)
        self.labels = labels or {}
        self.timestamp = timestamp or datetime.now()
        
        logger.debug(f"Métrica creada: {self.name} (type: {self.metric_type.value}, value: {self.value})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir métrica a diccionario."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat()
        }


class AlertRule:
    """Regla de alerta."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: Optional[str] = None,
        cooldown_seconds: int = 300
    ):
        """
        Inicializar regla de alerta.
        
        Args:
            name: Nombre de la regla
            metric_name: Nombre de la métrica a monitorear
            condition: Función que retorna True si debe alertar
            severity: Severidad de la alerta
            message: Mensaje personalizado
            cooldown_seconds: Tiempo mínimo entre alertas (segundos)
        """
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.message = message or f"Alert: {name}"
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[datetime] = None
    
    def should_alert(self, metric_value: float) -> bool:
        """
        Verificar si debe alertar.
        
        Args:
            metric_value: Valor actual de la métrica
            
        Returns:
            True si debe alertar
        """
        if not self.condition(metric_value):
            return False
        
        # Verificar cooldown
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        
        self.last_triggered = datetime.now()
        return True


class MonitoringService:
    """
    Servicio de monitoreo avanzado con mejoras.
    
    Attributes:
        window_size: Tamaño de ventana para métricas históricas
        metrics: Diccionario de métricas históricas
        gauges: Diccionario de gauges actuales
        counters: Diccionario de contadores
        histograms: Diccionario de histogramas
        alert_rules: Lista de reglas de alerta
        alert_handlers: Lista de handlers de alerta
        running: Si el monitoreo está activo
        monitoring_task: Tarea de monitoreo asíncrona
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Inicializar servicio de monitoreo con validaciones.
        
        Args:
            window_size: Tamaño de ventana para métricas históricas (debe ser entero positivo)
            
        Raises:
            ValueError: Si window_size es inválido
        """
        # Validación
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size debe ser un entero positivo, recibido: {window_size}")
        
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.gauges: Dict[str, float] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.alert_rules: List[AlertRule] = []
        self.alert_handlers: List[Callable] = []
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"✅ MonitoringService inicializado (window_size: {window_size})")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Registrar una métrica con validaciones.
        
        Args:
            name: Nombre de la métrica (debe ser string no vacío)
            value: Valor (debe ser número)
            metric_type: Tipo de métrica (default: GAUGE)
            labels: Etiquetas adicionales (opcional)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones básicas (Metric.__init__ también valida)
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"value debe ser un número, recibido: {type(value)}")
        
        if not isinstance(metric_type, MetricType):
            raise ValueError(f"metric_type debe ser un MetricType, recibido: {type(metric_type)}")
        
        try:
            metric = Metric(name, metric_type, value, labels)
            
            if metric_type == MetricType.COUNTER:
                self.counters[name] += int(value)
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self.histograms[name].append(value)
                if len(self.histograms[name]) > self.window_size:
                    self.histograms[name] = self.histograms[name][-self.window_size:]
            
            # Agregar a historial
            self.metrics[name].append(metric)
            
            # Verificar alertas
            self._check_alerts(name, value)
            
            logger.debug(f"✅ Métrica registrada: {name} = {value} (type: {metric_type.value})")
        except Exception as e:
            logger.error(f"Error al registrar métrica {name}: {e}", exc_info=True)
            raise
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Incrementar contador."""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Establecer gauge."""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Registrar valor en histograma."""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Agregar regla de alerta.
        
        Args:
            rule: Regla de alerta
        """
        self.alert_rules.append(rule)
        logger.info(f"Regla de alerta agregada: {rule.name}")
    
    def register_alert_handler(self, handler: Callable) -> None:
        """
        Registrar handler de alertas.
        
        Args:
            handler: Función que maneja alertas
        """
        self.alert_handlers.append(handler)
    
    def _check_alerts(self, metric_name: str, metric_value: float) -> None:
        """Verificar reglas de alerta para una métrica."""
        for rule in self.alert_rules:
            if rule.metric_name == metric_name and rule.should_alert(metric_value):
                self._trigger_alert(rule, metric_value)
    
    def _trigger_alert(self, rule: AlertRule, metric_value: float) -> None:
        """Disparar alerta."""
        alert = {
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "metric_value": metric_value,
            "severity": rule.severity.value,
            "message": rule.message,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(
            f"ALERT [{rule.severity.value}]: {rule.name} - {rule.message} "
            f"(metric: {rule.metric_name}={metric_value})"
        )
        
        # Ejecutar handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error en alert handler: {e}", exc_info=True)
    
    def get_metric(self, name: str, window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtener métricas históricas.
        
        Args:
            name: Nombre de la métrica
            window: Número de puntos a retornar (opcional)
            
        Returns:
            Lista de métricas
        """
        metrics = list(self.metrics.get(name, deque()))
        if window:
            metrics = metrics[-window:]
        return [m.to_dict() for m in metrics]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales."""
        return {
            "gauges": self.gauges.copy(),
            "counters": self.counters.copy(),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p50": self._percentile(values, 50) if values else 0,
                    "p95": self._percentile(values, 95) if values else 0,
                    "p99": self._percentile(values, 99) if values else 0
                }
                for name, values in self.histograms.items()
            }
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calcular percentil."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio."""
        return {
            "total_metrics": sum(len(m) for m in self.metrics.values()),
            "gauges_count": len(self.gauges),
            "counters_count": len(self.counters),
            "histograms_count": len(self.histograms),
            "alert_rules_count": len(self.alert_rules),
            "alert_handlers_count": len(self.alert_handlers),
            "current_metrics": self.get_current_metrics()
        }
    
    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Iniciar monitoreo periódico con validaciones.
        
        Args:
            interval_seconds: Intervalo entre verificaciones en segundos (debe ser entero positivo)
            
        Raises:
            ValueError: Si interval_seconds es inválido
            RuntimeError: Si el monitoreo ya está corriendo
        """
        # Validación
        if not isinstance(interval_seconds, int) or interval_seconds < 1:
            raise ValueError(f"interval_seconds debe ser un entero positivo, recibido: {interval_seconds}")
        
        if self.running:
            logger.warning("Monitoring service ya está corriendo")
            raise RuntimeError("Monitoring service ya está corriendo")
        
        self.running = True
        
        async def monitor_loop():
            logger.info(f"🔄 Monitoring loop iniciado (interval: {interval_seconds}s)")
            while self.running:
                try:
                    # Recolectar métricas del sistema
                    await self._collect_system_metrics()
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    logger.info("Monitoring loop cancelado")
                    break
                except Exception as e:
                    logger.error(f"❌ Error en monitoring loop: {e}", exc_info=True)
                    await asyncio.sleep(interval_seconds)
            logger.info("Monitoring loop finalizado")
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
        logger.info(f"✅ Monitoring service iniciado (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self) -> None:
        """Detener monitoreo."""
        self.running = False
        if self.monitoring_task:
            await self.monitoring_task
        logger.info("Monitoring service detenido")
    
    async def _collect_system_metrics(self) -> None:
        """Recolectar métricas del sistema."""
        try:
            import psutil
            process = psutil.Process()
            
            # CPU
            self.set_gauge("system.cpu.percent", process.cpu_percent())
            
            # Memoria
            mem_info = process.memory_info()
            self.set_gauge("system.memory.rss", mem_info.rss / 1024 / 1024)  # MB
            self.set_gauge("system.memory.vms", mem_info.vms / 1024 / 1024)  # MB
            
            # Threads
            self.set_gauge("system.threads", process.num_threads())
            
        except ImportError:
            logger.debug("psutil no disponible para métricas del sistema")
        except Exception as e:
            logger.debug(f"Error recolectando métricas del sistema: {e}")

