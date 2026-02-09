"""
Sistema de Métricas Avanzadas
==============================
Métricas con Prometheus y estadísticas
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import structlog
from collections import defaultdict
import time

logger = structlog.get_logger()

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, using in-memory metrics")


class MetricType(str, Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricsCollector:
    """Colector de métricas"""
    
    def __init__(self):
        """Inicializar colector"""
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._summaries: Dict[str, Any] = {}
        self._in_memory_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "value": 0,
            "count": 0,
            "sum": 0.0,
            "min": float('inf'),
            "max": float('-inf')
        })
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        logger.info("MetricsCollector initialized", prometheus=PROMETHEUS_AVAILABLE)
    
    def _init_prometheus_metrics(self) -> None:
        """Inicializar métricas de Prometheus"""
        # Contadores
        self._counters["validations_created"] = Counter(
            "psych_val_validations_created_total",
            "Total number of validations created"
        )
        self._counters["validations_completed"] = Counter(
            "psych_val_validations_completed_total",
            "Total number of validations completed"
        )
        self._counters["validations_failed"] = Counter(
            "psych_val_validations_failed_total",
            "Total number of validations failed"
        )
        self._counters["api_requests"] = Counter(
            "psych_val_api_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status"]
        )
        
        # Gauges
        self._gauges["active_validations"] = Gauge(
            "psych_val_active_validations",
            "Number of active validations"
        )
        self._gauges["active_connections"] = Gauge(
            "psych_val_active_connections",
            "Number of active social media connections"
        )
        
        # Histogramas
        self._histograms["validation_duration"] = Histogram(
            "psych_val_validation_duration_seconds",
            "Duration of validation processing",
            buckets=[1, 5, 10, 30, 60, 120, 300]
        )
        self._histograms["api_response_time"] = Histogram(
            "psych_val_api_response_time_seconds",
            "API response time",
            ["endpoint"],
            buckets=[0.1, 0.5, 1, 2, 5, 10]
        )
        
        # Summaries
        self._summaries["profile_confidence"] = Summary(
            "psych_val_profile_confidence",
            "Profile confidence score"
        )
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Incrementar contador
        
        Args:
            name: Nombre del contador
            value: Valor a incrementar
            labels: Etiquetas (opcional)
        """
        if PROMETHEUS_AVAILABLE and name in self._counters:
            counter = self._counters[name]
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)
        
        # In-memory
        self._in_memory_metrics[name]["value"] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Establecer gauge
        
        Args:
            name: Nombre del gauge
            value: Valor
            labels: Etiquetas (opcional)
        """
        if PROMETHEUS_AVAILABLE and name in self._gauges:
            gauge = self._gauges[name]
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
        
        # In-memory
        self._in_memory_metrics[name]["value"] = value
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observar histograma
        
        Args:
            name: Nombre del histograma
            value: Valor a observar
            labels: Etiquetas (opcional)
        """
        if PROMETHEUS_AVAILABLE and name in self._histograms:
            histogram = self._histograms[name]
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)
        
        # In-memory
        metric = self._in_memory_metrics[name]
        metric["count"] += 1
        metric["sum"] += value
        metric["min"] = min(metric["min"], value)
        metric["max"] = max(metric["max"], value)
        metric["value"] = metric["sum"] / metric["count"] if metric["count"] > 0 else 0
    
    def observe_summary(
        self,
        name: str,
        value: float
    ) -> None:
        """
        Observar summary
        
        Args:
            name: Nombre del summary
            value: Valor a observar
        """
        if PROMETHEUS_AVAILABLE and name in self._summaries:
            self._summaries[name].observe(value)
        
        # In-memory
        self.observe_histogram(name, value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener todas las métricas
        
        Returns:
            Diccionario con métricas
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metrics": dict(self._in_memory_metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_prometheus_metrics(self) -> Optional[str]:
        """
        Obtener métricas en formato Prometheus
        
        Returns:
            Métricas en formato Prometheus o None
        """
        if not PROMETHEUS_AVAILABLE:
            return None
        
        try:
            from prometheus_client import generate_latest
            return generate_latest().decode('utf-8')
        except Exception as e:
            logger.error("Error generating Prometheus metrics", error=str(e))
            return None


# Instancia global del colector de métricas
metrics_collector = MetricsCollector()




