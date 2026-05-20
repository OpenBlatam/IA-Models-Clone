"""
MCP Observability Manager - Gestor de observabilidad
======================================================

Gestor centralizado de observabilidad que integra:
- Métricas Prometheus
- Tracing OpenTelemetry
- Logging estructurado

Proporciona una interfaz unificada para monitoreo y debugging.
"""

import logging
import time
from typing import Dict, Any, Optional, ContextManager
from contextlib import contextmanager
from datetime import datetime

from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from .metrics import MCPMetrics
from .tracing import MCPTracer

logger = logging.getLogger(__name__)


class MCPObservability:
    """
    Gestor de observabilidad para MCP
    
    Integra:
    - Métricas Prometheus
    - Tracing OpenTelemetry
    - Logging estructurado
    """
    
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        otlp_endpoint: Optional[str] = None,
    ):
        """
        Inicializa observabilidad.
        
        Args:
            enable_tracing: Habilitar tracing
            enable_metrics: Habilitar métricas
            otlp_endpoint: Endpoint OTLP para tracing (opcional)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not isinstance(enable_tracing, bool):
            raise ValueError("enable_tracing must be a boolean")
        if not isinstance(enable_metrics, bool):
            raise ValueError("enable_metrics must be a boolean")
        if otlp_endpoint is not None:
            if not isinstance(otlp_endpoint, str):
                raise TypeError(f"otlp_endpoint must be a string or None, got {type(otlp_endpoint)}")
            if not otlp_endpoint or not otlp_endpoint.strip():
                raise ValueError("otlp_endpoint cannot be empty or whitespace if provided")
        
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        
        # Inicializar métricas
        if enable_metrics:
            try:
                self.metrics = MCPMetrics()
                logger.info("MCP Metrics initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize metrics: {e}", exc_info=True)
                self.metrics = None
                self.enable_metrics = False
        else:
            self.metrics = None
        
        # Inicializar tracing
        if enable_tracing:
            try:
                self.tracer = MCPTracer(otlp_endpoint=otlp_endpoint)
                logger.info("MCP Tracer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize tracer: {e}", exc_info=True)
                self.tracer = None
                self.enable_tracing = False
        else:
            self.tracer = None
    
    @contextmanager
    def trace(self, operation: str, **attributes):
        """
        Context manager para tracing de operaciones.
        
        Args:
            operation: Nombre de la operación (debe ser no vacío)
            **attributes: Atributos adicionales del trace
            
        Yields:
            Span del trace (o None si tracing está deshabilitado)
            
        Raises:
            ValueError: Si operation es inválido
        """
        if not operation or not isinstance(operation, str):
            raise ValueError("operation must be a non-empty string")
        
        operation = operation.strip()
        if not operation:
            raise ValueError("operation cannot be empty or whitespace only")
        
        if self.tracer:
            try:
                with self.tracer.start_span(operation, **attributes) as span:
                    start_time = time.time()
                    try:
                        yield span
                    except Exception as e:
                        span.record_exception(e)
                        if self.metrics:
                            self.metrics.record_error(f"{operation}_error", operation=operation)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration", duration)
                        if self.metrics:
                            self.metrics.record_latency(operation, duration)
            except Exception as e:
                logger.warning(f"Error in tracer for operation {operation}: {e}")
                # Fallback: solo métricas sin tracing
                start_time = time.time()
                try:
                    yield None
                finally:
                    duration = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_latency(operation, duration)
        else:
            start_time = time.time()
            try:
                yield None
            finally:
                duration = time.time() - start_time
                if self.metrics:
                    self.metrics.record_latency(operation, duration)
    
    def record_metric(self, metric_name: str, value: float, **labels):
        """
        Registra una métrica.
        
        Args:
            metric_name: Nombre de la métrica (debe ser no vacío)
            value: Valor de la métrica (debe ser numérico)
            **labels: Labels adicionales
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not metric_name or not isinstance(metric_name, str):
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(value, (int, float)):
            raise ValueError("value must be a number")
        
        if self.metrics:
            try:
                self.metrics.record(metric_name, value, **labels)
            except Exception as e:
                logger.warning(f"Failed to record metric {metric_name}: {e}")
    
    def record_error(self, error_type: str, **labels):
        """
        Registra un error.
        
        Args:
            error_type: Tipo de error (debe ser no vacío)
            **labels: Labels adicionales
            
        Raises:
            ValueError: Si error_type es inválido
        """
        if not error_type or not isinstance(error_type, str):
            raise ValueError("error_type must be a non-empty string")
        
        if self.metrics:
            try:
                self.metrics.record_error(error_type, **labels)
            except Exception as e:
                logger.warning(f"Failed to record error {error_type}: {e}")
        
        logger.error(
            f"MCP Error: {error_type}",
            extra={**labels, "error_type": error_type}
        )
    
    def record_context_size(self, size: int, resource_id: Optional[str] = None):
        """
        Registra tamaño de contexto.
        
        Args:
            size: Tamaño en tokens/bytes (debe ser >= 0)
            resource_id: ID del recurso (opcional)
            
        Raises:
            ValueError: Si size es inválido
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        
        if self.metrics:
            try:
                self.metrics.record_context_size(size, resource_id=resource_id)
            except Exception as e:
                logger.warning(f"Failed to record context size: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de métricas.
        
        Returns:
            Diccionario con resumen de métricas, incluyendo estado
            de habilitación de métricas y tracing
        """
        summary = {
            "metrics_enabled": self.enable_metrics,
            "tracing_enabled": self.enable_tracing,
        }
        
        if self.metrics:
            try:
                metrics_data = self.metrics.get_summary()
                summary.update(metrics_data)
            except Exception as e:
                logger.warning(f"Failed to get metrics summary: {e}")
                summary["error"] = str(e)
        else:
            summary["metrics_data"] = {}
        
        return summary

