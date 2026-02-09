"""
Tracing Module - Trazabilidad y Observabilidad
Trazabilidad, logging estructurado, métricas, y observabilidad.

Rol en el Ecosistema IA:
- Logging, métricas, trazabilidad
- Tracking de llamadas LLM, latencia, tokens usados, errores
- Observabilidad completa del sistema de IA

Reglas de Importación:
- Puede importar: configs
- NO debe importar: otros módulos del proyecto (evitar ciclos)
- Todos los módulos pueden importar este módulo para logging
"""

from .base import BaseTracer
from .service import TracingService
from .logger import StructuredLogger
from .metrics import Metrics
from .main import (
    get_tracing_service,
    log,
    trace,
    record_metric,
    get_logger,
    initialize_tracing,
)

__all__ = [
    # Clases principales
    "BaseTracer",
    "TracingService",
    "StructuredLogger",
    "Metrics",
    # Funciones de acceso rápido
    "get_tracing_service",
    "log",
    "trace",
    "record_metric",
    "get_logger",
    "initialize_tracing",
]

