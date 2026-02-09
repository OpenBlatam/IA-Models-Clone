"""
Tracing Main - Funciones base y entry points del módulo de trazabilidad

Rol en el Ecosistema IA:
- Logging, métricas, trazabilidad
- Tracking de llamadas LLM, latencia, tokens usados, errores
- Observabilidad completa del sistema de IA
"""

from typing import Optional, Dict, Any
from .service import TracingService
from .logger import StructuredLogger
from .metrics import Metrics


# Instancia global del servicio
_tracing_service: Optional[TracingService] = None


def get_tracing_service() -> TracingService:
    """
    Obtiene la instancia global del servicio de tracing.
    
    Returns:
        TracingService: Servicio de tracing
    """
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService()
    return _tracing_service


def log(level: str, message: str, **kwargs) -> None:
    """
    Registra un log.
    
    Args:
        level: Nivel de log (info, error, warning, debug)
        message: Mensaje a registrar
        **kwargs: Metadata adicional
    """
    service = get_tracing_service()
    service.log(level, message, **kwargs)


def trace(operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Traza una operación.
    
    Args:
        operation: Nombre de la operación
        metadata: Metadata adicional
    """
    service = get_tracing_service()
    service.trace(operation, metadata)


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Registra una métrica.
    
    Args:
        name: Nombre de la métrica
        value: Valor de la métrica
        tags: Tags adicionales
    """
    service = get_tracing_service()
    service.record_metric(name, value, tags)


def get_logger(name: str = "suno_clone_ai") -> StructuredLogger:
    """
    Obtiene un logger estructurado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        StructuredLogger: Logger estructurado
    """
    return StructuredLogger(name)


def initialize_tracing() -> TracingService:
    """
    Inicializa el sistema de tracing.
    
    Returns:
        TracingService: Servicio inicializado
    """
    return get_tracing_service()

