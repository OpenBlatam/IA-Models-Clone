"""
Core Module - Núcleo del Sistema
Módulo central con configuraciones, excepciones y utilidades comunes
"""

from .config import Settings, get_settings
from .exceptions import (
    AIHistoryException,
    ValidationError,
    NotFoundError,
    ExternalServiceError,
    CacheError
)
from .middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware
)
from .dependencies import (
    get_database_session,
    get_cache_service,
    get_llm_service,
    get_event_bus
)
from .utils import (
    generate_id,
    format_timestamp,
    validate_content,
    sanitize_input
)

__all__ = [
    # Configuración
    "Settings",
    "get_settings",
    
    # Excepciones
    "AIHistoryException",
    "ValidationError", 
    "NotFoundError",
    "ExternalServiceError",
    "CacheError",
    
    # Middleware
    "LoggingMiddleware",
    "MetricsMiddleware", 
    "RateLimitMiddleware",
    "SecurityMiddleware",
    
    # Dependencias
    "get_database_session",
    "get_cache_service",
    "get_llm_service",
    "get_event_bus",
    
    # Utilidades
    "generate_id",
    "format_timestamp",
    "validate_content",
    "sanitize_input"
]