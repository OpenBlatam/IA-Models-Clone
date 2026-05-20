"""
Servicios del sistema de recuperación
Organizados por dominios funcionales para mejor modularidad
"""

from services.service_factory import (
    ServiceFactory,
    get_service_factory,
    get_service_instance
)

__all__ = [
    'ServiceFactory',
    'get_service_factory',
    'get_service_instance',
]
