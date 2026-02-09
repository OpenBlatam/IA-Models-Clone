"""
Service Locator Pattern

Proporciona acceso global a servicios sin acoplamiento directo.
"""

import logging
from typing import Dict, Any, Optional, TypeVar, Type
from core.dependency_injection import get_container

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLocator:
    """Localizador de servicios global"""
    
    def __init__(self):
        self._container = get_container()
        logger.info("ServiceLocator initialized")
    
    def get(self, service_name: str) -> Optional[Any]:
        """
        Obtiene un servicio por nombre
        
        Args:
            service_name: Nombre del servicio
        
        Returns:
            Instancia del servicio o None
        """
        return self._container.get(service_name)
    
    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Resuelve un servicio por tipo
        
        Args:
            service_type: Tipo del servicio
        
        Returns:
            Instancia del servicio o None
        """
        return self._container.resolve(service_type)
    
    def has(self, service_name: str) -> bool:
        """Verifica si un servicio está disponible"""
        return self._container.has(service_name)


# Instancia global
_service_locator: Optional[ServiceLocator] = None


def get_service(service_name: str) -> Optional[Any]:
    """
    Obtiene un servicio por nombre (función helper)
    
    Args:
        service_name: Nombre del servicio
    
    Returns:
        Instancia del servicio
    """
    global _service_locator
    if _service_locator is None:
        _service_locator = ServiceLocator()
    return _service_locator.get(service_name)


def resolve_service(service_type: Type[T]) -> Optional[T]:
    """
    Resuelve un servicio por tipo (función helper)
    
    Args:
        service_type: Tipo del servicio
    
    Returns:
        Instancia del servicio
    """
    global _service_locator
    if _service_locator is None:
        _service_locator = ServiceLocator()
    return _service_locator.resolve(service_type)

