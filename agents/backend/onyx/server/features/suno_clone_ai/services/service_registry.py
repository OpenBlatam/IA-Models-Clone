"""
Registro de Servicios

Gestiona el registro y acceso a servicios de forma centralizada.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar
from core.dependency_injection import get_container

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """Registro centralizado de servicios"""
    
    def __init__(self):
        self._container = get_container()
        self._service_types: Dict[str, Type] = {}
        logger.info("ServiceRegistry initialized")
    
    def register(
        self,
        service_name: str,
        service: Any,
        service_type: Optional[Type] = None,
        singleton: bool = True
    ):
        """
        Registra un servicio
        
        Args:
            service_name: Nombre del servicio
            service: Instancia o clase del servicio
            service_type: Tipo del servicio (opcional)
            singleton: Si es True, se crea una sola instancia
        """
        self._container.register(service_name, service, singleton=singleton)
        
        if service_type:
            self._service_types[service_name] = service_type
        
        logger.debug(f"Service registered: {service_name}")
    
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
        """Verifica si un servicio está registrado"""
        return self._container.has(service_name)
    
    def list_services(self) -> list:
        """Lista todos los servicios registrados"""
        # Esto requeriría acceso interno al contenedor
        # Por ahora retornamos los tipos conocidos
        return list(self._service_types.keys())


# Instancia global
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Obtiene la instancia global del registro de servicios"""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


def register_service(
    service_name: str,
    service: Any,
    service_type: Optional[Type] = None,
    singleton: bool = True
):
    """
    Registra un servicio (función helper)
    
    Args:
        service_name: Nombre del servicio
        service: Instancia o clase
        service_type: Tipo del servicio
        singleton: Si es singleton
    """
    registry = get_service_registry()
    registry.register(service_name, service, service_type, singleton)


def get_service_instance(service_name: str) -> Optional[Any]:
    """
    Obtiene una instancia de servicio (función helper)
    
    Args:
        service_name: Nombre del servicio
    
    Returns:
        Instancia del servicio
    """
    registry = get_service_registry()
    return registry.get(service_name)

