"""
Sistema de Dependency Injection

Proporciona inyección de dependencias para desacoplar componentes.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DependencyContainer:
    """Contenedor de dependencias"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        logger.info("DependencyContainer initialized")
    
    def register(self, service_name: str, service: Any, singleton: bool = True):
        """
        Registra un servicio
        
        Args:
            service_name: Nombre del servicio
            service: Instancia o clase del servicio
            singleton: Si es True, se crea una sola instancia
        """
        if singleton:
            self._singletons[service_name] = service
        else:
            self._services[service_name] = service
        
        logger.debug(f"Service registered: {service_name} (singleton={singleton})")
    
    def register_factory(self, service_name: str, factory: Callable):
        """
        Registra una factory para crear servicios
        
        Args:
            service_name: Nombre del servicio
            factory: Función que crea el servicio
        """
        self._factories[service_name] = factory
        logger.debug(f"Factory registered: {service_name}")
    
    def get(self, service_name: str) -> Optional[Any]:
        """
        Obtiene un servicio
        
        Args:
            service_name: Nombre del servicio
        
        Returns:
            Instancia del servicio o None
        """
        # Verificar singletons
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Verificar factories
        if service_name in self._factories:
            if service_name not in self._services:
                self._services[service_name] = self._factories[service_name]()
            return self._services[service_name]
        
        # Verificar servicios normales
        return self._services.get(service_name)
    
    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Resuelve un servicio por tipo
        
        Args:
            service_type: Tipo del servicio
        
        Returns:
            Instancia del servicio o None
        """
        service_name = service_type.__name__
        service = self.get(service_name)
        
        if service and isinstance(service, service_type):
            return service
        
        # Buscar por tipo en todos los servicios
        for svc in list(self._services.values()) + list(self._singletons.values()):
            if isinstance(svc, service_type):
                return svc
        
        return None
    
    def has(self, service_name: str) -> bool:
        """Verifica si un servicio está registrado"""
        return (service_name in self._services or 
                service_name in self._singletons or 
                service_name in self._factories)
    
    def clear(self):
        """Limpia todos los servicios (útil para testing)"""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()


# Contenedor global
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Obtiene el contenedor global de dependencias"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def inject(service_name: str):
    """
    Decorador para inyectar dependencias en funciones
    
    Args:
        service_name: Nombre del servicio a inyectar
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            container = get_container()
            service = container.get(service_name)
            if service:
                kwargs[service_name] = service
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def resolve_dependency(service_name: str) -> Any:
    """
    Resuelve una dependencia por nombre
    
    Args:
        service_name: Nombre del servicio
    
    Returns:
        Instancia del servicio
    
    Raises:
        ValueError: Si el servicio no se encuentra
    """
    container = get_container()
    service = container.get(service_name)
    if not service:
        raise ValueError(f"Service not found: {service_name}")
    return service


def register_service(
    service_name: str,
    service: Any,
    singleton: bool = True
) -> None:
    """
    Register a service in the global container.
    
    Args:
        service_name: Name of the service
        service: Service instance or class
        singleton: Whether to use singleton pattern
    """
    container = get_container()
    container.register(service_name, service, singleton)


def get_service(service_name: str) -> Optional[Any]:
    """
    Get a service from the global container.
    
    Args:
        service_name: Name of the service
    
    Returns:
        Service instance or None
    """
    container = get_container()
    return container.get(service_name)

