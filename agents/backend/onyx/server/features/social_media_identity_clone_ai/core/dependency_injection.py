"""
Sistema simple de dependency injection
Mejora modularidad y testabilidad
"""

from typing import Dict, Type, TypeVar, Callable, Any, Optional
from functools import lru_cache

T = TypeVar('T')


class ServiceContainer:
    """Contenedor simple de servicios para dependency injection"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = True
    ) -> None:
        """
        Registra un servicio en el contenedor
        
        Args:
            service_type: Tipo/clase del servicio
            implementation: Instancia del servicio (opcional)
            factory: Función factory para crear instancia (opcional)
            singleton: Si es singleton (default: True)
        """
        if implementation is not None:
            if singleton:
                self._singletons[service_type] = implementation
            else:
                self._services[service_type] = implementation
        elif factory is not None:
            self._factories[service_type] = factory
        else:
            raise ValueError("Debe proporcionar implementation o factory")
    
    def get(self, service_type: Type[T]) -> T:
        """
        Obtiene instancia del servicio
        
        Args:
            service_type: Tipo del servicio a obtener
            
        Returns:
            Instancia del servicio
            
        Raises:
            ValueError: Si el servicio no está registrado
        """
        # Verificar singleton
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Verificar servicios directos
        if service_type in self._services:
            return self._services[service_type]
        
        # Verificar factories
        if service_type in self._factories:
            instance = self._factories[service_type]()
            if service_type in self._singletons:
                self._singletons[service_type] = instance
            return instance
        
        raise ValueError(f"Servicio {service_type} no registrado")
    
    def has(self, service_type: Type[T]) -> bool:
        """Verifica si un servicio está registrado"""
        return (
            service_type in self._services or
            service_type in self._factories or
            service_type in self._singletons
        )
    
    def clear(self) -> None:
        """Limpia todos los servicios"""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Contenedor global
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Obtiene el contenedor global de servicios"""
    return _container


def register_service(
    service_type: Type[T],
    implementation: Optional[T] = None,
    factory: Optional[Callable[[], T]] = None,
    singleton: bool = True
) -> None:
    """Registra un servicio en el contenedor global"""
    _container.register(service_type, implementation, factory, singleton)


def get_service(service_type: Type[T]) -> T:
    """Obtiene un servicio del contenedor global"""
    return _container.get(service_type)


@lru_cache(maxsize=128)
def resolve_service(service_type: Type[T]) -> T:
    """
    Resuelve servicio con caché (útil para funciones)
    
    Args:
        service_type: Tipo del servicio
        
    Returns:
        Instancia del servicio
    """
    return _container.get(service_type)

