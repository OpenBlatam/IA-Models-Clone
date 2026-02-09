"""
Factory for creating and managing singleton service instances.
Eliminates repetitive lazy loading patterns.
"""

from typing import TypeVar, Type, Callable, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceFactory:
    """
    Factory para crear y gestionar instancias singleton de servicios.
    """
    
    def __init__(self):
        self._instances: dict = {}
        logger.info("ServiceFactory initialized")
    
    def get_or_create(
        self,
        service_class: Type[T],
        *args,
        **kwargs
    ) -> T:
        """
        Obtiene instancia existente o crea una nueva.
        
        Args:
            service_class: Clase del servicio
            *args: Argumentos para el constructor
            **kwargs: Keyword arguments para el constructor
            
        Returns:
            Instancia del servicio
        """
        service_name = service_class.__name__
        
        if service_name not in self._instances:
            logger.info(f"Creating new instance of {service_name}")
            self._instances[service_name] = service_class(*args, **kwargs)
        else:
            logger.debug(f"Reusing existing instance of {service_name}")
        
        return self._instances[service_name]
    
    def create_factory_function(
        self,
        service_class: Type[T],
        *args,
        **kwargs
    ) -> Callable[[], T]:
        """
        Crea una función factory para un servicio específico.
        
        Args:
            service_class: Clase del servicio
            *args: Argumentos para el constructor
            **kwargs: Keyword arguments para el constructor
            
        Returns:
            Función que retorna instancia del servicio
        """
        def factory():
            return self.get_or_create(service_class, *args, **kwargs)
        
        factory.__name__ = f"get_{service_class.__name__.lower()}"
        return factory
    
    def reset(self, service_name: Optional[str] = None):
        """
        Resetea instancias (útil para testing).
        
        Args:
            service_name: Nombre del servicio a resetear, o None para todos
        """
        if service_name:
            if service_name in self._instances:
                del self._instances[service_name]
                logger.info(f"Reset instance of {service_name}")
        else:
            count = len(self._instances)
            self._instances.clear()
            logger.info(f"Reset all service instances ({count} services)")
    
    def list_services(self) -> list:
        """
        Lista todos los servicios registrados.
        
        Returns:
            Lista de nombres de servicios
        """
        return list(self._instances.keys())


# Instancia global
_factory = ServiceFactory()


def get_service_factory() -> ServiceFactory:
    """Obtiene la instancia global del factory."""
    return _factory


def create_service_getter(service_class: Type[T], *args, **kwargs) -> Callable[[], T]:
    """
    Helper para crear función getter de servicio.
    
    Args:
        service_class: Clase del servicio
        *args: Argumentos para el constructor
        **kwargs: Keyword arguments para el constructor
        
    Returns:
        Función getter del servicio
        
    Usage:
        get_analytics_service = create_service_getter(AnalyticsService)
    """
    return _factory.create_factory_function(service_class, *args, **kwargs)








