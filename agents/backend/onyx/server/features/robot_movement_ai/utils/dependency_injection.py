"""
Dependency Injection - Sistema de inyección de dependencias
============================================================

Sistema simple de dependency injection para facilitar testing y modularidad.
"""

from typing import TypeVar, Type, Dict, Any, Optional, Callable
from functools import wraps
import inspect

T = TypeVar('T')


class DependencyContainer:
    """Contenedor de dependencias."""
    
    def __init__(self):
        """Inicializar contenedor."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._singleton_flags: Dict[Type, bool] = {}
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = False
    ):
        """
        Registrar un servicio.
        
        Args:
            service_type: Tipo del servicio (clase o interfaz)
            implementation: Instancia concreta del servicio
            factory: Función factory para crear el servicio
            singleton: Si True, se crea una sola instancia
        """
        if implementation is not None:
            if singleton:
                self._singletons[service_type] = implementation
            else:
                self._services[service_type] = implementation
        elif factory is not None:
            self._factories[service_type] = factory
            self._singleton_flags[service_type] = singleton
        else:
            raise ValueError("Must provide either implementation or factory")
    
    def get(self, service_type: Type[T]) -> T:
        """
        Obtener instancia del servicio.
        
        Args:
            service_type: Tipo del servicio a obtener
        
        Returns:
            Instancia del servicio
        
        Raises:
            ValueError: Si el servicio no está registrado
        """
        if service_type in self._services:
            return self._services[service_type]
        
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        if service_type in self._factories:
            factory = self._factories[service_type]
            if self._singleton_flags.get(service_type, False):
                if service_type not in self._singletons:
                    self._singletons[service_type] = factory()
                return self._singletons[service_type]
            else:
                return factory()
        
        raise ValueError(f"Service {service_type} not registered")
    
    def has(self, service_type: Type[T]) -> bool:
        """
        Verificar si un servicio está registrado.
        
        Args:
            service_type: Tipo del servicio
        
        Returns:
            True si está registrado
        """
        return (
            service_type in self._services or
            service_type in self._singletons or
            service_type in self._factories
        )
    
    def clear(self):
        """Limpiar todas las dependencias."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_flags.clear()


_global_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Obtener contenedor global de dependencias."""
    return _global_container


def inject(*dependencies: Type):
    """
    Decorador para inyectar dependencias en una función.
    
    Args:
        *dependencies: Tipos de dependencias a inyectar
    
    Example:
        @inject(RobotConfig, Logger)
        def my_function(config: RobotConfig, logger: Logger):
            ...
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            injected = {}
            
            for dep_type in dependencies:
                if container.has(dep_type):
                    dep_name = None
                    for name, param in sig.parameters.items():
                        if param.annotation == dep_type or param.annotation == inspect.Parameter.empty:
                            dep_name = name
                            break
                    
                    if dep_name and dep_name not in kwargs and dep_name not in param_names[:len(args)]:
                        injected[dep_name] = container.get(dep_type)
            
            return func(*args, **{**injected, **kwargs})
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            container = get_container()
            injected = {}
            
            for dep_type in dependencies:
                if container.has(dep_type):
                    dep_name = None
                    for name, param in sig.parameters.items():
                        if param.annotation == dep_type or param.annotation == inspect.Parameter.empty:
                            dep_name = name
                            break
                    
                    if dep_name and dep_name not in kwargs and dep_name not in param_names[:len(args)]:
                        injected[dep_name] = container.get(dep_type)
            
            return await func(*args, **{**injected, **kwargs})
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator

