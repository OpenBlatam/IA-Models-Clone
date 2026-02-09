"""
Dependency Injection - Improved
===============================

Sistema mejorado de inyección de dependencias con gestión de ciclo de vida,
resolución automática de dependencias y soporte async.
"""

import logging
import inspect
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, Awaitable
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifecycle(Enum):
    """Ciclo de vida de servicios."""
    SINGLETON = "singleton"  # Una instancia para toda la aplicación
    SCOPED = "scoped"  # Una instancia por scope (request, etc.)
    TRANSIENT = "transient"  # Nueva instancia cada vez


class Container:
    """
    Contenedor mejorado de dependencias con gestión de ciclo de vida.
    """
    
    def __init__(self):
        """Inicializar contenedor."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._async_factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped: Dict[str, Dict[Type, Any]] = {}  # Scope ID -> services
        self._lifecycles: Dict[Type, Lifecycle] = {}
        self._current_scope: Optional[str] = None
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[T] = None,
        factory: Optional[Union[Callable[[], T], Callable[[], Awaitable[T]]]] = None,
        lifecycle: Lifecycle = Lifecycle.SINGLETON
    ):
        """
        Registrar servicio.
        
        Args:
            service_type: Tipo del servicio
            implementation: Implementación concreta (opcional)
            factory: Factory function (opcional, puede ser async)
            lifecycle: Ciclo de vida del servicio
        """
        if implementation is not None:
            if lifecycle == Lifecycle.SINGLETON:
                self._singletons[service_type] = implementation
            elif lifecycle == Lifecycle.SCOPED:
                # Para scoped, guardamos la factory
                self._factories[service_type] = lambda: implementation
            else:
                # Transient: guardamos factory que crea nueva instancia
                self._factories[service_type] = lambda: implementation
        elif factory is not None:
            # Verificar si es async
            if inspect.iscoroutinefunction(factory):
                self._async_factories[service_type] = factory
            else:
                self._factories[service_type] = factory
        else:
            raise ValueError("Debe proporcionar implementation o factory")
        
        self._lifecycles[service_type] = lifecycle
        logger.debug(f"Servicio registrado: {service_type.__name__} (lifecycle: {lifecycle.value})")
    
    def register_instance(self, service_type: Type[T], instance: T, lifecycle: Lifecycle = Lifecycle.SINGLETON):
        """
        Registrar instancia.
        
        Args:
            service_type: Tipo del servicio
            instance: Instancia
            lifecycle: Ciclo de vida
        """
        if lifecycle == Lifecycle.SINGLETON:
            self._singletons[service_type] = instance
        self._lifecycles[service_type] = lifecycle
    
    async def resolve_async(self, service_type: Type[T]) -> T:
        """
        Resolver servicio (async).
        
        Args:
            service_type: Tipo del servicio
            
        Returns:
            Instancia del servicio
        """
        lifecycle = self._lifecycles.get(service_type, Lifecycle.SINGLETON)
        
        # Singleton: siempre la misma instancia
        if lifecycle == Lifecycle.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
        
        # Scoped: una instancia por scope
        elif lifecycle == Lifecycle.SCOPED:
            if not self._current_scope:
                raise ValueError("No hay scope activo para servicio scoped")
            
            if self._current_scope not in self._scoped:
                self._scoped[self._current_scope] = {}
            
            if service_type in self._scoped[self._current_scope]:
                return self._scoped[self._current_scope][service_type]
        
        # Crear nueva instancia
        instance = None
        
        # Intentar resolver con factory async
        if service_type in self._async_factories:
            instance = await self._async_factories[service_type]()
        # Intentar resolver con factory normal
        elif service_type in self._factories:
            instance = self._factories[service_type]()
        # Intentar auto-resolver (buscar constructor con dependencias)
        else:
            instance = await self._auto_resolve(service_type)
        
        if instance is None:
            raise ValueError(f"Servicio '{service_type.__name__}' no registrado y no se puede auto-resolver")
        
        # Guardar según lifecycle
        if lifecycle == Lifecycle.SINGLETON:
            self._singletons[service_type] = instance
        elif lifecycle == Lifecycle.SCOPED:
            if self._current_scope:
                if self._current_scope not in self._scoped:
                    self._scoped[self._current_scope] = {}
                self._scoped[self._current_scope][service_type] = instance
        
        return instance
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolver servicio (sync).
        
        Args:
            service_type: Tipo del servicio
            
        Returns:
            Instancia del servicio
        """
        # Para sync, intentar usar async si hay event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si hay loop corriendo, no podemos usar await aquí
                # Usar versión sync
                return self._resolve_sync(service_type)
            else:
                return loop.run_until_complete(self.resolve_async(service_type))
        except RuntimeError:
            # No hay event loop, usar sync
            return self._resolve_sync(service_type)
    
    def _resolve_sync(self, service_type: Type[T]) -> T:
        """Resolver servicio de forma síncrona."""
        lifecycle = self._lifecycles.get(service_type, Lifecycle.SINGLETON)
        
        if lifecycle == Lifecycle.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
        
        elif lifecycle == Lifecycle.SCOPED:
            if not self._current_scope:
                raise ValueError("No hay scope activo para servicio scoped")
            if self._current_scope in self._scoped:
                if service_type in self._scoped[self._current_scope]:
                    return self._scoped[self._current_scope][service_type]
        
        # Crear instancia
        if service_type in self._factories:
            instance = self._factories[service_type]()
        else:
            instance = self._auto_resolve_sync(service_type)
        
        if instance is None:
            raise ValueError(f"Servicio '{service_type.__name__}' no registrado")
        
        if lifecycle == Lifecycle.SINGLETON:
            self._singletons[service_type] = instance
        elif lifecycle == Lifecycle.SCOPED:
            if self._current_scope:
                if self._current_scope not in self._scoped:
                    self._scoped[self._current_scope] = {}
                self._scoped[self._current_scope][service_type] = instance
        
        return instance
    
    async def _auto_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Auto-resolver servicio analizando constructor."""
        try:
            # Obtener signature del constructor
            sig = inspect.signature(service_type.__init__)
            params = list(sig.parameters.values())[1:]  # Skip 'self'
            
            # Resolver dependencias
            dependencies = []
            for param in params:
                if param.annotation != inspect.Parameter.empty:
                    dep = await self.resolve_async(param.annotation)
                    dependencies.append(dep)
                else:
                    dependencies.append(None)
            
            # Crear instancia
            return service_type(*dependencies)
        except Exception as e:
            logger.debug(f"No se pudo auto-resolver {service_type.__name__}: {e}")
            return None
    
    def _auto_resolve_sync(self, service_type: Type[T]) -> Optional[T]:
        """Auto-resolver servicio de forma síncrona."""
        try:
            sig = inspect.signature(service_type.__init__)
            params = list(sig.parameters.values())[1:]
            
            dependencies = []
            for param in params:
                if param.annotation != inspect.Parameter.empty:
                    dep = self._resolve_sync(param.annotation)
                    dependencies.append(dep)
                else:
                    dependencies.append(None)
            
            return service_type(*dependencies)
        except Exception as e:
            logger.debug(f"No se pudo auto-resolver {service_type.__name__}: {e}")
            return None
    
    def create_scope(self, scope_id: Optional[str] = None) -> str:
        """
        Crear nuevo scope.
        
        Args:
            scope_id: ID del scope (se genera si no se proporciona)
            
        Returns:
            ID del scope creado
        """
        import uuid
        scope_id = scope_id or str(uuid.uuid4())
        self._scoped[scope_id] = {}
        return scope_id
    
    def enter_scope(self, scope_id: str):
        """
        Entrar a un scope.
        
        Args:
            scope_id: ID del scope
        """
        if scope_id not in self._scoped:
            self._scoped[scope_id] = {}
        self._current_scope = scope_id
    
    def exit_scope(self):
        """Salir del scope actual."""
        if self._current_scope:
            # Limpiar servicios scoped del scope actual
            if self._current_scope in self._scoped:
                del self._scoped[self._current_scope]
        self._current_scope = None
    
    def is_registered(self, service_type: Type) -> bool:
        """
        Verificar si servicio está registrado.
        
        Args:
            service_type: Tipo del servicio
            
        Returns:
            True si está registrado
        """
        return (
            service_type in self._services or
            service_type in self._factories or
            service_type in self._async_factories or
            service_type in self._singletons or
            service_type in self._lifecycles
        )
    
    def clear(self):
        """Limpiar todos los servicios."""
        self._services.clear()
        self._factories.clear()
        self._async_factories.clear()
        self._singletons.clear()
        self._scoped.clear()
        self._lifecycles.clear()
        self._current_scope = None


class ServiceProvider:
    """
    Proveedor de servicios.
    """
    
    def __init__(self, container: Optional[Container] = None):
        """
        Inicializar proveedor.
        
        Args:
            container: Contenedor (opcional, crea uno nuevo si no se proporciona)
        """
        self.container = container or Container()
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Obtener servicio.
        
        Args:
            service_type: Tipo del servicio
            
        Returns:
            Instancia del servicio
        """
        return self.container.resolve(service_type)
    
    def register_service(
        self,
        service_type: Type[T],
        implementation: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        singleton: bool = True
    ):
        """
        Registrar servicio.
        
        Args:
            service_type: Tipo del servicio
            implementation: Implementación (opcional)
            factory: Factory (opcional)
            singleton: Si es singleton
        """
        self.container.register(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            singleton=singleton
        )


# Instancia global
_global_container = Container()
_global_provider = ServiceProvider(_global_container)


def register_service(
    service_type: Type[T],
    implementation: Optional[T] = None,
    factory: Optional[Callable[[], T]] = None,
    singleton: bool = True
):
    """
    Registrar servicio globalmente.
    
    Args:
        service_type: Tipo del servicio
        implementation: Implementación (opcional)
        factory: Factory (opcional)
        singleton: Si es singleton
    """
    _global_container.register(
        service_type=service_type,
        implementation=implementation,
        factory=factory,
        singleton=singleton
    )


def resolve_service(service_type: Type[T]) -> T:
    """
    Resolver servicio globalmente.
    
    Args:
        service_type: Tipo del servicio
        
    Returns:
        Instancia del servicio
    """
    return _global_container.resolve(service_type)


def inject(*dependencies):
    """
    Decorator para inyectar dependencias.
    
    Args:
        dependencies: Tipos de dependencias
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resolver dependencias
            resolved = []
            for dep_type in dependencies:
                resolved.append(resolve_service(dep_type))
            
            # Llamar función con dependencias inyectadas
            return func(*args, *resolved, **kwargs)
        return wrapper
    return decorator

