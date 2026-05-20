"""
Singleton Pattern Helper
========================
Decorator and helper for singleton pattern
"""

import logging
from typing import TypeVar, Type, Callable, Any, Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar('T')


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class a singleton.
    
    Usage:
        @singleton
        class MyService:
            def __init__(self):
                pass
    """
    instances: Dict[Type[T], T] = {}
    lock = Lock()
    
    def get_instance(*args, **kwargs) -> T:
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
                    logger.debug(f"Created singleton instance: {cls.__name__}")
        return instances[cls]
    
    # Replace class with factory function
    get_instance.__name__ = cls.__name__
    get_instance.__doc__ = cls.__doc__
    get_instance.__module__ = cls.__module__
    
    return get_instance


class SingletonMeta(type):
    """
    Metaclass for singleton pattern.
    
    Usage:
        class MyService(metaclass=SingletonMeta):
            def __init__(self):
                pass
    """
    _instances: Dict[Type, Any] = {}
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
                    logger.debug(f"Created singleton instance: {cls.__name__}")
        return cls._instances[cls]


def get_or_create_service(
    service_name: str,
    factory: Callable[[], T],
    registry: Optional[Any] = None
) -> T:
    """
    Get or create service instance (helper function).
    
    Args:
        service_name: Name of the service
        factory: Factory function to create service
        registry: Optional service registry (uses global if not provided)
    
    Returns:
        Service instance
    """
    if registry is None:
        from .service_registry import get_service_registry
        registry = get_service_registry()
    
    service = registry.get(service_name)
    if service is None:
        service = factory()
        registry.register(service_name, service)
    
    return service

