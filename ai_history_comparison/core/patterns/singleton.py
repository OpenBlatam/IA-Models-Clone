"""
Singleton Pattern Implementation

Thread-safe singleton pattern with lazy initialization
and support for different initialization strategies.
"""

import threading
import weakref
from typing import Any, Dict, Optional, TypeVar, Type, Callable
from functools import wraps

T = TypeVar('T')


class SingletonMeta(type):
    """Thread-safe singleton metaclass with lazy initialization"""
    
    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    def clear_instance(cls):
        """Clear singleton instance (useful for testing)"""
        if cls in cls._instances:
            del cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    """Base singleton class with thread-safe initialization"""
    
    def __init__(self):
        self._initialized = False
        self._lock = threading.Lock()
    
    def initialize(self, *args, **kwargs):
        """Initialize singleton instance (override in subclasses)"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._do_initialize(*args, **kwargs)
                    self._initialized = True
    
    def _do_initialize(self, *args, **kwargs):
        """Override this method in subclasses for initialization logic"""
        pass
    
    @classmethod
    def get_instance(cls) -> 'Singleton':
        """Get singleton instance"""
        return cls()
    
    @classmethod
    def clear_instance(cls):
        """Clear singleton instance"""
        SingletonMeta.clear_instance(cls)


class WeakSingletonMeta(type):
    """Singleton metaclass using weak references"""
    
    _instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class WeakSingleton(metaclass=WeakSingletonMeta):
    """Singleton using weak references (allows garbage collection)"""
    pass


def singleton(cls: Type[T]) -> Type[T]:
    """Decorator to make a class a singleton"""
    instances = {}
    lock = threading.Lock()
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


class ThreadLocalSingleton:
    """Thread-local singleton pattern"""
    
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._local = threading.local()
    
    def get_instance(self) -> Any:
        """Get thread-local instance"""
        if not hasattr(self._local, 'instance'):
            self._local.instance = self._factory()
        return self._local.instance
    
    def clear_instance(self):
        """Clear thread-local instance"""
        if hasattr(self._local, 'instance'):
            delattr(self._local, 'instance')


class LazySingleton:
    """Lazy singleton with custom initialization"""
    
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance: Optional[Any] = None
        self._lock = threading.Lock()
    
    def get_instance(self) -> Any:
        """Get lazy singleton instance"""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance
    
    def clear_instance(self):
        """Clear singleton instance"""
        with self._lock:
            self._instance = None





















