"""Singleton pattern utilities."""

from typing import Any, Dict, Callable
import threading


class Singleton:
    """Thread-safe singleton base class."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance


def singleton(cls: type) -> type:
    """Decorator for singleton class."""
    instances: Dict[type, Any] = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


class SingletonMeta(type):
    """Metaclass for singleton."""
    
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ThreadLocalSingleton:
    """Thread-local singleton."""
    
    _instances: Dict[int, Any] = {}
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id not in cls._instances:
            with cls._lock:
                if thread_id not in cls._instances:
                    cls._instances[thread_id] = super().__new__(cls)
        return cls._instances[thread_id]



