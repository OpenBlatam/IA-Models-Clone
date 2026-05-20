"""
Decorator Pattern - Add functionality to objects
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


class IDecorator(ABC):
    """
    Interface for decorators
    """
    
    @abstractmethod
    def decorate(self, func: Callable) -> Callable:
        """Decorate a function"""
        pass


class BaseDecorator(IDecorator):
    """
    Base decorator implementation
    """
    
    def __init__(self, name: str = "BaseDecorator"):
        self.name = name
    
    def decorate(self, func: Callable) -> Callable:
        """Base decorate - override in subclasses"""
        return func


def decorator_class(func: Callable) -> Callable:
    """
    Class decorator for easy decoration
    """
    class DecoratedClass:
        def __init__(self, *args, **kwargs):
            self._wrapped = func(*args, **kwargs)
        
        def __getattr__(self, name):
            return getattr(self._wrapped, name)
    
    return DecoratedClass








