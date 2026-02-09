"""
Type Utilities for Piel Mejorador AI SAM3
=========================================

Unified type checking and introspection utilities.
"""

import asyncio
import logging
from typing import Any, Type, TypeVar, Union, Callable, Optional, List, Dict, Generic, Awaitable
from pathlib import Path
from inspect import isfunction, ismethod, isclass, signature, Parameter

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Result(Generic[T]):
    """Result type for operations that can fail."""
    
    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self.value = value
        self.error = error
        self.is_success = error is None
        self.is_error = error is not None
    
    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create successful result."""
        return cls(value=value, error=None)
    
    @classmethod
    def failure(cls, error: Exception) -> "Result[T]":
        """Create failed result."""
        return cls(value=None, error=error)
    
    def unwrap(self) -> T:
        """Unwrap result, raising error if failed."""
        if self.is_error:
            raise self.error
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Unwrap result or return default."""
        return self.value if self.is_success else default


# Type aliases for common patterns
FilePath = Union[str, Path]
ConfigDict = Dict[str, Any]
TaskDict = Dict[str, Any]
ServiceResult = Dict[str, Any]

# Async function types
AsyncFunction = Callable[..., Awaitable[Any]]
SyncFunction = Callable[..., Any]


class TypeUtils:
    """Unified type checking utilities."""
    
    @staticmethod
    def is_type(value: Any, expected_type: Type) -> bool:
        """
        Check if value is of expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type
            
        Returns:
            True if value is of expected type
        """
        return isinstance(value, expected_type)
    
    @staticmethod
    def is_one_of(value: Any, *types: Type) -> bool:
        """
        Check if value is one of multiple types.
        
        Args:
            value: Value to check
            *types: Types to check against
            
        Returns:
            True if value is one of the types
        """
        return isinstance(value, types)
    
    @staticmethod
    def is_callable(value: Any) -> bool:
        """
        Check if value is callable.
        
        Args:
            value: Value to check
            
        Returns:
            True if callable
        """
        return callable(value)
    
    @staticmethod
    def is_async_callable(value: Any) -> bool:
        """
        Check if value is async callable.
        
        Args:
            value: Value to check
            
        Returns:
            True if async callable
        """
        if not callable(value):
            return False
        
        # Check if it's a coroutine function
        if asyncio.iscoroutinefunction(value):
            return True
        
        # Check if it's a method that returns a coroutine
        if hasattr(value, '__call__'):
            try:
                sig = signature(value)
                # This is a simplified check
                return asyncio.iscoroutinefunction(value.__call__)
            except Exception:
                pass
        
        return False
    
    @staticmethod
    def has_attr(value: Any, *attrs: str) -> bool:
        """
        Check if value has all specified attributes.
        
        Args:
            value: Value to check
            *attrs: Attribute names
            
        Returns:
            True if all attributes exist
        """
        return all(hasattr(value, attr) for attr in attrs)
    
    @staticmethod
    def has_method(value: Any, method_name: str) -> bool:
        """
        Check if value has a method.
        
        Args:
            value: Value to check
            method_name: Method name
            
        Returns:
            True if method exists and is callable
        """
        return hasattr(value, method_name) and callable(getattr(value, method_name, None))
    
    @staticmethod
    def get_type_name(value: Any) -> str:
        """
        Get type name of value.
        
        Args:
            value: Value to inspect
            
        Returns:
            Type name string
        """
        return type(value).__name__
    
    @staticmethod
    def get_type(value: Any) -> Type:
        """
        Get type of value.
        
        Args:
            value: Value to inspect
            
        Returns:
            Type object
        """
        return type(value)
    
    @staticmethod
    def is_subclass(value: Any, base_class: Type) -> bool:
        """
        Check if value is a subclass of base_class.
        
        Args:
            value: Class to check
            base_class: Base class
            
        Returns:
            True if subclass
        """
        return isclass(value) and issubclass(value, base_class)
    
    @staticmethod
    def is_instance_of_any(value: Any, *classes: Type) -> bool:
        """
        Check if value is instance of any of the classes.
        
        Args:
            value: Value to check
            *classes: Classes to check against
            
        Returns:
            True if instance of any class
        """
        return isinstance(value, classes)
    
    @staticmethod
    def get_annotations(obj: Any) -> Dict[str, Any]:
        """
        Get type annotations from object.
        
        Args:
            obj: Object to inspect (function, class, etc.)
            
        Returns:
            Dictionary of annotations
        """
        if hasattr(obj, '__annotations__'):
            return obj.__annotations__
        return {}
    
    @staticmethod
    def get_signature(func: Callable) -> signature:
        """
        Get function signature.
        
        Args:
            func: Function to inspect
            
        Returns:
            Signature object
        """
        return signature(func)
    
    @staticmethod
    def is_optional_type(type_hint: Any) -> bool:
        """
        Check if type hint is Optional.
        
        Args:
            type_hint: Type hint to check
            
        Returns:
            True if Optional
        """
        # Check for Union with None
        if hasattr(type_hint, '__origin__'):
            origin = type_hint.__origin__
            if origin is Union:
                args = type_hint.__args__
                return type(None) in args
        return False


# Convenience functions
def is_type(value: Any, expected_type: Type) -> bool:
    """Check if value is of expected type."""
    return TypeUtils.is_type(value, expected_type)


def is_callable(value: Any) -> bool:
    """Check if value is callable."""
    return TypeUtils.is_callable(value)


def is_async_callable(value: Any) -> bool:
    """Check if value is async callable."""
    return TypeUtils.is_async_callable(value)


def has_attr(value: Any, *attrs: str) -> bool:
    """Check if value has attributes."""
    return TypeUtils.has_attr(value, *attrs)


def has_method(value: Any, method_name: str) -> bool:
    """Check if value has method."""
    return TypeUtils.has_method(value, method_name)

