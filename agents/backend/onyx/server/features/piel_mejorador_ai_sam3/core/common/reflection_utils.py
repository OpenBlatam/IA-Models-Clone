"""
Reflection Utilities for Piel Mejorador AI SAM3
===============================================

Unified reflection and introspection utilities.
"""

import logging
from typing import Any, Optional, Callable, List, Dict, Type, TypeVar
from inspect import (
    getmembers,
    isfunction,
    ismethod,
    isclass,
    getdoc,
    signature,
    Parameter,
    getfile,
    getsourcefile,
    getmodule
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ReflectionUtils:
    """Unified reflection utilities."""
    
    @staticmethod
    def safe_getattr(
        obj: Any,
        attr: str,
        default: Any = None
    ) -> Any:
        """
        Safely get attribute with default.
        
        Args:
            obj: Object
            attr: Attribute name
            default: Default value
            
        Returns:
            Attribute value or default
        """
        return getattr(obj, attr, default)
    
    @staticmethod
    def safe_setattr(
        obj: Any,
        attr: str,
        value: Any
    ) -> bool:
        """
        Safely set attribute.
        
        Args:
            obj: Object
            attr: Attribute name
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            setattr(obj, attr, value)
            return True
        except (AttributeError, TypeError) as e:
            logger.warning(f"Cannot set attribute {attr}: {e}")
            return False
    
    @staticmethod
    def safe_delattr(
        obj: Any,
        attr: str
    ) -> bool:
        """
        Safely delete attribute.
        
        Args:
            obj: Object
            attr: Attribute name
            
        Returns:
            True if successful
        """
        try:
            if hasattr(obj, attr):
                delattr(obj, attr)
                return True
            return False
        except (AttributeError, TypeError) as e:
            logger.warning(f"Cannot delete attribute {attr}: {e}")
            return False
    
    @staticmethod
    def get_attributes(obj: Any, include_private: bool = False) -> List[str]:
        """
        Get all attribute names.
        
        Args:
            obj: Object
            include_private: Whether to include private attributes
            
        Returns:
            List of attribute names
        """
        attrs = dir(obj)
        if not include_private:
            attrs = [a for a in attrs if not a.startswith("_")]
        return attrs
    
    @staticmethod
    def get_methods(obj: Any, include_private: bool = False) -> List[str]:
        """
        Get all method names.
        
        Args:
            obj: Object
            include_private: Whether to include private methods
            
        Returns:
            List of method names
        """
        methods = [
            name for name, value in getmembers(obj, predicate=callable)
            if not name.startswith("__") or include_private
        ]
        return methods
    
    @staticmethod
    def get_properties(obj: Any) -> List[str]:
        """
        Get all property names.
        
        Args:
            obj: Object
            
        Returns:
            List of property names
        """
        return [
            name for name, value in getmembers(obj)
            if isinstance(value, property)
        ]
    
    @staticmethod
    def get_class_hierarchy(cls: Type) -> List[Type]:
        """
        Get class hierarchy (MRO).
        
        Args:
            cls: Class
            
        Returns:
            List of classes in MRO order
        """
        return list(cls.__mro__)
    
    @staticmethod
    def get_base_classes(cls: Type) -> List[Type]:
        """
        Get direct base classes.
        
        Args:
            cls: Class
            
        Returns:
            List of base classes
        """
        return list(cls.__bases__)
    
    @staticmethod
    def has_method(obj: Any, method_name: str) -> bool:
        """
        Check if object has method.
        
        Args:
            obj: Object
            method_name: Method name
            
        Returns:
            True if method exists
        """
        return hasattr(obj, method_name) and callable(getattr(obj, method_name, None))
    
    @staticmethod
    def get_method_signature(obj: Any, method_name: str) -> Optional[signature]:
        """
        Get method signature.
        
        Args:
            obj: Object
            method_name: Method name
            
        Returns:
            Signature or None
        """
        if not ReflectionUtils.has_method(obj, method_name):
            return None
        
        method = getattr(obj, method_name)
        try:
            return signature(method)
        except Exception:
            return None
    
    @staticmethod
    def get_method_parameters(obj: Any, method_name: str) -> Dict[str, Parameter]:
        """
        Get method parameters.
        
        Args:
            obj: Object
            method_name: Method name
            
        Returns:
            Dictionary mapping parameter names to Parameter objects
        """
        sig = ReflectionUtils.get_method_signature(obj, method_name)
        if sig:
            return dict(sig.parameters)
        return {}
    
    @staticmethod
    def call_method(
        obj: Any,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call method by name.
        
        Args:
            obj: Object
            method_name: Method name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
            
        Raises:
            AttributeError: If method doesn't exist
        """
        method = getattr(obj, method_name)
        if not callable(method):
            raise AttributeError(f"{method_name} is not callable")
        return method(*args, **kwargs)
    
    @staticmethod
    def get_docstring(obj: Any) -> Optional[str]:
        """
        Get docstring.
        
        Args:
            obj: Object
            
        Returns:
            Docstring or None
        """
        return getdoc(obj)
    
    @staticmethod
    def get_source_file(obj: Any) -> Optional[str]:
        """
        Get source file path.
        
        Args:
            obj: Object
            
        Returns:
            Source file path or None
        """
        try:
            return getfile(obj)
        except (TypeError, OSError):
            return None
    
    @staticmethod
    def get_module(obj: Any) -> Optional[str]:
        """
        Get module name.
        
        Args:
            obj: Object
            
        Returns:
            Module name or None
        """
        module = getmodule(obj)
        return module.__name__ if module else None


class AttributeAccessor:
    """Safe attribute accessor with fallbacks."""
    
    def __init__(self, obj: Any):
        """
        Initialize attribute accessor.
        
        Args:
            obj: Object to access
        """
        self.obj = obj
    
    def get(self, attr: str, default: Any = None) -> Any:
        """
        Get attribute value.
        
        Args:
            attr: Attribute name (supports dot notation: "attr.subattr")
            default: Default value
            
        Returns:
            Attribute value or default
        """
        parts = attr.split(".")
        value = self.obj
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value
    
    def set(self, attr: str, value: Any) -> bool:
        """
        Set attribute value.
        
        Args:
            attr: Attribute name (supports dot notation: "attr.subattr")
            value: Value to set
            
        Returns:
            True if successful
        """
        parts = attr.split(".")
        if len(parts) == 1:
            return ReflectionUtils.safe_setattr(self.obj, attr, value)
        
        # Navigate to parent
        parent_attr = ".".join(parts[:-1])
        final_attr = parts[-1]
        
        parent = self.get(parent_attr)
        if parent is None:
            return False
        
        return ReflectionUtils.safe_setattr(parent, final_attr, value)
    
    def has(self, attr: str) -> bool:
        """
        Check if attribute exists.
        
        Args:
            attr: Attribute name (supports dot notation)
            
        Returns:
            True if exists
        """
        parts = attr.split(".")
        value = self.obj
        
        for part in parts:
            if not hasattr(value, part):
                return False
            value = getattr(value, part)
        
        return True
    
    def delete(self, attr: str) -> bool:
        """
        Delete attribute.
        
        Args:
            attr: Attribute name (supports dot notation)
            
        Returns:
            True if deleted
        """
        parts = attr.split(".")
        if len(parts) == 1:
            return ReflectionUtils.safe_delattr(self.obj, attr)
        
        # Navigate to parent
        parent_attr = ".".join(parts[:-1])
        final_attr = parts[-1]
        
        parent = self.get(parent_attr)
        if parent is None:
            return False
        
        return ReflectionUtils.safe_delattr(parent, final_attr)


# Convenience functions
def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute."""
    return ReflectionUtils.safe_getattr(obj, attr, default)


def safe_setattr(obj: Any, attr: str, value: Any) -> bool:
    """Safely set attribute."""
    return ReflectionUtils.safe_setattr(obj, attr, value)


def safe_delattr(obj: Any, attr: str) -> bool:
    """Safely delete attribute."""
    return ReflectionUtils.safe_delattr(obj, attr)


def get_methods(obj: Any, **kwargs) -> List[str]:
    """Get all method names."""
    return ReflectionUtils.get_methods(obj, **kwargs)


def call_method(obj: Any, method_name: str, *args, **kwargs) -> Any:
    """Call method by name."""
    return ReflectionUtils.call_method(obj, method_name, *args, **kwargs)




