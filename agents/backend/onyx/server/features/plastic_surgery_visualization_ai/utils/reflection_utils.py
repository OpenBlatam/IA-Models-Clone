"""Reflection utilities."""

import inspect
from typing import Any, Callable, Optional, Dict


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    Get function signature information.
    
    Args:
        func: Function to inspect
        
    Returns:
        Dictionary with signature information
    """
    sig = inspect.signature(func)
    
    return {
        "name": func.__name__,
        "parameters": {
            name: {
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                "kind": str(param.kind),
            }
            for name, param in sig.parameters.items()
        },
        "return_annotation": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None,
    }


def get_class_methods(cls: type) -> list:
    """
    Get all methods of a class.
    
    Args:
        cls: Class to inspect
        
    Returns:
        List of method names
    """
    return [
        name for name, method in inspect.getmembers(cls, inspect.ismethod)
        if not name.startswith('_')
    ]


def get_class_attributes(cls: type) -> list:
    """
    Get all attributes of a class.
    
    Args:
        cls: Class to inspect
        
    Returns:
        List of attribute names
    """
    return [
        name for name, attr in inspect.getmembers(cls)
        if not name.startswith('_') and not inspect.ismethod(attr) and not inspect.isfunction(attr)
    ]


def is_callable(obj: Any) -> bool:
    """
    Check if object is callable.
    
    Args:
        obj: Object to check
        
    Returns:
        True if callable
    """
    return callable(obj)


def get_source_code(obj: Any) -> Optional[str]:
    """
    Get source code of object.
    
    Args:
        obj: Object to inspect
        
    Returns:
        Source code or None
    """
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return None


def get_docstring(obj: Any) -> Optional[str]:
    """
    Get docstring of object.
    
    Args:
        obj: Object to inspect
        
    Returns:
        Docstring or None
    """
    return inspect.getdoc(obj)


def get_file_location(obj: Any) -> Optional[str]:
    """
    Get file location of object.
    
    Args:
        obj: Object to inspect
        
    Returns:
        File path or None
    """
    try:
        return inspect.getfile(obj)
    except (OSError, TypeError):
        return None

