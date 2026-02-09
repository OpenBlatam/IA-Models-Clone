"""Object utilities."""

from typing import Any, Dict, Optional
import copy
import inspect


def deep_copy(obj: Any) -> Any:
    """
    Create deep copy of object.
    
    Args:
        obj: Object to copy
        
    Returns:
        Deep copy
    """
    return copy.deepcopy(obj)


def shallow_copy(obj: Any) -> Any:
    """
    Create shallow copy of object.
    
    Args:
        obj: Object to copy
        
    Returns:
        Shallow copy
    """
    return copy.copy(obj)


def get_object_attributes(obj: Any) -> Dict[str, Any]:
    """
    Get all attributes of object.
    
    Args:
        obj: Object
        
    Returns:
        Dictionary of attributes
    """
    return {key: getattr(obj, key) for key in dir(obj) if not key.startswith('_')}


def set_object_attributes(obj: Any, attributes: Dict[str, Any]) -> None:
    """
    Set multiple attributes on object.
    
    Args:
        obj: Object
        attributes: Dictionary of attributes to set
    """
    for key, value in attributes.items():
        setattr(obj, key, value)


def has_attribute(obj: Any, attr_name: str) -> bool:
    """
    Check if object has attribute.
    
    Args:
        obj: Object
        attr_name: Attribute name
        
    Returns:
        True if object has attribute
    """
    return hasattr(obj, attr_name)


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    Get attribute from object with default.
    
    Args:
        obj: Object
        attr_name: Attribute name
        default: Default value if attribute doesn't exist
        
    Returns:
        Attribute value or default
    """
    return getattr(obj, attr_name, default)


def call_method(obj: Any, method_name: str, *args, **kwargs) -> Any:
    """
    Call method on object.
    
    Args:
        obj: Object
        method_name: Method name
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Method result
    """
    method = getattr(obj, method_name)
    return method(*args, **kwargs)


def get_class_name(obj: Any) -> str:
    """
    Get class name of object.
    
    Args:
        obj: Object
        
    Returns:
        Class name
    """
    return obj.__class__.__name__


def get_module_name(obj: Any) -> str:
    """
    Get module name of object's class.
    
    Args:
        obj: Object
        
    Returns:
        Module name
    """
    return obj.__class__.__module__


def is_instance(obj: Any, class_or_tuple) -> bool:
    """
    Check if object is instance of class.
    
    Args:
        obj: Object
        class_or_tuple: Class or tuple of classes
        
    Returns:
        True if object is instance
    """
    return isinstance(obj, class_or_tuple)


def get_methods(obj: Any) -> list:
    """
    Get all methods of object.
    
    Args:
        obj: Object
        
    Returns:
        List of method names
    """
    return [name for name, method in inspect.getmembers(obj, inspect.ismethod)]


def get_properties(obj: Any) -> list:
    """
    Get all properties of object.
    
    Args:
        obj: Object
        
    Returns:
        List of property names
    """
    return [name for name, prop in inspect.getmembers(obj.__class__, inspect.isdatadescriptor) if isinstance(prop, property)]

