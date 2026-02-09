"""
Type Utilities
==============

Utilidades para manejo de tipos y type checking.
"""

from typing import Any, TypeVar, Union, get_origin, get_args
import inspect

T = TypeVar('T')


def get_type_name(tp: Any) -> str:
    """
    Obtener nombre legible de un tipo.
    
    Args:
        tp: Tipo a analizar
    
    Returns:
        Nombre del tipo
    """
    if tp is None:
        return "None"
    
    if isinstance(tp, type):
        return tp.__name__
    
    origin = get_origin(tp)
    if origin is None:
        return str(tp)
    
    args = get_args(tp)
    if not args:
        return origin.__name__ if hasattr(origin, '__name__') else str(origin)
    
    args_str = ", ".join(get_type_name(arg) for arg in args)
    origin_name = origin.__name__ if hasattr(origin, '__name__') else str(origin)
    return f"{origin_name}[{args_str}]"


def is_optional_type(tp: Any) -> bool:
    """
    Verificar si un tipo es Optional.
    
    Args:
        tp: Tipo a verificar
    
    Returns:
        True si es Optional
    """
    origin = get_origin(tp)
    return origin is Union and type(None) in get_args(tp)


def get_optional_inner_type(tp: Any) -> Any:
    """
    Obtener el tipo interno de un Optional.
    
    Args:
        tp: Tipo Optional
    
    Returns:
        Tipo interno o None si no es Optional
    """
    if not is_optional_type(tp):
        return None
    
    args = get_args(tp)
    non_none_args = [arg for arg in args if arg is not type(None)]
    return non_none_args[0] if non_none_args else None


def check_type(value: Any, expected_type: Any, name: str = "value") -> None:
    """
    Verificar que un valor coincida con un tipo esperado.
    
    Args:
        value: Valor a verificar
        expected_type: Tipo esperado
        name: Nombre del valor para mensajes de error
    
    Raises:
        TypeError: Si el tipo no coincide
    """
    if is_optional_type(expected_type):
        if value is None:
            return
        inner_type = get_optional_inner_type(expected_type)
        if inner_type:
            check_type(value, inner_type, name)
        return
    
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {get_type_name(expected_type)}, "
            f"got {get_type_name(type(value))}"
        )


def get_function_signature(func: Any) -> dict:
    """
    Obtener información de la firma de una función.
    
    Args:
        func: Función a analizar
    
    Returns:
        Diccionario con información de la firma
    """
    sig = inspect.signature(func)
    params = {}
    
    for name, param in sig.parameters.items():
        params[name] = {
            "name": name,
            "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "kind": str(param.kind)
        }
    
    return {
        "name": func.__name__,
        "parameters": params,
        "return_annotation": sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
    }

