"""
Helper functions for common conditional patterns.
Eliminates repetitive if/else patterns.
"""

from typing import Any, Optional, Callable, TypeVar, List

T = TypeVar('T')


def if_none(value: Any, default: T) -> T:
    """
    Retorna default si value es None, sino retorna value.
    
    Args:
        value: Valor a verificar
        default: Valor por defecto
        
    Returns:
        value si no es None, sino default
        
    Usage:
        >>> if_none(None, "default")
        'default'
        
        >>> if_none("value", "default")
        'value'
    """
    return default if value is None else value


def if_empty(value: Any, default: T) -> T:
    """
    Retorna default si value está vacío, sino retorna value.
    
    Args:
        value: Valor a verificar (string, list, dict, etc.)
        default: Valor por defecto
        
    Returns:
        value si no está vacío, sino default
        
    Usage:
        >>> if_empty("", "default")
        'default'
        
        >>> if_empty([], [1, 2, 3])
        [1, 2, 3]
    """
    if not value:
        return default
    return value


def if_falsy(value: Any, default: T) -> T:
    """
    Retorna default si value es falsy, sino retorna value.
    
    Args:
        value: Valor a verificar
        default: Valor por defecto
        
    Returns:
        value si es truthy, sino default
        
    Usage:
        >>> if_falsy(0, 10)
        10
        
        >>> if_falsy(5, 10)
        5
    """
    return default if not value else value


def first_not_none(*values: Any) -> Optional[Any]:
    """
    Retorna el primer valor que no sea None.
    
    Args:
        *values: Valores a verificar
        
    Returns:
        Primer valor no None o None si todos son None
        
    Usage:
        >>> first_not_none(None, None, "value", "other")
        'value'
    """
    for value in values:
        if value is not None:
            return value
    return None


def first_not_empty(*values: Any) -> Optional[Any]:
    """
    Retorna el primer valor que no esté vacío.
    
    Args:
        *values: Valores a verificar
        
    Returns:
        Primer valor no vacío o None si todos están vacíos
        
    Usage:
        >>> first_not_empty("", [], "value", "other")
        'value'
    """
    for value in values:
        if value:
            return value
    return None


def coalesce(*values: Any, default: Any = None) -> Any:
    """
    Retorna el primer valor truthy o default.
    
    Args:
        *values: Valores a verificar
        default: Valor por defecto si todos son falsy
        
    Returns:
        Primer valor truthy o default
        
    Usage:
        >>> coalesce(None, "", 0, "value")
        'value'
        
        >>> coalesce(None, "", 0, default="default")
        'default'
    """
    for value in values:
        if value:
            return value
    return default


def when(
    condition: bool,
    true_value: T,
    false_value: Optional[T] = None
) -> Optional[T]:
    """
    Retorna true_value si condition es True, sino false_value.
    
    Args:
        condition: Condición a evaluar
        true_value: Valor si condición es True
        false_value: Valor si condición es False (default: None)
        
    Returns:
        true_value o false_value
        
    Usage:
        >>> when(len(items) > 0, items[0], [])
        [1, 2, 3]  # si items no está vacío
    """
    return true_value if condition else false_value


def unless(
    condition: bool,
    false_value: T,
    true_value: Optional[T] = None
) -> Optional[T]:
    """
    Retorna false_value si condition es False, sino true_value.
    Es el opuesto de when().
    
    Args:
        condition: Condición a evaluar
        false_value: Valor si condición es False
        true_value: Valor si condición es True (default: None)
        
    Returns:
        false_value o true_value
        
    Usage:
        >>> unless(items is None, items, [])
        []  # si items es None
    """
    return false_value if not condition else true_value








