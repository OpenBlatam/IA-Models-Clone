"""
Configuration Helper Module
===========================

Helper functions para trabajar con configuración de forma más limpia.
"""

from typing import Dict, Any, Optional, TypeVar, Callable

T = TypeVar('T')


def get_config_value(
    config: Optional[Dict[str, Any]],
    key: str,
    default: T,
    validator: Optional[Callable[[Any], T]] = None
) -> T:
    """
    Obtener valor de configuración con validación opcional.
    
    Args:
        config: Diccionario de configuración
        key: Clave a obtener
        default: Valor por defecto
        validator: Función de validación opcional
        
    Returns:
        Valor de configuración validado
    """
    if not config:
        return default
    
    value = config.get(key, default)
    
    if validator:
        try:
            return validator(value)
        except (ValueError, TypeError):
            return default
    
    return value


def get_bool_config(config: Optional[Dict[str, Any]], key: str, default: bool = False) -> bool:
    """Obtener valor booleano de configuración."""
    return get_config_value(config, key, default, bool)


def get_int_config(
    config: Optional[Dict[str, Any]],
    key: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> int:
    """Obtener valor entero de configuración con límites."""
    def validator(value: Any) -> int:
        val = int(value)
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    
    return get_config_value(config, key, default, validator)


def get_float_config(
    config: Optional[Dict[str, Any]],
    key: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """Obtener valor float de configuración con límites."""
    def validator(value: Any) -> float:
        val = float(value)
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return val
    
    return get_config_value(config, key, default, validator)


def get_str_config(
    config: Optional[Dict[str, Any]],
    key: str,
    default: str,
    valid_values: Optional[list] = None
) -> str:
    """Obtener valor string de configuración con validación."""
    def validator(value: Any) -> str:
        val = str(value)
        if valid_values and val not in valid_values:
            return default
        return val
    
    return get_config_value(config, key, default, validator)
