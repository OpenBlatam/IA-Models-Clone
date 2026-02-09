"""
Dict Utils

Utilities for dict utils.
"""

from typing import Any, Dict, List

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionar diccionarios de forma recursiva"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Obtener valor anidado usando notación de punto (ej: 'user.profile.name')"""
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """Establecer valor anidado usando notación de punto"""
    keys = path.split('.')
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def normalize_dict_keys(data: Dict[str, Any], case: str = "lower") -> Dict[str, Any]:
    """Normalizar claves de diccionario a minúsculas o mayúsculas"""
    if case == "lower":
        return {k.lower(): v for k, v in data.items()}
    elif case == "upper":
        return {k.upper(): v for k, v in data.items()}
    return data

def filter_dict(data: Dict[str, Any], keys: list[str]) -> Dict[str, Any]:
    """Filtrar diccionario manteniendo solo las claves especificadas"""
    return {k: v for k, v in data.items() if k in keys}

def exclude_dict_keys(data: Dict[str, Any], keys: list[str]) -> Dict[str, Any]:
    """Excluir claves específicas de un diccionario"""
    return {k: v for k, v in data.items() if k not in keys}

def flatten_dict(data: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """Aplanar diccionario anidado"""
    items = []
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)

def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Desaplanar diccionario usando separador"""
    result = {}
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result

