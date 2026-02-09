"""
Helper functions for dictionary operations.
Eliminates repetitive .get() patterns and nested dictionary access.
"""

from typing import Any, Dict, Optional, List, Callable, TypeVar

T = TypeVar('T')


def safe_get(
    data: Dict[str, Any],
    key: str,
    default: Any = None,
    transform: Optional[Callable[[Any], Any]] = None
) -> Any:
    """
    Obtiene un valor de un diccionario de forma segura con transformación opcional.
    
    Args:
        data: Diccionario a consultar
        key: Clave a obtener
        default: Valor por defecto si la clave no existe
        transform: Función de transformación opcional
        
    Returns:
        Valor obtenido o default
        
    Usage:
        >>> safe_get(profile_data, "display_name", "Unknown")
        'John Doe'
        
        >>> safe_get(profile_data, "followers_count", 0, transform=int)
        1234
    """
    value = data.get(key, default)
    if transform and value is not None and value != default:
        try:
            return transform(value)
        except (ValueError, TypeError):
            return default
    return value


def nested_get(
    data: Dict[str, Any],
    *keys: str,
    default: Any = None
) -> Any:
    """
    Obtiene un valor de un diccionario anidado de forma segura.
    
    Args:
        data: Diccionario a consultar
        *keys: Claves anidadas (ej: "features", "has_emojis")
        default: Valor por defecto si alguna clave no existe
        
    Returns:
        Valor obtenido o default
        
    Usage:
        >>> nested_get(data, "features", "has_emojis", default=False)
        True
        
        >>> nested_get(data, "user", "profile", "name", default="Unknown")
        'John Doe'
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def get_or_default(
    data: Dict[str, Any],
    key: str,
    default: Any = None,
    fallback_key: Optional[str] = None
) -> Any:
    """
    Obtiene un valor con múltiples opciones de fallback.
    
    Args:
        data: Diccionario a consultar
        key: Clave principal
        default: Valor por defecto si la clave no existe
        fallback_key: Clave alternativa si la principal no existe
        
    Returns:
        Valor obtenido o default
        
    Usage:
        >>> get_or_default(headers, "X-API-Key", fallback_key="Authorization")
        'api_key_123'
    """
    value = data.get(key)
    if value is not None:
        return value
    
    if fallback_key:
        value = data.get(fallback_key)
        if value is not None:
            return value
    
    return default


def extract_fields(
    data: Dict[str, Any],
    fields: List[str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extrae múltiples campos de un diccionario con defaults.
    
    Args:
        data: Diccionario fuente
        fields: Lista de campos a extraer
        defaults: Diccionario con valores por defecto por campo
        
    Returns:
        Diccionario con los campos extraídos
        
    Usage:
        >>> extract_fields(
        ...     profile_data,
        ...     ["display_name", "bio", "followers_count"],
        ...     defaults={"followers_count": 0}
        ... )
        {"display_name": "John", "bio": "...", "followers_count": 1234}
    """
    defaults = defaults or {}
    return {
        field: data.get(field, defaults.get(field))
        for field in fields
    }


def safe_pop(
    data: Dict[str, Any],
    key: str,
    default: Any = None
) -> Any:
    """
    Elimina y retorna un valor de un diccionario de forma segura.
    
    Args:
        data: Diccionario
        key: Clave a eliminar
        default: Valor por defecto si la clave no existe
        
    Returns:
        Valor eliminado o default
        
    Usage:
        >>> value = safe_pop(data, "temp_field", default=None)
    """
    return data.pop(key, default)


def merge_dicts(
    *dicts: Dict[str, Any],
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Combina múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a combinar
        overwrite: Si sobrescribir valores existentes (default: True)
        
    Returns:
        Diccionario combinado
        
    Usage:
        >>> merge_dicts(dict1, dict2, dict3)
        {**dict1, **dict2, **dict3}
    """
    result = {}
    for d in dicts:
        if overwrite:
            result.update(d)
        else:
            for key, value in d.items():
                if key not in result:
                    result[key] = value
    return result








