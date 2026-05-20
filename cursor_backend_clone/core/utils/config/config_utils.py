"""
Config Utils - Utilidades de Configuración Avanzadas
=====================================================

Utilidades avanzadas para manejo y validación de configuración.
"""

import logging
import os
from typing import Any, Optional, Dict, List, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def get_env(
    key: str,
    default: Optional[Any] = None,
    required: bool = False,
    type_cast: Optional[Callable] = None
) -> Optional[Any]:
    """
    Obtener variable de entorno con validación y casting.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si es requerida
        type_cast: Función para convertir tipo
        
    Returns:
        Valor de la variable
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    if value is not None and type_cast:
        try:
            return type_cast(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {key}: {e}")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Obtener variable de entorno como boolean.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        
    Returns:
        Valor boolean
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    """
    Obtener variable de entorno como integer.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si es requerida
        
    Returns:
        Valor integer
    """
    return get_env(key, default, required, int)


def get_env_float(key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
    """
    Obtener variable de entorno como float.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        required: Si es requerida
        
    Returns:
        Valor float
    """
    return get_env(key, default, required, float)


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ',') -> List[str]:
    """
    Obtener variable de entorno como lista.
    
    Args:
        key: Nombre de la variable
        default: Valor por defecto
        separator: Separador de elementos
        
    Returns:
        Lista de strings
    """
    value = os.getenv(key)
    if value is None:
        return default or []
    
    return [item.strip() for item in value.split(separator) if item.strip()]


def load_config_from_file(
    filepath: Union[str, Path],
    format: str = "json"
) -> Dict[str, Any]:
    """
    Cargar configuración desde archivo.
    
    Args:
        filepath: Ruta del archivo
        format: Formato (json, yaml, toml)
        
    Returns:
        Diccionario de configuración
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if format.lower() == "json":
        try:
            import orjson
            return orjson.loads(filepath.read_bytes())
        except ImportError:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    elif format.lower() == "yaml":
        try:
            import yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files")
    
    elif format.lower() == "toml":
        try:
            import tomllib
            with open(filepath, 'rb') as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import tomli
                with open(filepath, 'rb') as f:
                    return tomli.load(f)
            except ImportError:
                raise ImportError("tomllib or tomli is required for TOML config files")
    
    else:
        raise ValueError(f"Unsupported config format: {format}")


def save_config_to_file(
    config: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Guardar configuración a archivo.
    
    Args:
        config: Diccionario de configuración
        filepath: Ruta del archivo
        format: Formato (json, yaml)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        try:
            import orjson
            filepath.write_bytes(
                orjson.dumps(config, option=orjson.OPT_INDENT_2)
            )
        except ImportError:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == "yaml":
        try:
            import yaml
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files")
    
    else:
        raise ValueError(f"Unsupported config format: {format}")


def merge_configs(*configs: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Fusionar múltiples configuraciones.
    
    Args:
        *configs: Configuraciones a fusionar
        deep: Si hacer merge profundo
        
    Returns:
        Configuración fusionada
    """
    if not configs:
        return {}
    
    result = configs[0].copy()
    
    for config in configs[1:]:
        if deep:
            result = _deep_merge(result, config)
        else:
            result.update(config)
    
    return result


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge profundo de diccionarios"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(
    config: Dict[str, Any],
    schema: Dict[str, Any],
    strict: bool = False
) -> tuple[bool, List[str]]:
    """
    Validar configuración contra esquema.
    
    Args:
        config: Configuración a validar
        schema: Esquema de validación
        strict: Si rechazar keys no definidas
        
    Returns:
        Tupla (es_válido, errores)
    """
    errors = []
    
    # Validar keys requeridas
    required_keys = schema.get('required', [])
    for key in required_keys:
        if key not in config:
            errors.append(f"Required key '{key}' is missing")
    
    # Validar tipos
    types = schema.get('types', {})
    for key, expected_type in types.items():
        if key in config:
            value = config[key]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Key '{key}' has invalid type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
    
    # Validar valores
    validators = schema.get('validators', {})
    for key, validator in validators.items():
        if key in config:
            if not validator(config[key]):
                errors.append(f"Key '{key}' failed validation")
    
    # Verificar keys no permitidas (strict mode)
    if strict:
        allowed_keys = set(schema.get('required', [])) | set(schema.get('types', {}).keys())
        for key in config:
            if key not in allowed_keys:
                errors.append(f"Unexpected key '{key}' in config")
    
    return len(errors) == 0, errors


def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtener valor anidado de configuración usando path.
    
    Args:
        config: Configuración
        path: Path separado por puntos
        default: Valor por defecto
        
    Returns:
        Valor encontrado o default
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    
    return value


def set_nested_config(config: Dict[str, Any], path: str, value: Any) -> None:
    """
    Establecer valor anidado en configuración.
    
    Args:
        config: Configuración
        path: Path separado por puntos
        value: Valor a establecer
    """
    keys = path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def flatten_config(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Aplanar configuración anidada.
    
    Args:
        config: Configuración anidada
        separator: Separador para keys
        
    Returns:
        Configuración aplanada
    """
    result = {}
    
    def _flatten(obj: Any, prefix: str = ''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj
    
    _flatten(config)
    return result


def unflatten_config(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Desaplanar configuración.
    
    Args:
        config: Configuración aplanada
        separator: Separador de keys
        
    Returns:
        Configuración anidada
    """
    result = {}
    
    for key, value in config.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result




