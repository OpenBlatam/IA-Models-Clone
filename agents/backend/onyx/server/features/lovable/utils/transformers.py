"""
Data transformation utilities.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def transform_dict(
    data: Dict[str, Any],
    mappings: Dict[str, str],
    transformers: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Transform a dictionary by renaming keys and applying transformers.
    
    Args:
        data: Source dictionary
        mappings: Dictionary mapping old keys to new keys
        transformers: Optional dictionary of key -> transformer function
        
    Returns:
        Transformed dictionary
    """
    result = {}
    transformers = transformers or {}
    
    for old_key, new_key in mappings.items():
        if old_key in data:
            value = data[old_key]
            
            # Apply transformer if exists
            if new_key in transformers:
                try:
                    value = transformers[new_key](value)
                except Exception as e:
                    logger.warning(f"Transformer failed for {new_key}: {e}")
                    continue
            
            result[new_key] = value
    
    return result


def normalize_datetime(value: Any) -> Optional[str]:
    """
    Normalize datetime to ISO format string.
    
    Args:
        value: Datetime value
        
    Returns:
        ISO format string or None
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value.isoformat()
    
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.isoformat()
        except Exception:
            return value
    
    return str(value)


def normalize_list(value: Any) -> List[Any]:
    """
    Normalize value to a list.
    
    Args:
        value: Value to normalize
        
    Returns:
        List
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        # Try to parse comma-separated string
        return [item.strip() for item in value.split(",") if item.strip()]
    
    return [value]


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        data: Nested dictionary
        separator: Separator for nested keys
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    result = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator=separator, prefix=new_key))
        else:
            result[new_key] = value
    
    return result


def unflatten_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot notation keys.
    
    Args:
        data: Flattened dictionary
        separator: Separator used in keys
        
    Returns:
        Nested dictionary
    """
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


def rename_keys(
    data: Dict[str, Any],
    key_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Rename keys in a dictionary based on a mapping.
    
    Args:
        data: Source dictionary
        key_map: Dictionary mapping old keys to new keys
        
    Returns:
        Dictionary with renamed keys
    """
    result = data.copy()
    
    for old_key, new_key in key_map.items():
        if old_key in result:
            result[new_key] = result.pop(old_key)
            
    return result


def normalize_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize types in a dictionary (e.g. convert strings to numbers where appropriate).
    
    Args:
        data: Source dictionary
        
    Returns:
        Dictionary with normalized types
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Try integer
            if value.isdigit():
                result[key] = int(value)
                continue
                
            # Try float
            try:
                float_val = float(value)
                if "." in value:
                    result[key] = float_val
                    continue
            except ValueError:
                pass
                
            # Try boolean
            if value.lower() in ("true", "yes", "on"):
                result[key] = True
                continue
            if value.lower() in ("false", "no", "off"):
                result[key] = False
                continue
                
        result[key] = value
        
    return result






