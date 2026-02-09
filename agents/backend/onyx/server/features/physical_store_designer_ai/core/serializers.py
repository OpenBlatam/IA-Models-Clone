"""
Serialization and data transformation utilities
"""

import json
import base64
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        return super().default(obj)


def serialize_to_json(data: Any, indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
    """Serialize data to JSON string"""
    try:
        return json.dumps(data, cls=JSONEncoder, indent=indent, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        raise ValueError(f"Failed to serialize data to JSON: {e}")


def deserialize_from_json(json_str: str) -> Any:
    """Deserialize JSON string to Python object"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON deserialization failed: {e}")
        raise ValueError(f"Failed to deserialize JSON: {e}")


def serialize_to_base64(data: Any) -> str:
    """Serialize data to base64 encoded JSON string"""
    json_str = serialize_to_json(data)
    return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')


def deserialize_from_base64(base64_str: str) -> Any:
    """Deserialize base64 encoded JSON string"""
    try:
        json_str = base64.b64decode(base64_str.encode('utf-8')).decode('utf-8')
        return deserialize_from_json(json_str)
    except Exception as e:
        logger.error(f"Base64 deserialization failed: {e}")
        raise ValueError(f"Failed to deserialize base64: {e}")


def serialize_dict_keys(data: Dict[str, Any], key_transform: str = "snake_case") -> Dict[str, Any]:
    """Transform dictionary keys"""
    if key_transform == "snake_case":
        return {_to_snake_case(k): v for k, v in data.items()}
    elif key_transform == "camel_case":
        return {_to_camel_case(k): v for k, v in data.items()}
    elif key_transform == "lower":
        return {k.lower(): v for k, v in data.items()}
    elif key_transform == "upper":
        return {k.upper(): v for k, v in data.items()}
    return data


def _to_snake_case(s: str) -> str:
    """Convert string to snake_case"""
    import re
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.lower()


def _to_camel_case(s: str) -> str:
    """Convert string to camelCase"""
    components = s.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])


def flatten_dict(data: Dict[str, Any], separator: str = ".", prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Unflatten dictionary with nested keys"""
    result = {}
    for key, value in data.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        return str(obj)


def normalize_data_types(data: Any) -> Any:
    """Normalize data types for consistency"""
    if isinstance(data, dict):
        return {k: normalize_data_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_data_types(item) for item in data]
    elif isinstance(data, str):
        # Remove extra whitespace
        return ' '.join(data.split())
    elif isinstance(data, (int, float)):
        # Ensure numeric types
        return float(data) if isinstance(data, (int, float)) else data
    return data


def extract_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Extract nested value from dictionary using dot notation"""
    keys = path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested value in dictionary using dot notation"""
    keys = path.split('.')
    d = data
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, later ones override earlier ones"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def deep_merge_dicts(base: Dict[str, Any], *updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries"""
    result = base.copy()
    for update in updates:
        if not isinstance(update, dict):
            continue
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
    return result

