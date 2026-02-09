"""
Json Utils

Utilities for json utils.
"""

import json
from typing import Any

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Cargar JSON de forma segura"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Serializar a JSON de forma segura"""
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default

