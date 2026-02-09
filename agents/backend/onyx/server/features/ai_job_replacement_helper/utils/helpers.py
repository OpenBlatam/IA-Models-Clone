"""
Helper utilities
"""

import secrets
import html
from typing import Any


def generate_id(prefix: str = "id") -> str:
    """Generar ID único"""
    return f"{prefix}_{secrets.token_hex(8)}"


def sanitize_input(text: str) -> str:
    """Sanitizar input del usuario"""
    # Escapar HTML
    sanitized = html.escape(text)
    # Remover caracteres de control
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    return sanitized.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncar texto"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get(data: dict, *keys, default: Any = None) -> Any:
    """Obtener valor de diccionario de forma segura"""
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result




