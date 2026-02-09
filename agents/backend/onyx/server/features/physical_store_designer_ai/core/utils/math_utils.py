"""
Math Utils

Utilities for math utils.
"""

from typing import Any

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """División segura que evita ZeroDivisionError"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def parse_bool(value: Any) -> bool:
    """Parsear valor a booleano de forma segura"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on", "enabled")
    return bool(value)

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Limitar valor entre min y max"""
    return max(min_value, min(max_value, value))

def calculate_percentage(part: float, total: float, default: float = 0.0) -> float:
    """Calcular porcentaje de forma segura"""
    if total == 0:
        return default
    return (part / total) * 100.0

def normalize_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Normalizar porcentaje entre min y max"""
    return clamp(value, min_val, max_val)

