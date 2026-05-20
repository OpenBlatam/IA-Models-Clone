"""
Formatting Utilities
=====================
Utilidades para formatear datos.
"""

from typing import Any, Optional
from datetime import datetime
import json


def format_number(value: float, decimals: int = 2, thousands_separator: str = ",") -> str:
    """
    Formatear número con decimales y separador de miles.
    
    Args:
        value: Número a formatear
        decimals: Número de decimales
        thousands_separator: Separador de miles
        
    Returns:
        String formateado
    """
    formatted = f"{value:,.{decimals}f}"
    if thousands_separator != ",":
        formatted = formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", thousands_separator)
    return formatted


def format_bytes(bytes_value: int) -> str:
    """
    Formatear bytes a formato legible.
    
    Args:
        bytes_value: Bytes a formatear
        
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatear porcentaje.
    
    Args:
        value: Valor a formatear (0-100)
        decimals: Número de decimales
        
    Returns:
        String formateado (ej: "45.5%")
    """
    return f"{value:.{decimals}f}%"


def format_duration_human(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Segundos
        
    Returns:
        String formateado (ej: "2 hours 30 minutes")
    """
    if seconds < 60:
        return f"{int(seconds)} second{'s' if seconds != 1 else ''}"
    
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    if hours < 24:
        if mins > 0:
            return f"{hours} hour{'s' if hours != 1 else ''} {mins} minute{'s' if mins != 1 else ''}"
        return f"{hours} hour{'s' if hours != 1 else ''}"
    
    days = int(hours // 24)
    hrs = hours % 24
    if hrs > 0:
        return f"{days} day{'s' if days != 1 else ''} {hrs} hour{'s' if hrs != 1 else ''}"
    return f"{days} day{'s' if days != 1 else ''}"


def format_datetime_human(dt: datetime, include_time: bool = True) -> str:
    """
    Formatear datetime en formato legible.
    
    Args:
        dt: Datetime a formatear
        include_time: Incluir hora
        
    Returns:
        String formateado (ej: "January 15, 2024 at 3:30 PM")
    """
    date_str = dt.strftime("%B %d, %Y")
    
    if include_time:
        time_str = dt.strftime("%I:%M %p")
        return f"{date_str} at {time_str}"
    
    return date_str


def format_json_pretty(data: Any, indent: int = 2) -> str:
    """
    Formatear JSON de forma legible.
    
    Args:
        data: Datos a formatear
        indent: Indentación
        
    Returns:
        String JSON formateado
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


def format_list_human(items: list, conjunction: str = "and") -> str:
    """
    Formatear lista en formato legible.
    
    Args:
        items: Lista de items
        conjunction: Conjunción a usar (and, or, etc.)
        
    Returns:
        String formateado (ej: "apple, banana, and orange")
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return str(items[0])
    
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    
    return f"{', '.join(str(item) for item in items[:-1])}, {conjunction} {items[-1]}"

