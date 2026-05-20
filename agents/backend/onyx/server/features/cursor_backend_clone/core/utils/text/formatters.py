"""
Formatters - Utilidades de Formateo
===================================

Utilidades para formatear y presentar datos de manera consistente.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)


def format_bytes(bytes_value: int, precision: int = 2) -> str:
    """
    Formatear bytes en formato legible.
    
    Args:
        bytes_value: Valor en bytes
        precision: Precisión decimal
        
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.{precision}f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.{precision}f} PB"


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Segundos
        precision: Precisión decimal
        
    Returns:
        String formateado (ej: "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.{precision}f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.{precision}f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m {secs:.{precision}f}s"


def format_number(number: float, precision: int = 2) -> str:
    """
    Formatear número con separadores de miles.
    
    Args:
        number: Número a formatear
        precision: Precisión decimal
        
    Returns:
        String formateado (ej: "1,234.56")
    """
    if isinstance(number, (int, float, Decimal)):
        return f"{number:,.{precision}f}"
    return str(number)


def format_percentage(value: float, total: float, precision: int = 1) -> str:
    """
    Formatear porcentaje.
    
    Args:
        value: Valor
        total: Total
        precision: Precisión decimal
        
    Returns:
        String formateado (ej: "45.5%")
    """
    if total == 0:
        return "0%"
    
    percentage = (value / total) * 100
    return f"{percentage:.{precision}f}%"


def format_timestamp(
    timestamp: datetime,
    format_type: str = "iso",
    timezone: Optional[str] = None
) -> str:
    """
    Formatear timestamp.
    
    Args:
        timestamp: Timestamp a formatear
        format_type: Tipo de formato (iso, readable, short)
        timezone: Zona horaria (opcional)
        
    Returns:
        String formateado
    """
    if format_type == "iso":
        return timestamp.isoformat()
    elif format_type == "readable":
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "short":
        return timestamp.strftime("%Y-%m-%d")
    elif format_type == "relative":
        return format_relative_time(timestamp)
    else:
        return timestamp.isoformat()


def format_relative_time(timestamp: datetime) -> str:
    """
    Formatear tiempo relativo (ej: "2 hours ago").
    
    Args:
        timestamp: Timestamp
        
    Returns:
        String formateado
    """
    now = datetime.now()
    delta = now - timestamp
    
    if delta.total_seconds() < 60:
        return "just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days < 30:
        days = delta.days
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif delta.days < 365:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"


def format_table(
    data: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
    max_width: int = 80
) -> str:
    """
    Formatear datos como tabla.
    
    Args:
        data: Lista de diccionarios
        headers: Headers personalizados (opcional)
        max_width: Ancho máximo de columna
        
    Returns:
        String formateado como tabla
    """
    if not data:
        return "No data"
    
    # Obtener headers
    if headers is None:
        headers = list(data[0].keys())
    
    # Calcular anchos de columna
    col_widths = {}
    for header in headers:
        col_widths[header] = min(len(str(header)), max_width)
        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], min(len(value), max_width))
    
    # Construir tabla
    lines = []
    
    # Header
    header_line = " | ".join(
        str(h).ljust(col_widths[h])[:col_widths[h]] for h in headers
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in data:
        row_line = " | ".join(
            str(row.get(h, "")).ljust(col_widths[h])[:col_widths[h]] for h in headers
        )
        lines.append(row_line)
    
    return "\n".join(lines)


def format_json_pretty(data: Any, indent: int = 2) -> str:
    """
    Formatear JSON de forma legible.
    
    Args:
        data: Datos a formatear
        indent: Indentación
        
    Returns:
        String JSON formateado
    """
    import json
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Formatear mensaje de error.
    
    Args:
        error: Excepción
        include_traceback: Si incluir traceback
        
    Returns:
        String formateado
    """
    message = f"{type(error).__name__}: {str(error)}"
    
    if include_traceback:
        import traceback
        message += f"\n{traceback.format_exc()}"
    
    return message


def format_task_status(status: str) -> str:
    """
    Formatear estado de tarea con emoji.
    
    Args:
        status: Estado de la tarea
        
    Returns:
        String formateado con emoji
    """
    status_emojis = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫"
    }
    
    emoji = status_emojis.get(status.lower(), "❓")
    return f"{emoji} {status.upper()}"




