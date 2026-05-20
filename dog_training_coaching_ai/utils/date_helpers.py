"""
Date Helpers
============
Utilidades para manejo de fechas.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import pytz


def now_utc() -> datetime:
    """Obtener fecha/hora actual en UTC."""
    return datetime.now(timezone.utc)


def parse_iso_date(date_string: str) -> datetime:
    """
    Parsear fecha en formato ISO.
    
    Args:
        date_string: String de fecha ISO
        
    Returns:
        Datetime parseado
    """
    # Intentar con timezone
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        # Intentar sin timezone
        return datetime.fromisoformat(date_string)


def format_iso_date(dt: datetime) -> str:
    """
    Formatear datetime a ISO string.
    
    Args:
        dt: Datetime a formatear
        
    Returns:
        String ISO
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.isoformat()


def add_days(dt: datetime, days: int) -> datetime:
    """
    Agregar días a una fecha.
    
    Args:
        dt: Fecha base
        days: Días a agregar (puede ser negativo)
        
    Returns:
        Nueva fecha
    """
    return dt + timedelta(days=days)


def add_hours(dt: datetime, hours: int) -> datetime:
    """
    Agregar horas a una fecha.
    
    Args:
        dt: Fecha base
        hours: Horas a agregar (puede ser negativo)
        
    Returns:
        Nueva fecha
    """
    return dt + timedelta(hours=hours)


def days_between(start: datetime, end: datetime) -> int:
    """
    Calcular días entre dos fechas.
    
    Args:
        start: Fecha de inicio
        end: Fecha de fin
        
    Returns:
        Número de días
    """
    return (end - start).days


def is_weekend(dt: datetime) -> bool:
    """
    Verificar si una fecha es fin de semana.
    
    Args:
        dt: Fecha a verificar
        
    Returns:
        True si es fin de semana
    """
    return dt.weekday() >= 5


def start_of_day(dt: datetime) -> datetime:
    """
    Obtener inicio del día.
    
    Args:
        dt: Fecha
        
    Returns:
        Fecha al inicio del día
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: datetime) -> datetime:
    """
    Obtener fin del día.
    
    Args:
        dt: Fecha
        
    Returns:
        Fecha al fin del día
    """
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_timezone_offset(tz_name: str) -> Optional[timedelta]:
    """
    Obtener offset de timezone.
    
    Args:
        tz_name: Nombre del timezone
        
    Returns:
        Offset o None si no se encuentra
    """
    try:
        tz = pytz.timezone(tz_name)
        return tz.utcoffset(datetime.now())
    except pytz.exceptions.UnknownTimeZoneError:
        return None

