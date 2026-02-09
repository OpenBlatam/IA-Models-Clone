"""
DateTime Utilities - Utilidades de fecha y hora
================================================

Utilidades para trabajar con fechas y horas.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import time


def now() -> datetime:
    """Obtener fecha/hora actual con timezone."""
    return datetime.now(timezone.utc)


def parse_datetime(
    date_string: str,
    format: Optional[str] = None
) -> Optional[datetime]:
    """
    Parsear string a datetime.
    
    Args:
        date_string: String de fecha
        format: Formato opcional (si None, intenta varios formatos)
    
    Returns:
        Datetime o None si falla
    """
    if format:
        try:
            return datetime.strptime(date_string, format)
        except ValueError:
            return None
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    return None


def format_datetime(
    dt: datetime,
    format: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Formatear datetime a string.
    
    Args:
        dt: Datetime
        format: Formato de salida
    
    Returns:
        String formateado
    """
    return dt.strftime(format)


def to_timestamp(dt: datetime) -> float:
    """
    Convertir datetime a timestamp.
    
    Args:
        dt: Datetime
    
    Returns:
        Timestamp (segundos desde epoch)
    """
    return dt.timestamp()


def from_timestamp(timestamp: float) -> datetime:
    """
    Convertir timestamp a datetime.
    
    Args:
        timestamp: Timestamp (segundos desde epoch)
    
    Returns:
        Datetime
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def add_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """
    Agregar tiempo a datetime.
    
    Args:
        dt: Datetime base
        days: Días a agregar
        hours: Horas a agregar
        minutes: Minutos a agregar
        seconds: Segundos a agregar
    
    Returns:
        Nuevo datetime
    """
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return dt + delta


def time_ago(dt: datetime) -> str:
    """
    Obtener string "hace X tiempo" desde datetime.
    
    Args:
        dt: Datetime
    
    Returns:
        String descriptivo (ej: "hace 2 horas")
    """
    now_dt = now()
    diff = now_dt - dt
    
    if diff.total_seconds() < 60:
        return "hace unos segundos"
    elif diff.total_seconds() < 3600:
        minutes = int(diff.total_seconds() / 60)
        return f"hace {minutes} minuto{'s' if minutes != 1 else ''}"
    elif diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"hace {hours} hora{'s' if hours != 1 else ''}"
    elif diff.days < 30:
        days = diff.days
        return f"hace {days} día{'s' if days != 1 else ''}"
    elif diff.days < 365:
        months = diff.days // 30
        return f"hace {months} mes{'es' if months != 1 else ''}"
    else:
        years = diff.days // 365
        return f"hace {years} año{'s' if years != 1 else ''}"


def is_business_day(dt: datetime) -> bool:
    """
    Verificar si es día laboral (lunes-viernes).
    
    Args:
        dt: Datetime
    
    Returns:
        True si es día laboral
    """
    return dt.weekday() < 5


def next_business_day(dt: datetime) -> datetime:
    """
    Obtener siguiente día laboral.
    
    Args:
        dt: Datetime base
    
    Returns:
        Siguiente día laboral
    """
    next_day = dt + timedelta(days=1)
    while not is_business_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def time_range(
    start: datetime,
    end: datetime,
    step: timedelta
) -> list:
    """
    Generar rango de fechas.
    
    Args:
        start: Fecha inicio
        end: Fecha fin
        step: Paso (timedelta)
    
    Returns:
        Lista de datetimes
    """
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += step
    return result

