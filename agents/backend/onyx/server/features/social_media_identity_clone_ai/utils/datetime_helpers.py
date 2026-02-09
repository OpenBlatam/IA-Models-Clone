"""
Helper functions for datetime operations.
Eliminates repetitive datetime.now() and datetime.utcnow() patterns.
"""

from datetime import datetime, timedelta
from typing import Optional


def now() -> datetime:
    """
    Retorna la fecha/hora actual (timezone-aware si es posible).
    
    Returns:
        datetime actual
        
    Usage:
        >>> created_at = now()
    """
    return datetime.now()


def utcnow() -> datetime:
    """
    Retorna la fecha/hora UTC actual.
    
    Returns:
        datetime UTC actual
        
    Usage:
        >>> updated_at = utcnow()
    """
    return datetime.utcnow()


def now_iso() -> str:
    """
    Retorna la fecha/hora actual en formato ISO.
    
    Returns:
        String en formato ISO
        
    Usage:
        >>> timestamp = now_iso()
        '2024-01-15T10:30:45.123456'
    """
    return datetime.now().isoformat()


def utcnow_iso() -> str:
    """
    Retorna la fecha/hora UTC actual en formato ISO.
    
    Returns:
        String en formato ISO UTC
        
    Usage:
        >>> timestamp = utcnow_iso()
        '2024-01-15T10:30:45.123456'
    """
    return datetime.utcnow().isoformat()


def now_timestamp() -> float:
    """
    Retorna el timestamp Unix actual.
    
    Returns:
        Timestamp Unix (segundos desde epoch)
        
    Usage:
        >>> ts = now_timestamp()
        1705315845.123456
    """
    return datetime.now().timestamp()


def days_ago(days: int) -> datetime:
    """
    Retorna la fecha/hora de hace N días.
    
    Args:
        days: Número de días hacia atrás
        
    Returns:
        datetime de hace N días
        
    Usage:
        >>> cutoff = days_ago(30)
    """
    return datetime.utcnow() - timedelta(days=days)


def hours_ago(hours: int) -> datetime:
    """
    Retorna la fecha/hora de hace N horas.
    
    Args:
        hours: Número de horas hacia atrás
        
    Returns:
        datetime de hace N horas
        
    Usage:
        >>> cutoff = hours_ago(24)
    """
    return datetime.utcnow() - timedelta(hours=hours)


def format_timestamp(
    dt: Optional[datetime] = None,
    format_str: str = "%Y%m%d_%H%M%S"
) -> str:
    """
    Formatea un datetime a string.
    
    Args:
        dt: datetime a formatear (default: ahora)
        format_str: Formato a usar (default: "%Y%m%d_%H%M%S")
        
    Returns:
        String formateado
        
    Usage:
        >>> format_timestamp()
        '20240115_103045'
        
        >>> format_timestamp(format_str="%Y-%m-%d")
        '2024-01-15'
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.strftime(format_str)


def start_of_day(dt: Optional[datetime] = None) -> datetime:
    """
    Retorna el inicio del día (00:00:00).
    
    Args:
        dt: datetime base (default: ahora)
        
    Returns:
        datetime al inicio del día
        
    Usage:
        >>> today_start = start_of_day()
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: Optional[datetime] = None) -> datetime:
    """
    Retorna el final del día (23:59:59.999999).
    
    Args:
        dt: datetime base (default: ahora)
        
    Returns:
        datetime al final del día
        
    Usage:
        >>> today_end = end_of_day()
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)








