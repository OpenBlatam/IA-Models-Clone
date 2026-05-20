"""
Time Utils - Utilidades de Tiempo y Fechas
==========================================

Utilidades avanzadas para manejo de tiempo, fechas y zonas horarias.
"""

import logging
from typing import Optional, Union, List
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# Intentar importar dateutil
try:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta
    _has_dateutil = True
except ImportError:
    _has_dateutil = False
    logger.warning("python-dateutil not available. Some features will be limited.")


def parse_datetime(
    date_string: str,
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parsear string de fecha de forma flexible.
    
    Args:
        date_string: String de fecha
        default: Valor por defecto si falla
        
    Returns:
        Datetime parseado o default
    """
    if not date_string:
        return default
    
    if _has_dateutil:
        try:
            return date_parser.parse(date_string)
        except (ValueError, TypeError):
            return default
    else:
        # Fallback básico
        try:
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return default


def to_utc(dt: datetime) -> datetime:
    """
    Convertir datetime a UTC.
    
    Args:
        dt: Datetime a convertir
        
    Returns:
        Datetime en UTC
    """
    if dt.tzinfo is None:
        # Asumir que es local time
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(timezone.utc)


def to_timezone(
    dt: datetime,
    tz: Union[str, timezone]
) -> datetime:
    """
    Convertir datetime a zona horaria específica.
    
    Args:
        dt: Datetime a convertir
        tz: Zona horaria (string o timezone object)
        
    Returns:
        Datetime en zona horaria especificada
    """
    if isinstance(tz, str):
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz)
        except ImportError:
            try:
                import pytz
                tz = pytz.timezone(tz)
            except ImportError:
                logger.warning("zoneinfo or pytz not available")
                return dt
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(tz)


def add_time(
    dt: datetime,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """
    Agregar tiempo a datetime.
    
    Args:
        dt: Datetime base
        years, months, weeks, days, hours, minutes, seconds: Cantidades a agregar
        
    Returns:
        Nuevo datetime
    """
    if _has_dateutil:
        return dt + relativedelta(
            years=years,
            months=months,
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
    else:
        return dt + timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )


def time_ago(dt: datetime) -> str:
    """
    Obtener tiempo relativo desde datetime.
    
    Args:
        dt: Datetime
        
    Returns:
        String con tiempo relativo (ej: "2 hours ago")
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    
    if delta.total_seconds() < 60:
        return "just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days < 30:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    elif delta.days < 365:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"


def is_business_day(dt: datetime) -> bool:
    """
    Verificar si es día hábil (lunes-viernes).
    
    Args:
        dt: Datetime
        
    Returns:
        True si es día hábil
    """
    return dt.weekday() < 5  # 0-4 = lunes-viernes


def next_business_day(dt: datetime) -> datetime:
    """
    Obtener próximo día hábil.
    
    Args:
        dt: Datetime base
        
    Returns:
        Próximo día hábil
    """
    next_day = dt + timedelta(days=1)
    while not is_business_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def get_time_range(
    start: datetime,
    end: datetime,
    interval: timedelta
) -> List[datetime]:
    """
    Obtener rango de tiempos con intervalo.
    
    Args:
        start: Fecha inicio
        end: Fecha fin
        interval: Intervalo entre fechas
        
    Returns:
        Lista de datetimes
    """
    result = []
    current = start
    
    while current <= end:
        result.append(current)
        current += interval
    
    return result


def format_duration(
    seconds: float,
    precision: int = 0
) -> str:
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


def get_week_range(dt: datetime) -> tuple[datetime, datetime]:
    """
    Obtener rango de semana (lunes-domingo).
    
    Args:
        dt: Datetime dentro de la semana
        
    Returns:
        Tupla (inicio_semana, fin_semana)
    """
    # Lunes = 0
    days_since_monday = dt.weekday()
    start = dt - timedelta(days=days_since_monday)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return start, end


def get_month_range(dt: datetime) -> tuple[datetime, datetime]:
    """
    Obtener rango de mes.
    
    Args:
        dt: Datetime dentro del mes
        
    Returns:
        Tupla (inicio_mes, fin_mes)
    """
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    if _has_dateutil:
        end = start + relativedelta(months=1) - timedelta(seconds=1)
    else:
        # Aproximación sin dateutil
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1) - timedelta(seconds=1)
        else:
            end = start.replace(month=start.month + 1) - timedelta(seconds=1)
    
    return start, end




