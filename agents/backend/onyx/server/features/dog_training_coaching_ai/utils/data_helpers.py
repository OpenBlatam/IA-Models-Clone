"""
Data Helpers
============
Utilidades para manipulación de datos.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionar múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a fusionar
        
    Returns:
        Diccionario fusionado
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filtrar valores None de un diccionario.
    
    Args:
        data: Diccionario a filtrar
        
    Returns:
        Diccionario sin valores None
    """
    return {k: v for k, v in data.items() if v is not None}


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Dividir lista en chunks.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
        
    Returns:
        Lista de chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def calculate_average(numbers: List[float]) -> Optional[float]:
    """
    Calcular promedio de una lista de números.
    
    Args:
        numbers: Lista de números
        
    Returns:
        Promedio o None si la lista está vacía
    """
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def format_duration(seconds: float) -> str:
    """
    Formatear duración en segundos a string legible.
    
    Args:
        seconds: Segundos
        
    Returns:
        String formateado (ej: "2h 30m 15s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours < 24:
        return f"{hours}h {mins}m {secs}s"
    
    days = int(hours // 24)
    hrs = int(hours % 24)
    
    return f"{days}d {hrs}h {mins}m"


def parse_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """
    Parsear rango de fechas.
    
    Args:
        start_date: Fecha de inicio (ISO format)
        end_date: Fecha de fin (ISO format)
        
    Returns:
        Tupla con fechas parseadas
    """
    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    return start, end

