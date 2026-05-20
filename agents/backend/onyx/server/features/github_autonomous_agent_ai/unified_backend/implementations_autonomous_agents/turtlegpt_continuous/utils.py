"""
Utility Functions Module
========================

Funciones de utilidad para el agente continuo.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


def format_context(context: Optional[Dict[str, Any]]) -> str:
    """
    Formatear contexto para prompts.
    
    Args:
        context: Contexto a formatear
        
    Returns:
        String formateado
    """
    if not context:
        return "No hay contexto adicional disponible."
    
    context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    return context_str


def calculate_success_rate(completed: int, failed: int) -> float:
    """
    Calcular tasa de éxito.
    
    Args:
        completed: Número de tareas completadas
        failed: Número de tareas fallidas
        
    Returns:
        Tasa de éxito (0.0 a 1.0)
    """
    total = completed + failed
    if total == 0:
        return 0.0
    return completed / total


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Segundos a formatear
        
    Returns:
        String formateado (ej: "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.1f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Cargar JSON de forma segura.
    
    Args:
        json_str: String JSON
        default: Valor por defecto si falla
        
    Returns:
        Objeto parseado o default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_str[:100]}")
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncar texto a longitud máxima.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combinar múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a combinar
        
    Returns:
        Diccionario combinado (últimos valores ganan)
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtener valor anidado de diccionario usando path.
    
    Args:
        data: Diccionario
        path: Path separado por puntos (ej: "a.b.c")
        default: Valor por defecto
        
    Returns:
        Valor encontrado o default
    """
    keys = path.split(".")
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def validate_priority(priority: int, min_val: int = 1, max_val: int = 10) -> int:
    """
    Validar y normalizar prioridad.
    
    Args:
        priority: Prioridad a validar
        min_val: Valor mínimo
        max_val: Valor máximo
        
    Returns:
        Prioridad validada
    """
    return max(min_val, min(max_val, int(priority)))


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


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Formatear métricas para logging.
    
    Args:
        metrics: Métricas a formatear
        
    Returns:
        String formateado
    """
    lines = ["Metrics:"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.2f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
