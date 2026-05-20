"""
Helpers - Utilidades y funciones auxiliares
===========================================

Funciones auxiliares útiles para el agente.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, List, Dict
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorador para reintentar función en caso de error"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def timeout(seconds: float):
    """Decorador para timeout en función async"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """Decorador para medir tiempo de ejecución"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def format_bytes(bytes_size: int) -> str:
    """Formatear bytes en formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncar string a longitud máxima"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_eval(expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Any:
    """Evaluar expresión de forma segura"""
    allowed_names = allowed_names or {}
    
    # Solo permitir operaciones matemáticas básicas
    allowed_chars = set("0123456789+-*/()., ")
    if all(c in allowed_chars for c in expression):
        try:
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except:
            return None
    return None


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Dividir lista en chunks"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """Fusionar múltiples diccionarios"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Obtener variable de entorno de forma segura"""
    import os
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value


def create_timestamp() -> str:
    """Crear timestamp legible"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_duration(duration_str: str) -> float:
    """Parsear duración en formato legible a segundos"""
    duration_str = duration_str.strip().lower()
    
    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('h'):
        return float(duration_str[:-1]) * 3600
    elif duration_str.endswith('d'):
        return float(duration_str[:-1]) * 86400
    else:
        return float(duration_str)


async def run_with_progress(
    func: Callable,
    total_steps: int = 100,
    update_callback: Optional[Callable] = None
):
    """Ejecutar función con barra de progreso"""
    try:
        from tqdm import tqdm
        
        with tqdm(total=total_steps) as pbar:
            if asyncio.iscoroutinefunction(func):
                result = await func(pbar)
            else:
                result = func(pbar)
            pbar.update(total_steps - pbar.n)
            return result
    except ImportError:
        # tqdm no disponible, ejecutar sin progreso
        if asyncio.iscoroutinefunction(func):
            return await func(None)
        else:
            return func(None)


