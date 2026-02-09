"""
Data Utils

Utilities for data utils.
"""

from typing import Any, List, Callable
import time
import random
import uuid
from datetime import datetime

def generate_id(prefix: str = "id") -> str:
    """Generar ID único"""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncar texto a longitud máxima"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def batch_process(items: list[Any], batch_size: int = 10) -> list[list[Any]]:
    """Dividir lista en batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Dividir lista en chunks de tamaño fijo"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """Ejecutar función con retry y backoff exponencial"""
    import time
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise last_exception
    
    raise last_exception

