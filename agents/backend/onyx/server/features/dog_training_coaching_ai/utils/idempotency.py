"""
Idempotency Utilities
=====================
Utilidades para garantizar idempotencia en operaciones.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import hashlib
import json
import asyncio

from .cache import get_cached_response, set_cached_response
from .logger import get_logger
from functools import wraps

logger = get_logger(__name__)


class IdempotencyKey:
    """Clave de idempotencia."""
    
    def __init__(self, key: str, ttl: int = 3600):
        """
        Inicializar clave de idempotencia.
        
        Args:
            key: Clave única
            ttl: Time to live en segundos
        """
        self.key = key
        self.ttl = ttl
        self.created_at = datetime.now()
    
    def generate_cache_key(self) -> str:
        """Generar clave de cache."""
        return f"idempotency:{self.key}"


def generate_idempotency_key(
    request_data: Dict[str, Any],
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None
) -> str:
    """
    Generar clave de idempotencia desde datos de request.
    
    Args:
        request_data: Datos de la request
        user_id: ID de usuario (opcional)
        endpoint: Endpoint (opcional)
        
    Returns:
        Clave de idempotencia
    """
    key_parts = []
    
    if user_id:
        key_parts.append(f"user:{user_id}")
    
    if endpoint:
        key_parts.append(f"endpoint:{endpoint}")
    
    # Agregar datos de request (ordenados para consistencia)
    sorted_data = json.dumps(request_data, sort_keys=True, default=str)
    key_parts.append(f"data:{sorted_data}")
    
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()


async def check_idempotency(
    idempotency_key: str,
    ttl: int = 3600
) -> Optional[Dict[str, Any]]:
    """
    Verificar si una request es idempotente.
    
    Args:
        idempotency_key: Clave de idempotencia
        ttl: Time to live en segundos
        
    Returns:
        Respuesta cacheada si existe, None si no
    """
    cache_key = f"idempotency:{idempotency_key}"
    cached = await get_cached_response(cache_key)
    
    if cached:
        logger.info(f"Idempotent request detected: {idempotency_key}")
        return cached
    
    return None


async def store_idempotency_result(
    idempotency_key: str,
    result: Dict[str, Any],
    ttl: int = 3600
):
    """
    Almacenar resultado de request idempotente.
    
    Args:
        idempotency_key: Clave de idempotencia
        result: Resultado a almacenar
        ttl: Time to live en segundos
    """
    cache_key = f"idempotency:{idempotency_key}"
    await set_cached_response(cache_key, result, ttl=ttl)
    logger.debug(f"Stored idempotency result: {idempotency_key}")


def idempotent(ttl: int = 3600, key_generator: Optional[Callable] = None):
    """
    Decorador para hacer funciones idempotentes.
    
    Args:
        ttl: Time to live en segundos
        key_generator: Función para generar clave (opcional)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar clave de idempotencia
            if key_generator:
                idempotency_key = key_generator(*args, **kwargs)
            else:
                # Generar desde args y kwargs
                key_data = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
                }
                idempotency_key = generate_idempotency_key(key_data)
            
            # Verificar si ya existe
            cached_result = await check_idempotency(idempotency_key, ttl)
            if cached_result:
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Almacenar resultado
            await store_idempotency_result(idempotency_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

