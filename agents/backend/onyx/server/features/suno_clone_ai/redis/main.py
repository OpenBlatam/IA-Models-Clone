"""
Redis Main - Funciones base y entry points del módulo de Redis

Rol en el Ecosistema IA:
- Almacenamiento rápido, sesiones, caché
- Caché de respuestas LLM, estado de conversaciones, rate limiting
- Estado compartido entre instancias del sistema
"""

from typing import Optional, Any
from .service import RedisService
from .client import RedisClient
from .pool import RedisPool
from configs.main import get_settings


# Instancia global del servicio
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """
    Obtiene la instancia global del servicio de Redis.
    
    Returns:
        RedisService: Servicio de Redis
    """
    global _redis_service
    if _redis_service is None:
        settings = get_settings()
        _redis_service = RedisService(settings)
    return _redis_service


def get_redis_client() -> RedisClient:
    """
    Obtiene un cliente Redis directo.
    
    Returns:
        RedisClient: Cliente Redis
    """
    settings = get_settings()
    return RedisClient(settings.redis_url)


def cache_get(key: str) -> Optional[Any]:
    """
    Obtiene un valor del caché.
    
    Args:
        key: Clave del caché
        
    Returns:
        Valor del caché o None
    """
    service = get_redis_service()
    return service.get(key)


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """
    Establece un valor en el caché.
    
    Args:
        key: Clave del caché
        value: Valor a guardar
        ttl: Tiempo de vida en segundos (opcional)
        
    Returns:
        True si se guardó correctamente
    """
    service = get_redis_service()
    return service.set(key, value, ttl)


def cache_delete(key: str) -> bool:
    """
    Elimina un valor del caché.
    
    Args:
        key: Clave del caché
        
    Returns:
        True si se eliminó correctamente
    """
    service = get_redis_service()
    return service.delete(key)


def initialize_redis() -> RedisService:
    """
    Inicializa el sistema de Redis.
    
    Returns:
        RedisService: Servicio inicializado
    """
    return get_redis_service()

