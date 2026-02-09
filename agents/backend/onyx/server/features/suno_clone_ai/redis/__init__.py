"""
Redis Module - Cliente Redis
Cliente Redis, conexiones, y operaciones Redis.

Rol en el Ecosistema IA:
- Almacenamiento rápido, sesiones, caché
- Caché de respuestas LLM, estado de conversaciones, rate limiting
- Estado compartido entre instancias del sistema

Reglas de Importación:
- Puede importar: configs
- NO debe importar: otros módulos del proyecto
- Todos los módulos pueden usar este módulo para caché
"""

from .base import BaseRedis
from .service import RedisService
from .client import RedisClient
from .pool import RedisPool
from .main import (
    get_redis_service,
    get_redis_client,
    cache_get,
    cache_set,
    cache_delete,
    initialize_redis,
)

__all__ = [
    # Clases principales
    "BaseRedis",
    "RedisService",
    "RedisClient",
    "RedisPool",
    # Funciones de acceso rápido
    "get_redis_service",
    "get_redis_client",
    "cache_get",
    "cache_set",
    "cache_delete",
    "initialize_redis",
]

