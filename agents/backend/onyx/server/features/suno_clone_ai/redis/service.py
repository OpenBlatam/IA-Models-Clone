"""
Redis Service - Servicio de Redis
"""

from typing import Any, Optional
from .client import RedisClient
from configs.settings import Settings


class RedisService:
    """Servicio para gestionar operaciones Redis"""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el servicio de Redis"""
        self.settings = settings or Settings()
        self.client = RedisClient(self.settings.redis_url)

    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor por clave"""
        return self.client.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece un valor con opcional TTL"""
        return self.client.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Elimina una clave"""
        return self.client.delete(key)

    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        return self.client.exists(key)

