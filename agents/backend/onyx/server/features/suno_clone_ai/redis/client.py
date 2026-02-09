"""
Redis Client - Cliente Redis principal
"""

from typing import Any, Optional
import redis
import json


class RedisClient:
    """Cliente Redis principal"""

    def __init__(self, redis_url: str):
        """Inicializa el cliente Redis"""
        self.client = redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor por clave"""
        value = self.client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece un valor con opcional TTL"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return self.client.set(key, value, ex=ttl)

    def delete(self, key: str) -> bool:
        """Elimina una clave"""
        return bool(self.client.delete(key))

    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        return bool(self.client.exists(key))

