"""
Redis Pool - Pool de conexiones Redis
"""

from typing import Optional
import redis
from configs.settings import Settings


class RedisPool:
    """Pool de conexiones Redis"""

    def __init__(self, settings: Optional[Settings] = None, max_connections: int = 50):
        """Inicializa el pool de conexiones"""
        self.settings = settings or Settings()
        self.pool = redis.ConnectionPool.from_url(
            self.settings.redis_url,
            max_connections=max_connections,
            decode_responses=True
        )

    def get_client(self) -> redis.Redis:
        """Obtiene un cliente del pool"""
        return redis.Redis(connection_pool=self.pool)

