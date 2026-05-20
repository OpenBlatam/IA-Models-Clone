"""
Connection Pooling - Pool de conexiones optimizado
==================================================

Pools de conexiones para Redis, HTTP, y otros servicios.
"""

import logging
from typing import Optional
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class RedisConnectionPool:
    """Pool de conexiones Redis optimizado"""
    
    def __init__(self, url: str, max_connections: int = 10, min_connections: int = 2):
        """
        Args:
            url: URL de Redis
            max_connections: Máximo de conexiones
            min_connections: Mínimo de conexiones
        """
        self.url = url
        self.max_connections = max_connections
        self.min_connections = min_connections
        self._pool: deque = deque(maxlen=max_connections)
        self._lock = asyncio.Lock()
        self._created = 0
    
    async def get_connection(self):
        """Obtiene conexión del pool"""
        async with self._lock:
            if self._pool:
                return self._pool.popleft()
            
            # Crear nueva conexión
            if self._created < self.max_connections:
                try:
                    import redis.asyncio as aioredis
                    conn = await aioredis.from_url(self.url)
                    self._created += 1
                    return conn
                except Exception as e:
                    logger.error(f"Error creating Redis connection: {e}")
                    raise
            else:
                # Esperar a que haya una conexión disponible
                # En producción, usar semáforo
                raise RuntimeError("Connection pool exhausted")
    
    async def return_connection(self, conn):
        """Devuelve conexión al pool"""
        async with self._lock:
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
            else:
                # Cerrar conexión si el pool está lleno
                await conn.close()
                self._created -= 1


class HTTPConnectionPool:
    """Pool de conexiones HTTP optimizado"""
    
    def __init__(self, max_connections: int = 100):
        """
        Args:
            max_connections: Máximo de conexiones
        """
        self.max_connections = max_connections
        self._client: Optional[Any] = None
    
    def get_client(self):
        """Obtiene cliente HTTP con connection pooling"""
        if self._client is None:
            import httpx
            limits = httpx.Limits(
                max_keepalive_connections=self.max_connections,
                max_connections=self.max_connections
            )
            self._client = httpx.AsyncClient(limits=limits)
        return self._client
    
    async def close(self):
        """Cierra el cliente"""
        if self._client:
            await self._client.aclose()


class DatabaseConnectionPool:
    """Pool de conexiones de base de datos"""
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        """
        Args:
            connection_string: String de conexión
            pool_size: Tamaño del pool
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._pool = None
    
    async def get_pool(self):
        """Obtiene pool de conexiones"""
        if self._pool is None:
            # Ejemplo con asyncpg (PostgreSQL)
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=self.pool_size
                )
            except ImportError:
                logger.warning("asyncpg not available")
        return self._pool
    
    async def close(self):
        """Cierra el pool"""
        if self._pool:
            await self._pool.close()


def get_connection_pools() -> dict:
    """Obtiene todos los pools de conexiones"""
    return {
        "redis": None,  # Se inicializa cuando se necesita
        "http": HTTPConnectionPool(),
        "database": None  # Se inicializa cuando se necesita
    }















