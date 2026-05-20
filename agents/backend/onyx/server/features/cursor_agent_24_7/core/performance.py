"""
Performance Optimizations - Optimizaciones de rendimiento
=========================================================

Utilidades para optimizar performance del sistema.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional
from functools import wraps, lru_cache
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Pool de conexiones para reutilización."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._created = 0
    
    async def acquire(self, factory: Callable) -> Any:
        """Adquirir conexión del pool."""
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            if self._created < self.max_size:
                self._created += 1
                return await factory()
            else:
                # Esperar hasta que haya una disponible
                return await self.pool.get()
    
    async def release(self, connection: Any) -> None:
        """Liberar conexión al pool."""
        try:
            self.pool.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool lleno, descartar conexión
            if hasattr(connection, 'close'):
                await connection.close()


def async_cache(ttl: Optional[float] = None, max_size: int = 128):
    """
    Decorador para cache async con TTL.
    
    Args:
        ttl: Time to live en segundos (None = sin expiración).
        max_size: Tamaño máximo del cache.
    """
    cache = {}
    timestamps = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Crear key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Verificar TTL
            if ttl and key in timestamps:
                if time.time() - timestamps[key] > ttl:
                    cache.pop(key, None)
                    timestamps.pop(key, None)
            
            # Verificar cache
            if key in cache:
                return cache[key]
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            if len(cache) >= max_size:
                # Eliminar el más antiguo
                oldest_key = min(timestamps.items(), key=lambda x: x[1])[0]
                cache.pop(oldest_key, None)
                timestamps.pop(oldest_key, None)
            
            cache[key] = result
            timestamps[key] = time.time()
            
            return result
        
        return wrapper
    return decorator


def batch_process(items: list, batch_size: int = 100, processor: Callable = None):
    """
    Procesar items en batches.
    
    Args:
        items: Lista de items.
        batch_size: Tamaño del batch.
        processor: Función para procesar cada batch.
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if processor:
            yield processor(batch)
        else:
            yield batch


@asynccontextmanager
async def timeout_context(timeout: float):
    """Context manager para timeout."""
    try:
        yield await asyncio.wait_for(asyncio.sleep(0), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout}s")


class RateLimiter:
    """Rate limiter con token bucket."""
    
    def __init__(self, rate: float, capacity: int):
        """
        Inicializar rate limiter.
        
        Args:
            rate: Tokens por segundo.
            capacity: Capacidad máxima.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Adquirir tokens.
        
        Args:
            tokens: Número de tokens a adquirir.
        
        Returns:
            True si se adquirieron exitosamente.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Agregar tokens según el rate
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False


def optimize_json_serialization():
    """Optimizar serialización JSON."""
    try:
        import orjson
        return orjson
    except ImportError:
        import json
        return json


class QueryOptimizer:
    """Optimizador de queries."""
    
    @staticmethod
    def optimize_dynamodb_query(key_condition: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimizar query de DynamoDB.
        
        Args:
            key_condition: Condición de clave.
            filters: Filtros adicionales.
        
        Returns:
            Query optimizado.
        """
        query = {
            "KeyConditionExpression": key_condition
        }
        
        if filters:
            query["FilterExpression"] = filters
        
        # Agregar projection si es necesario
        query["ProjectionExpression"] = "id, #ts, status"
        query["ExpressionAttributeNames"] = {
            "#ts": "timestamp"
        }
        
        return query
    
    @staticmethod
    def optimize_elasticsearch_query(query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimizar query de Elasticsearch.
        
        Args:
            query: Query de texto.
            filters: Filtros adicionales.
        
        Returns:
            Query optimizado.
        """
        es_query = {
            "bool": {
                "should": [
                    {"match": {"_all": {"query": query, "boost": 1.0}}}
                ],
                "must": []
            }
        }
        
        if filters:
            for key, value in filters.items():
                es_query["bool"]["must"].append({"term": {key: value}})
        
        return es_query




