"""
Cache Distribuido - Sistema de cache distribuido con Redis.
"""

import json
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DistributedCache:
    """Cache distribuido con soporte para Redis y fallback en memoria."""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        """
        Inicializar cache distribuido.
        
        Args:
            redis_url: URL de Redis (opcional)
            default_ttl: TTL por defecto en segundos
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl
        self.redis_client = None
        self.fallback_cache: Dict[str, Dict[str, Any]] = {}
        self.use_redis = False
        
        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Cache distribuido: Redis conectado")
            except ImportError:
                logger.warning("Redis no disponible, usando cache en memoria")
            except Exception as e:
                logger.warning(f"Error conectando a Redis: {e}, usando fallback")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave
            
        Returns:
            Valor o None si no existe
        """
        try:
            if self.use_redis and self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Fallback en memoria
                cached = self.fallback_cache.get(key)
                if cached:
                    # Verificar expiración
                    if datetime.now() < cached["expires_at"]:
                        return cached["value"]
                    else:
                        del self.fallback_cache[key]
        except Exception as e:
            logger.error(f"Error obteniendo del cache: {e}", exc_info=True)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Establecer valor en cache.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL en segundos (opcional)
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, default=str)
            
            if self.use_redis and self.redis_client:
                self.redis_client.setex(key, ttl, serialized)
            else:
                # Fallback en memoria
                self.fallback_cache[key] = {
                    "value": value,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
                # Limpiar entradas expiradas periódicamente
                if len(self.fallback_cache) > 10000:
                    self._cleanup_expired()
            
            return True
        except Exception as e:
            logger.error(f"Error guardando en cache: {e}", exc_info=True)
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Eliminar valor del cache.
        
        Args:
            key: Clave
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            if self.use_redis and self.redis_client:
                self.redis_client.delete(key)
            else:
                self.fallback_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Error eliminando del cache: {e}", exc_info=True)
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Limpiar cache.
        
        Args:
            pattern: Patrón para limpiar (opcional)
            
        Returns:
            Número de claves eliminadas
        """
        try:
            if self.use_redis and self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        return self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
                    return -1  # Indica que se limpió todo
            else:
                if pattern:
                    # Limpieza por patrón en memoria (simplificado)
                    keys_to_delete = [k for k in self.fallback_cache.keys() if pattern in k]
                    for key in keys_to_delete:
                        del self.fallback_cache[key]
                    return len(keys_to_delete)
                else:
                    count = len(self.fallback_cache)
                    self.fallback_cache.clear()
                    return count
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}", exc_info=True)
            return 0
    
    def _cleanup_expired(self) -> None:
        """Limpiar entradas expiradas del fallback cache."""
        now = datetime.now()
        expired = [
            key for key, cached in self.fallback_cache.items()
            if now >= cached["expires_at"]
        ]
        for key in expired:
            del self.fallback_cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        stats = {
            "backend": "redis" if self.use_redis else "memory",
            "fallback_size": len(self.fallback_cache)
        }
        
        if self.use_redis and self.redis_client:
            try:
                info = self.redis_client.info("stats")
                stats.update({
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "total_keys": self.redis_client.dbsize()
                })
            except Exception:
                pass
        
        return stats
    
    def close(self) -> None:
        """Cerrar conexión a Redis."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass



