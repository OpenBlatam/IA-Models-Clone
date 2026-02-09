"""
Cache Service - Servicio de Cache
==================================

Sistema de cache para mejorar performance.
"""

import logging
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class CacheService:
    """Servicio de cache"""
    
    def __init__(self, backend: str = "memory"):
        """
        Inicializar servicio de cache
        
        Args:
            backend: Backend de cache (memory, redis)
        """
        self.backend = backend
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        if backend == "redis":
            try:
                import os
                import redis
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=0
                )
                logger.info("Cache Service inicializado con Redis")
            except ImportError:
                logger.warning("Redis no disponible, usando memoria")
                self.backend = "memory"
        else:
            logger.info("Cache Service inicializado con memoria")
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generar clave de cache"""
        key_str = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache
        
        Args:
            key: Clave del cache
            
        Returns:
            Valor cacheado o None
        """
        if self.backend == "redis":
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Error obteniendo de Redis: {e}")
                return None
        else:
            cached = self.memory_cache.get(key)
            if cached:
                # Verificar expiración
                if cached.get("expires_at") and datetime.now() > cached["expires_at"]:
                    del self.memory_cache[key]
                    return None
                return cached.get("value")
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Guardar valor en cache
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: Time to live en segundos
        """
        if self.backend == "redis":
            try:
                json_value = json.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, json_value)
                else:
                    self.redis_client.set(key, json_value)
            except Exception as e:
                logger.error(f"Error guardando en Redis: {e}")
        else:
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self.memory_cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }
    
    def delete(self, key: str):
        """
        Eliminar del cache
        
        Args:
            key: Clave a eliminar
        """
        if self.backend == "redis":
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Error eliminando de Redis: {e}")
        else:
            self.memory_cache.pop(key, None)
    
    def clear(self, pattern: Optional[str] = None):
        """
        Limpiar cache
        
        Args:
            pattern: Patrón para limpiar (solo Redis)
        """
        if self.backend == "redis":
            try:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Error limpiando Redis: {e}")
        else:
            if pattern:
                # Limpiar por patrón en memoria
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()
    
    def get_or_set(
        self,
        key: str,
        func: callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Obtener del cache o ejecutar función y cachear
        
        Args:
            key: Clave del cache
            func: Función a ejecutar si no hay cache
            ttl: Time to live
            
        Returns:
            Valor del cache o resultado de la función
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = func()
        self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del cache
        
        Returns:
            Dict con estadísticas
        """
        if self.backend == "redis":
            try:
                info = self.redis_client.info("stats")
                return {
                    "backend": "redis",
                    "keys": self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            except Exception as e:
                logger.error(f"Error obteniendo stats de Redis: {e}")
                return {"backend": "redis", "error": str(e)}
        else:
            return {
                "backend": "memory",
                "keys": len(self.memory_cache),
                "expired": len([k for k, v in self.memory_cache.items() 
                               if v.get("expires_at") and datetime.now() > v["expires_at"]])
            }
    
    def cleanup_expired(self) -> int:
        """
        Limpiar entradas expiradas del cache
        
        Returns:
            Número de entradas eliminadas
        """
        if self.backend == "redis":
            return 0
        
        expired_keys = [
            k for k, v in self.memory_cache.items()
            if v.get("expires_at") and datetime.now() > v["expires_at"]
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.info(f"Limpiadas {len(expired_keys)} entradas expiradas del cache")
        
        return len(expired_keys)
    
    def exists(self, key: str) -> bool:
        """
        Verificar si una clave existe en el cache
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si existe
        """
        if self.backend == "redis":
            try:
                return self.redis_client.exists(key) > 0
            except Exception as e:
                logger.error(f"Error verificando existencia en Redis: {e}")
                return False
        else:
            cached = self.memory_cache.get(key)
            if cached:
                if cached.get("expires_at") and datetime.now() > cached["expires_at"]:
                    del self.memory_cache[key]
                    return False
                return True
            return False
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Incrementar valor numérico en el cache
        
        Args:
            key: Clave
            amount: Cantidad a incrementar
            
        Returns:
            Nuevo valor
        """
        if self.backend == "redis":
            try:
                return self.redis_client.incrby(key, amount)
            except Exception as e:
                logger.error(f"Error incrementando en Redis: {e}")
                return 0
        else:
            current = self.get(key) or 0
            new_value = current + amount
            self.set(key, new_value)
            return new_value

