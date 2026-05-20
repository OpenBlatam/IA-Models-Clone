"""
Sistema de caché distribuido
"""

import time
import hashlib
import json
from typing import Optional, Any, Dict
from threading import Lock
import pickle
from ..utils.logger import logger


class DistributedCache:
    """Caché distribuido (preparado para Redis, etc.)"""
    
    def __init__(self, backend: str = "memory"):
        """
        Inicializa el caché distribuido
        
        Args:
            backend: Backend a usar ('memory', 'redis', etc.)
        """
        self.backend = backend
        self.memory_cache: Dict[str, Dict] = {}
        self.lock = Lock()
        self.redis_client = None
        
        if backend == "redis":
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False
                )
            except ImportError:
                logger.warning("Redis no disponible, usando memoria")
                self.backend = "memory"
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Genera clave de cache
        
        Args:
            prefix: Prefijo
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Clave generada
        """
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene valor del cache
        
        Args:
            key: Clave
            
        Returns:
            Valor o None
        """
        if self.backend == "redis" and self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Error obteniendo de Redis: {e}")
                return None
        
        # Memory cache
        with self.lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expires_at"]:
                    return entry["value"]
                else:
                    del self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Guarda valor en cache
        
        Args:
            key: Clave
            value: Valor
            ttl: Tiempo de vida en segundos
        """
        if self.backend == "redis" and self.redis_client:
            try:
                data = pickle.dumps(value)
                self.redis_client.setex(key, ttl, data)
                return
            except Exception as e:
                logger.warning(f"Error guardando en Redis: {e}")
        
        # Memory cache
        with self.lock:
            self.memory_cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl,
                "created_at": time.time()
            }
    
    def delete(self, key: str):
        """
        Elimina valor del cache
        
        Args:
            key: Clave
        """
        if self.backend == "redis" and self.redis_client:
            try:
                self.redis_client.delete(key)
                return
            except Exception as e:
                logger.warning(f"Error eliminando de Redis: {e}")
        
        # Memory cache
        with self.lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
    
    def clear(self):
        """Limpia todo el cache"""
        if self.backend == "redis" and self.redis_client:
            try:
                self.redis_client.flushdb()
                return
            except Exception as e:
                logger.warning(f"Error limpiando Redis: {e}")
        
        # Memory cache
        with self.lock:
            self.memory_cache.clear()
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del cache"""
        if self.backend == "redis" and self.redis_client:
            try:
                info = self.redis_client.info("memory")
                return {
                    "backend": "redis",
                    "used_memory": info.get("used_memory", 0),
                    "keys": self.redis_client.dbsize()
                }
            except Exception:
                pass
        
        # Memory cache
        with self.lock:
            return {
                "backend": "memory",
                "keys": len(self.memory_cache),
                "size_mb": sum(
                    len(str(v["value"]).encode()) for v in self.memory_cache.values()
                ) / 1024 / 1024
            }

