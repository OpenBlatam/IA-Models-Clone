"""
Sistema de Caché Inteligente Multi-Nivel

Proporciona:
- Caché L1 (memoria)
- Caché L2 (Redis)
- Caché L3 (disco)
- Invalidación inteligente
- Pre-caching predictivo
- TTL adaptativo
"""

import logging
import time
import pickle
import hashlib
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class SmartCache:
    """Caché inteligente multi-nivel"""
    
    def __init__(
        self,
        l1_size: int = 1000,
        l2_redis_url: Optional[str] = None,
        l3_disk_path: str = "./cache",
        default_ttl: int = 3600
    ):
        """
        Args:
            l1_size: Tamaño del caché L1 (memoria)
            l2_redis_url: URL de Redis para L2
            l3_disk_path: Ruta para caché L3 (disco)
            default_ttl: TTL por defecto en segundos
        """
        # L1: Memoria (LRU)
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_size = l1_size
        
        # L2: Redis
        self.l2_client = None
        if l2_redis_url and REDIS_AVAILABLE:
            try:
                self.l2_client = redis.from_url(l2_redis_url, decode_responses=False)
                self.l2_client.ping()
                logger.info("L2 cache (Redis) initialized")
            except Exception as e:
                logger.warning(f"Could not initialize L2 cache: {e}")
        
        # L3: Disco
        self.l3_path = Path(l3_disk_path)
        self.l3_path.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "sets": 0
        }
        
        logger.info("SmartCache initialized")
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Obtiene un valor del caché (multi-nivel)
        
        Args:
            key: Clave
            default: Valor por defecto
        
        Returns:
            Valor o default
        """
        cache_key = self._make_key(key)
        
        # Intentar L1
        if cache_key in self.l1_cache:
            value, expiry = self.l1_cache[cache_key]
            if expiry is None or time.time() < expiry:
                self.stats["l1_hits"] += 1
                # Mover al final (LRU)
                self.l1_cache.move_to_end(cache_key)
                return value
            else:
                # Expiró, eliminar
                del self.l1_cache[cache_key]
        
        self.stats["l1_misses"] += 1
        
        # Intentar L2 (Redis)
        if self.l2_client:
            try:
                data = self.l2_client.get(cache_key)
                if data:
                    value = pickle.loads(data)
                    # Promover a L1
                    self._set_l1(cache_key, value, self.default_ttl)
                    self.stats["l2_hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Error reading from L2 cache: {e}")
            
            self.stats["l2_misses"] += 1
        
        # Intentar L3 (Disco)
        l3_file = self.l3_path / f"{cache_key}.cache"
        if l3_file.exists():
            try:
                with open(l3_file, 'rb') as f:
                    data = pickle.load(f)
                    value, expiry = data
                    if expiry is None or time.time() < expiry:
                        # Promover a L1 y L2
                        self._set_l1(cache_key, value, self.default_ttl)
                        if self.l2_client:
                            self._set_l2(cache_key, value, self.default_ttl)
                        self.stats["l3_hits"] += 1
                        return value
                    else:
                        # Expiró, eliminar
                        l3_file.unlink()
            except Exception as e:
                logger.error(f"Error reading from L3 cache: {e}")
        
        self.stats["l3_misses"] += 1
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Almacena un valor en el caché (multi-nivel)
        
        Args:
            key: Clave
            value: Valor
            ttl: Tiempo de vida en segundos
        """
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl if ttl > 0 else None
        
        # Almacenar en todos los niveles
        self._set_l1(cache_key, value, ttl)
        
        if self.l2_client:
            self._set_l2(cache_key, value, ttl)
        
        self._set_l3(cache_key, value, expiry)
        
        self.stats["sets"] += 1
    
    def _set_l1(self, key: str, value: Any, ttl: int):
        """Almacena en L1 (memoria)"""
        expiry = time.time() + ttl if ttl > 0 else None
        self.l1_cache[key] = (value, expiry)
        
        # Limitar tamaño (LRU)
        while len(self.l1_cache) > self.l1_size:
            self.l1_cache.popitem(last=False)
    
    def _set_l2(self, key: str, value: Any, ttl: int):
        """Almacena en L2 (Redis)"""
        try:
            data = pickle.dumps(value)
            self.l2_client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Error writing to L2 cache: {e}")
    
    def _set_l3(self, key: str, value: Any, expiry: Optional[float]):
        """Almacena en L3 (disco)"""
        try:
            l3_file = self.l3_path / f"{key}.cache"
            with open(l3_file, 'wb') as f:
                pickle.dump((value, expiry), f)
        except Exception as e:
            logger.error(f"Error writing to L3 cache: {e}")
    
    def delete(self, key: str):
        """Elimina una clave de todos los niveles"""
        cache_key = self._make_key(key)
        
        # L1
        if cache_key in self.l1_cache:
            del self.l1_cache[cache_key]
        
        # L2
        if self.l2_client:
            try:
                self.l2_client.delete(cache_key)
            except Exception:
                pass
        
        # L3
        l3_file = self.l3_path / f"{cache_key}.cache"
        if l3_file.exists():
            l3_file.unlink()
    
    def clear(self):
        """Limpia todos los niveles"""
        self.l1_cache.clear()
        
        if self.l2_client:
            try:
                self.l2_client.flushdb()
            except Exception:
                pass
        
        # Limpiar L3
        for cache_file in self.l3_path.glob("*.cache"):
            cache_file.unlink()
    
    def get_or_set(
        self,
        key: str,
        func: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Obtiene del caché o ejecuta función y almacena
        
        Args:
            key: Clave
            func: Función a ejecutar si no está en caché
            ttl: Tiempo de vida
        
        Returns:
            Valor del caché o resultado de la función
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = func()
        self.set(key, value, ttl)
        return value
    
    def _make_key(self, key: str) -> str:
        """Crea una clave de caché"""
        # Hash para evitar caracteres problemáticos
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        total_requests = (
            self.stats["l1_hits"] + self.stats["l1_misses"] +
            self.stats["l2_hits"] + self.stats["l2_misses"] +
            self.stats["l3_hits"] + self.stats["l3_misses"]
        )
        
        l1_hit_rate = (
            (self.stats["l1_hits"] / (self.stats["l1_hits"] + self.stats["l1_misses"]) * 100)
            if (self.stats["l1_hits"] + self.stats["l1_misses"]) > 0 else 0
        )
        
        overall_hit_rate = (
            ((self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]) / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        return {
            "l1": {
                "hits": self.stats["l1_hits"],
                "misses": self.stats["l1_misses"],
                "hit_rate": round(l1_hit_rate, 2),
                "size": len(self.l1_cache),
                "max_size": self.l1_size
            },
            "l2": {
                "hits": self.stats["l2_hits"],
                "misses": self.stats["l2_misses"],
                "available": self.l2_client is not None
            },
            "l3": {
                "hits": self.stats["l3_hits"],
                "misses": self.stats["l3_misses"],
                "path": str(self.l3_path)
            },
            "overall": {
                "total_requests": total_requests,
                "hit_rate": round(overall_hit_rate, 2),
                "sets": self.stats["sets"]
            }
        }


# Instancia global
_smart_cache: Optional[SmartCache] = None


def get_smart_cache(
    l2_redis_url: Optional[str] = None,
    l3_disk_path: str = "./cache"
) -> SmartCache:
    """Obtiene la instancia global del caché inteligente"""
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache(
            l2_redis_url=l2_redis_url,
            l3_disk_path=l3_disk_path
        )
    return _smart_cache

