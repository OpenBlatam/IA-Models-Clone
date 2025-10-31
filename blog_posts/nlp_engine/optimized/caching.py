from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import time
import asyncio
import hashlib
import threading
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
    import aioredis
    import cachetools
            import json
            import json
from typing import Any, List, Dict, Optional
import logging
"""
🚀 ULTRA-FAST CACHING - Multi-Level Performance
===============================================

Sistema de cache ultra-optimizado con múltiples niveles.
"""


# Cache libraries
try:
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

try:
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False


class CacheLevel(Enum):
    """Niveles de cache."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"


@dataclass
class CacheMetrics:
    """Métricas de cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    hit_ratio: float = 0.0
    avg_hit_time_ms: float = 0.0


class UltraFastCache:
    """
    🚀 Sistema de cache multi-nivel ultra-optimizado.
    
    Features:
    - L1: In-memory ultra-rápido
    - L2: Redis distribuido
    - Async operations
    - Performance monitoring
    """
    
    def __init__(
        self,
        l1_size: int = 10000,
        l1_ttl: int = 300,
        redis_url: str = "redis://localhost:6379",
        enable_metrics: bool = True
    ):
        
    """__init__ function."""
self.l1_size = l1_size
        self.l1_ttl = l1_ttl
        self.redis_url = redis_url
        self.enable_metrics = enable_metrics
        
        # Cache layers
        self.l1_cache = None
        self.l2_cache = None
        
        # Thread safety
        self.l1_lock = threading.RLock()
        
        # Métricas
        self.metrics = {
            CacheLevel.L1_MEMORY: CacheMetrics(),
            CacheLevel.L2_REDIS: CacheMetrics()
        }
        
        # Inicializar
        self._initialize_l1()
    
    def _initialize_l1(self) -> Any:
        """Inicializar L1 cache."""
        if CACHETOOLS_AVAILABLE:
            self.l1_cache = cachetools.TTLCache(
                maxsize=self.l1_size,
                ttl=self.l1_ttl
            )
        else:
            self.l1_cache = {}
    
    async def initialize_l2(self) -> Any:
        """Inicializar L2 cache (Redis)."""
        if AIOREDIS_AVAILABLE:
            try:
                self.l2_cache = await aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_keepalive=True
                )
                await self.l2_cache.ping()
                print("✅ Redis L2 cache connected")
            except Exception as e:
                print(f"⚠️  Redis L2 cache failed: {e}")
                self.l2_cache = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache multi-nivel."""
        start_time = time.perf_counter()
        
        # L1 Cache (memoria)
        value = self._get_l1(key)
        if value is not None:
            self._update_metrics(CacheLevel.L1_MEMORY, True, start_time)
            return value
        
        # L2 Cache (Redis)
        if self.l2_cache:
            value = await self._get_l2(key)
            if value is not None:
                # Promocionar a L1
                self._set_l1(key, value)
                self._update_metrics(CacheLevel.L2_REDIS, True, start_time)
                return value
        
        # Cache miss
        self._update_metrics(CacheLevel.L1_MEMORY, False, start_time)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en cache multi-nivel."""
        success = True
        
        # Set en L1
        self._set_l1(key, value)
        
        # Set en L2
        if self.l2_cache:
            success = await self._set_l2(key, value, ttl)
        
        # Actualizar métricas
        self.metrics[CacheLevel.L1_MEMORY].sets += 1
        if self.l2_cache:
            self.metrics[CacheLevel.L2_REDIS].sets += 1
        
        return success
    
    def _get_l1(self, key: str) -> Optional[Any]:
        """Get de L1 cache."""
        try:
            with self.l1_lock:
                return self.l1_cache.get(key)
        except:
            return None
    
    def _set_l1(self, key: str, value: Any) -> bool:
        """Set en L1 cache."""
        try:
            with self.l1_lock:
                self.l1_cache[key] = value
            return True
        except:
            return False
    
    async def _get_l2(self, key: str) -> Optional[Any]:
        """Get de L2 cache (Redis)."""
        try:
            data = await self.l2_cache.get(key)
            if data:
                return json.loads(data)
        except:
            pass
        return None
    
    async def _set_l2(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set en L2 cache (Redis)."""
        try:
            data = json.dumps(value, ensure_ascii=False)
            
            if ttl:
                await self.l2_cache.setex(key, ttl, data)
            else:
                await self.l2_cache.set(key, data)
            return True
        except:
            return False
    
    def _update_metrics(self, level: CacheLevel, hit: bool, start_time: float):
        """Actualizar métricas."""
        if not self.enable_metrics:
            return
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics = self.metrics[level]
        
        if hit:
            metrics.hits += 1
            metrics.avg_hit_time_ms = (
                (metrics.avg_hit_time_ms * (metrics.hits - 1) + duration_ms) / metrics.hits
            )
        else:
            metrics.misses += 1
        
        total = metrics.hits + metrics.misses
        metrics.hit_ratio = metrics.hits / total if total > 0 else 0.0
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generar clave de cache determinística."""
        key_parts = []
        
        for arg in args:
            key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        stats = {
            "levels": {},
            "library_availability": {
                "aioredis": AIOREDIS_AVAILABLE,
                "cachetools": CACHETOOLS_AVAILABLE
            }
        }
        
        for level, metrics in self.metrics.items():
            total_ops = metrics.hits + metrics.misses
            stats["levels"][level.value] = {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "hit_ratio": metrics.hit_ratio,
                "total_operations": total_ops,
                "avg_hit_time_ms": metrics.avg_hit_time_ms
            }
        
        return stats


# Global cache instance
_global_cache: Optional[UltraFastCache] = None

def get_optimized_cache() -> UltraFastCache:
    """Obtener instancia global del cache optimizado."""
    global _global_cache
    if _global_cache is None:
        _global_cache = UltraFastCache()
    return _global_cache 