"""
Cache para contenido extraído con estadísticas
"""

import hashlib
from typing import Optional, Dict, Any
from cachetools import TTLCache
import logging
from time import time

logger = logging.getLogger(__name__)


class ContentCache:
    """Cache con TTL para contenido extraído"""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Args:
            maxsize: Tamaño máximo del cache
            ttl: Tiempo de vida en segundos (1 hora por defecto)
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0
        self._start_time = time()
    
    def _get_key(self, url: str) -> str:
        """Generar clave de cache desde URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Obtener contenido del cache"""
        key = self._get_key(url)
        result = self.cache.get(key)
        if result:
            self._hits += 1
            logger.debug(f"Cache hit para {url}")
        else:
            self._misses += 1
        return result
    
    def set(self, url: str, content: Dict[str, Any]) -> None:
        """Guardar contenido en cache"""
        key = self._get_key(url)
        self.cache[key] = content
        logger.debug(f"Contenido cacheado para {url}")
    
    def clear(self) -> None:
        """Limpiar cache"""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        self._start_time = time()
        logger.info("Cache limpiado")
    
    def size(self) -> int:
        """Tamaño actual del cache"""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": self.size(),
            "maxsize": self.cache.maxsize,
            "ttl": self.cache.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "uptime_seconds": round(time() - self._start_time, 2)
        }

