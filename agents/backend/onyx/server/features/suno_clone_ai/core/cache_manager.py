"""
Sistema de caché para resultados de generación de música
Usa diskcache para almacenamiento persistente
"""

import logging
import hashlib
from typing import Optional, Any
import diskcache as dc
from pathlib import Path
import orjson

try:
    from config.settings import settings
except ImportError:
    from ..config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor de caché para resultados de generación"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or "./cache/suno_clone"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.cache = dc.Cache(cache_dir, size_limit=10**10)  # 10GB limit
        logger.info(f"Cache initialized at: {cache_dir}")
    
    def _generate_key(self, prompt: str, duration: Optional[int] = None, 
                     genre: Optional[str] = None, **kwargs) -> str:
        """Genera una clave única para el caché (optimizado con orjson)"""
        key_data = {
            "prompt": prompt,
            "duration": duration,
            "genre": genre,
            **kwargs
        }
        # Usar orjson para serialización más rápida
        key_str = orjson.dumps(key_data, option=orjson.OPT_SORT_KEYS).decode()
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, duration: Optional[int] = None,
            genre: Optional[str] = None, **kwargs) -> Optional[Any]:
        """Obtiene un resultado del caché"""
        try:
            key = self._generate_key(prompt, duration, genre, **kwargs)
            result = self.cache.get(key)
            if result:
                logger.info(f"Cache hit for key: {key[:16]}...")
                return result
            return None
        except Exception as e:
            # Los errores de caché no deberían detener el proceso
            logger.warning(f"Error getting from cache (non-critical): {e}")
            return None
    
    def set(self, prompt: str, result: Any, duration: Optional[int] = None,
            genre: Optional[str] = None, ttl: Optional[int] = None, **kwargs):
        """Almacena un resultado en el caché"""
        try:
            key = self._generate_key(prompt, duration, genre, **kwargs)
            ttl = ttl or settings.cache_ttl
            self.cache.set(key, result, expire=ttl)
            logger.info(f"Cached result with key: {key[:16]}...")
        except Exception as e:
            # Los errores de caché no deberían detener el proceso
            logger.warning(f"Error setting cache (non-critical): {e}")
    
    def clear(self):
        """Limpia todo el caché"""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def stats(self) -> dict:
        """Obtiene estadísticas del caché"""
        try:
            return {
                "size": len(self.cache),
                "volume": self.cache.volume(),
                "count": self.cache.stats()
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Instancia global
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Obtiene la instancia global del gestor de caché"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

