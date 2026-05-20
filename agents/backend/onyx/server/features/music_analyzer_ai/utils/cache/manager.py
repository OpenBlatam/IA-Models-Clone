"""
Cache Manager Module

Main cache manager class.
"""

from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import logging
import hashlib

from ...config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor de cache para análisis musical"""
    
    def __init__(self):
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_enabled = settings.CACHE_ENABLED
        self.cache_ttl = settings.CACHE_TTL
        self.max_memory_items = 100  # Límite de items en memoria
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Genera una clave única para el cache"""
        key_string = f"{prefix}:{identifier}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        """Obtiene un valor del cache"""
        if not self.cache_enabled:
            return None
        
        key = self._generate_key(prefix, identifier)
        
        # Buscar en memoria
        if key in self.memory_cache:
            item = self.memory_cache[key]
            
            # Verificar expiración
            if datetime.now() < item.get("expires_at", datetime.min):
                logger.debug(f"Cache hit: {prefix}:{identifier}")
                return item.get("data")
            else:
                # Eliminar si expiró
                del self.memory_cache[key]
                logger.debug(f"Cache expired: {prefix}:{identifier}")
        
        return None
    
    def delete(self, prefix: str, identifier: str) -> None:
        """Elimina un valor del cache"""
        key = self._generate_key(prefix, identifier)
        if key in self.memory_cache:
            del self.memory_cache[key]
            logger.debug(f"Cache deleted: {prefix}:{identifier}")
    
    def clear(self, prefix: Optional[str] = None) -> None:
        """Limpia el cache"""
        if prefix:
            # Eliminar solo items con el prefijo
            keys_to_delete = [
                k for k in self.memory_cache.keys()
                if k.startswith(self._generate_key(prefix, "").split(":")[0])
            ]
            for key in keys_to_delete:
                del self.memory_cache[key]
        else:
            # Limpiar todo
            self.memory_cache.clear()
        
        logger.info(f"Cache cleared: {prefix or 'all'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        # Cleanup will be called by _cleanup_expired method
        return {
            "enabled": self.cache_enabled,
            "items_count": len(self.memory_cache),
            "max_items": self.max_memory_items,
            "ttl": self.cache_ttl
        }

