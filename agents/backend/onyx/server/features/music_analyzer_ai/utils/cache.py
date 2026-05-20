"""
Sistema de cache para mejorar el rendimiento
"""

import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import logging
from pathlib import Path

from ..config.settings import settings
from .exceptions import CacheException

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
    
    def set(self, prefix: str, identifier: str, data: Any, ttl: Optional[int] = None) -> None:
        """Almacena un valor en el cache"""
        if not self.cache_enabled:
            return
        
        key = self._generate_key(prefix, identifier)
        ttl = ttl or self.cache_ttl
        
        # Limpiar cache si está lleno
        if len(self.memory_cache) >= self.max_memory_items:
            self._cleanup_expired()
            # Si aún está lleno, eliminar el más antiguo
            if len(self.memory_cache) >= self.max_memory_items:
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].get("created_at", datetime.min)
                )
                del self.memory_cache[oldest_key]
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.memory_cache[key] = {
            "data": data,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        
        logger.debug(f"Cache set: {prefix}:{identifier} (expires in {ttl}s)")
    
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
    
    def _cleanup_expired(self) -> None:
        """Elimina items expirados del cache"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self.memory_cache.items()
            if now >= v.get("expires_at", datetime.min)
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        self._cleanup_expired()
        
        return {
            "enabled": self.cache_enabled,
            "items_count": len(self.memory_cache),
            "max_items": self.max_memory_items,
            "ttl": self.cache_ttl
        }


# Instancia global del cache
cache_manager = CacheManager()

