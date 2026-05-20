"""
Cache Manager
=============

Sistema de cache optimizado con LRU para respuestas similares.
"""

import hashlib
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor de cache optimizado con LRU en memoria."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        """
        Inicializar cache manager.
        
        Args:
            ttl_seconds: Tiempo de vida en segundos (default: 1 hora)
            max_size: Tamaño máximo de cache (número de entradas)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._hits = 0
        self._misses = 0
        self._logger = logger
    
    def _generate_key(self, text: str, category: str, **kwargs) -> str:
        """
        Generar clave de cache optimizada.
        
        Args:
            text: Texto del problema
            category: Categoría
            **kwargs: Otros parámetros
        
        Returns:
            Clave hash
        """
        # Normalizar texto más agresivamente para mejor cache hit rate
        normalized_text = ' '.join(text.lower().strip()[:300].split())  # Normalizar espacios
        
        # Crear string para hash (más eficiente)
        key_parts = [normalized_text, category]
        if kwargs:
            key_parts.append(json.dumps(kwargs, sort_keys=True, separators=(',', ':')))
        
        key_string = ':'.join(key_parts)
        
        # Usar SHA256 para mejor distribución (más rápido que MD5 en Python moderno)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(
        self,
        text: str,
        category: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener valor del cache con LRU.
        
        Args:
            text: Texto del problema
            category: Categoría
            **kwargs: Otros parámetros
        
        Returns:
            Valor cacheado o None
        """
        key = self._generate_key(text, category, **kwargs)
        
        if key not in self.cache:
            self._misses += 1
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if datetime.now() > entry['expires_at']:
            del self.cache[key]
            self._misses += 1
            return None
        
        # Mover al final (LRU)
        self.cache.move_to_end(key)
        self._hits += 1
        self._logger.debug(f"Cache hit para key: {key[:8]}...")
        return entry['value']
    
    def set(
        self,
        text: str,
        category: str,
        value: Dict[str, Any],
        **kwargs
    ):
        """
        Guardar valor en cache con LRU.
        
        Args:
            text: Texto del problema
            category: Categoría
            value: Valor a cachear
            **kwargs: Otros parámetros
        """
        key = self._generate_key(text, category, **kwargs)
        
        # Limpiar cache si está lleno (LRU - eliminar el más antiguo primero)
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
            
            # Si aún está lleno, eliminar el más antiguo (LRU)
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        
        expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        self.cache[key] = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': expires_at
        }
        
        # Mover al final (más reciente)
        self.cache.move_to_end(key)
        
        self._logger.debug(f"Cache set para key: {key[:8]}...")
    
    def _cleanup_expired(self):
        """Limpiar entradas expiradas."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._logger.debug(f"Limpiadas {len(expired_keys)} entradas expiradas del cache")
    
    def clear(self):
        """Limpiar todo el cache."""
        self.cache.clear()
        self._logger.info("Cache limpiado")
    
    def stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        self._cleanup_expired()
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests,
            "entries": [
                {
                    "key": key[:8] + "...",
                    "created_at": entry['created_at'].isoformat(),
                    "expires_at": entry['expires_at'].isoformat()
                }
                for key, entry in list(self.cache.items())[:10]
            ]
        }


# Instancia global del cache
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Obtener instancia global del cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance




