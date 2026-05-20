"""
Cache Service - Sistema de caché
=================================

Sistema de caché para mejorar performance.
"""

import logging
import json
from typing import Optional, Any
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CacheService:
    """Servicio de caché"""
    
    def __init__(self, ttl_seconds: int = 300):
        """Inicializar servicio de caché"""
        self.cache: dict = {}
        self.default_ttl = ttl_seconds
        logger.info("CacheService initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en caché"""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
        }
        
        return True
    
    def delete(self, key: str) -> bool:
        """Eliminar del caché"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Limpiar todo el caché"""
        self.cache.clear()
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del caché"""
        return {
            "total_entries": len(self.cache),
            "default_ttl": self.default_ttl,
        }


# Instancia global
cache_service = CacheService()


def cached(ttl: int = 300):
    """Decorator para cachear resultados de funciones"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave de caché
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Intentar obtener del caché
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator




