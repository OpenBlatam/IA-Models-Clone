"""
Cache Service - Sistema de caché y performance
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CacheService:
    """Servicio para caché y optimización de performance"""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hora por defecto
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(
        self,
        key: str
    ) -> Optional[Any]:
        """Obtener valor del caché"""
        if key not in self.cache:
            return None
        
        cached_item = self.cache[key]
        
        # Verificar expiración
        expires_at = cached_item.get("expires_at")
        if expires_at:
            if datetime.now() > datetime.fromisoformat(expires_at):
                del self.cache[key]
                return None
        
        return cached_item.get("value")
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Guardar valor en caché"""
        ttl = ttl or self.default_ttl
        expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now().isoformat()
        }
    
    def delete(self, key: str) -> bool:
        """Eliminar del caché"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Limpiar todo el caché"""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generar clave de caché"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached(
        self,
        ttl: Optional[int] = None
    ):
        """Decorador para cachear resultados de funciones"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generar clave de caché
                cache_key = f"{func.__name__}:{self.generate_key(*args, **kwargs)}"
                
                # Intentar obtener del caché
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_value
                
                # Ejecutar función
                logger.debug(f"Cache miss: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Guardar en caché
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        total_items = len(self.cache)
        expired_items = 0
        
        now = datetime.now()
        for item in self.cache.values():
            expires_at = item.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < now:
                expired_items += 1
        
        return {
            "total_items": total_items,
            "active_items": total_items - expired_items,
            "expired_items": expired_items,
            "default_ttl": self.default_ttl
        }
    
    def cleanup_expired(self) -> int:
        """Limpiar items expirados"""
        now = datetime.now()
        expired_keys = []
        
        for key, item in self.cache.items():
            expires_at = item.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)




