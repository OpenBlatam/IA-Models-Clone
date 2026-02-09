"""
Sistema de cache inteligente
"""

from typing import Any, Optional, Dict, Callable, List
from datetime import datetime, timedelta
import hashlib
import json
import time
from functools import wraps


class IntelligentCache:
    """Cache inteligente con TTL adaptativo"""
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        """
        Inicializa el cache
        
        Args:
            default_ttl: TTL por defecto en segundos
            max_size: Tamaño máximo del cache
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count: Dict[str, int] = {}
        self.miss_count: Dict[str, int] = {}
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Genera clave de cache"""
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
        if key not in self.cache:
            self.miss_count[key] = self.miss_count.get(key, 0) + 1
            return None
        
        entry = self.cache[key]
        
        # Verificar expiración
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            self.miss_count[key] = self.miss_count.get(key, 0) + 1
            return None
        
        # Actualizar acceso
        self.access_times[key] = time.time()
        self.hit_count[key] = self.hit_count.get(key, 0) + 1
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Guarda valor en cache
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL en segundos (opcional)
        """
        # Limpiar si está lleno
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
        self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Elimina entrada menos usada recientemente"""
        if not self.access_times:
            # Si no hay acceso times, eliminar la más antigua
            if self.cache:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
                del self.cache[oldest_key]
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def delete(self, key: str):
        """Elimina entrada del cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def clear(self):
        """Limpia todo el cache"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count.clear()
        self.miss_count.clear()
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del cache"""
        total_hits = sum(self.hit_count.values())
        total_misses = sum(self.miss_count.values())
        total_requests = total_hits + total_misses
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "most_accessed": self._get_most_accessed(limit=10)
        }
    
    def _get_most_accessed(self, limit: int = 10) -> List[Dict]:
        """Obtiene entradas más accedidas"""
        access_counts = {
            key: self.hit_count.get(key, 0)
            for key in self.cache.keys()
        }
        
        sorted_keys = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"key": key, "hits": count}
            for key, count in sorted_keys[:limit]
        ]
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorador para cachear resultados de funciones
        
        Args:
            ttl: TTL en segundos
            key_prefix: Prefijo para la clave
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self._generate_key(key_prefix or func.__name__, *args, **kwargs)
                
                # Intentar obtener del cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Ejecutar función
                result = func(*args, **kwargs)
                
                # Guardar en cache
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator

