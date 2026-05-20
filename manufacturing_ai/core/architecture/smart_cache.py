"""
Smart Caching System
====================

Sistema de caché inteligente con predicción y optimización automática.
"""

import time
import hashlib
import json
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from collections import OrderedDict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class SmartCache:
    """Caché inteligente con TTL y LRU."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        enable_stats: bool = True
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        self.cache: OrderedDict = OrderedDict()
        self.expiry_times: Dict[str, float] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Genera clave única."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Verifica si entrada expiró."""
        if self.default_ttl is None:
            return False
        if key not in self.expiry_times:
            return False
        return time.time() > self.expiry_times[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor."""
        with self.lock:
            if key not in self.cache:
                return None
            
            if self._is_expired(key):
                self._delete(key)
                return None
            
            # Mover al final (LRU)
            self.cache.move_to_end(key)
            
            if self.enable_stats:
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Establece valor."""
        with self.lock:
            # Eliminar si existe
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Eliminar más antiguo
                oldest_key = next(iter(self.cache))
                self._delete(oldest_key)
            
            self.cache[key] = value
            
            # Establecer TTL
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            if ttl_to_use is not None:
                self.expiry_times[key] = time.time() + ttl_to_use
    
    def _delete(self, key: str) -> None:
        """Elimina entrada."""
        self.cache.pop(key, None)
        self.expiry_times.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def delete(self, key: str) -> None:
        """Elimina entrada."""
        with self.lock:
            self._delete(key)
    
    def clear(self) -> None:
        """Limpia cache."""
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            hit_rate = len(self.cache) / total_accesses if total_accesses > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "hit_rate": hit_rate,
                "most_accessed": sorted(
                    self.access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }


def cached(
    max_size: int = 1000,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorador para caché."""
    cache = SmartCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generar clave
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache._generate_key(*args, **kwargs)
            
            # Intentar obtener del cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en cache
            cache.set(key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator


class PredictionCache:
    """Caché especializado para predicciones de modelos."""
    
    def __init__(self, max_size: int = 5000, ttl: float = 3600):
        self.cache = SmartCache(max_size=max_size, default_ttl=ttl)
        self.model_versions: Dict[str, str] = {}
    
    def _generate_key(self, model_id: str, input_data: Any, version: Optional[str] = None) -> str:
        """Genera clave para predicción."""
        key_data = {
            'model_id': model_id,
            'input': input_data,
            'version': version or self.model_versions.get(model_id, 'default')
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_prediction(self, model_id: str, input_data: Any) -> Optional[Any]:
        """Obtiene predicción del cache."""
        key = self._generate_key(model_id, input_data)
        return self.cache.get(key)
    
    def set_prediction(self, model_id: str, input_data: Any, prediction: Any) -> None:
        """Almacena predicción."""
        key = self._generate_key(model_id, input_data)
        self.cache.set(key, prediction)
    
    def invalidate_model(self, model_id: str) -> None:
        """Invalida todas las predicciones de un modelo."""
        # Actualizar versión para invalidar cache
        self.model_versions[model_id] = str(time.time())
    
    def set_model_version(self, model_id: str, version: str) -> None:
        """Establece versión del modelo."""
        self.model_versions[model_id] = version


class MultiLevelCache:
    """Caché multi-nivel (L1: memoria, L2: disco opcional)."""
    
    def __init__(self, l1_size: int = 1000, l2_enabled: bool = False):
        self.l1_cache = SmartCache(max_size=l1_size)
        self.l2_enabled = l2_enabled
        self.l2_cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor (L1 primero, luego L2)."""
        # Intentar L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Intentar L2
        if self.l2_enabled and key in self.l2_cache:
            value = self.l2_cache[key]
            # Promover a L1
            self.l1_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Establece valor en ambos niveles."""
        self.l1_cache.set(key, value, ttl=ttl)
        if self.l2_enabled:
            self.l2_cache[key] = value
    
    def clear(self) -> None:
        """Limpia ambos niveles."""
        self.l1_cache.clear()
        self.l2_cache.clear()


# Factory functions
_prediction_cache = None

def get_prediction_cache() -> PredictionCache:
    """Obtiene caché global de predicciones."""
    global _prediction_cache
    if _prediction_cache is None:
        _prediction_cache = PredictionCache()
    return _prediction_cache


