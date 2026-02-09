"""
Cache management for code explanations
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExplanationCache:
    """
    Manages caching of code explanations.
    
    Supports both external cache instances and internal simple cache.
    Can optionally integrate with ModelStats for tracking.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        ttl: float = 3600.0,
        cache_instance: Optional[Any] = None,
        max_internal_size: int = 100,
        stats: Optional[Any] = None
    ):
        """
        Initialize explanation cache.
        
        Args:
            enabled: Enable caching
            ttl: Time-to-live in seconds
            cache_instance: External cache instance (optional)
            max_internal_size: Maximum size of internal cache
            stats: ModelStats instance for tracking (optional)
        """
        self.enabled = enabled
        self.ttl = max(0.0, float(ttl))
        self._cache = cache_instance
        # Usar OrderedDict para LRU cache
        self._internal_cache: OrderedDict[str, str] = OrderedDict()
        self.max_internal_size = max_internal_size
        self._stats = stats
        self._lock = threading.RLock()  # Thread safety
    
    def _generate_key(self, code: str, **kwargs: Any) -> str:
        """
        Generate cache key from code and parameters.
        
        Args:
            code: Code to explain
            **kwargs: Additional parameters
            
        Returns:
            SHA-256 hash of the key data
        """
        # Ordenar kwargs para consistencia en la generación de keys
        sorted_kwargs = sorted(kwargs.items())
        # Usar JSON para serialización consistente
        try:
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback si hay objetos no serializables
            kwargs_str = str(sorted_kwargs)
        
        key_data = f"{code}|{kwargs_str}"
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()
    
    def get(self, code: str, **kwargs: Any) -> Optional[str]:
        """
        Get cached explanation (thread-safe).
        
        Args:
            code: Code to explain
            **kwargs: Generation parameters
            
        Returns:
            Cached explanation or None
        """
        if not self.enabled:
            return None
        
        cache_key = self._generate_key(code, **kwargs)
        
        # Intentar caché externo primero
        if self._cache is not None:
            try:
                if hasattr(self._cache, 'get'):
                    result = self._cache.get(cache_key)
                    if result is not None:
                        logger.debug(f"External cache hit - key: {cache_key[:8]}")
                        if self._stats:
                            self._stats.increment_cache_hit()
                        return result
            except Exception as e:
                logger.warning(f"External cache get error: {e}")
        
        # Intentar caché interno (thread-safe)
        with self._lock:
            if cache_key in self._internal_cache:
                # Mover al final (LRU)
                self._internal_cache.move_to_end(cache_key)
                logger.debug(f"Internal cache hit - key: {cache_key[:8]}")
                if self._stats:
                    self._stats.increment_cache_hit()
                return self._internal_cache[cache_key]
        
        # Cache miss
        if self._stats:
            self._stats.increment_cache_miss()
        
        return None
    
    def set(self, code: str, explanation: str, **kwargs: Any) -> bool:
        """
        Cache explanation (thread-safe LRU).
        
        Args:
            code: Code that was explained
            explanation: Generated explanation
            **kwargs: Generation parameters
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not explanation:
            return False
        
        cache_key = self._generate_key(code, **kwargs)
        
        # Guardar en caché externo si está disponible
        if self._cache is not None:
            try:
                if hasattr(self._cache, 'set'):
                    self._cache.set(cache_key, explanation, ttl=self.ttl)
            except Exception as e:
                logger.warning(f"Failed to save to external cache: {e}")
        
        # Guardar en caché interno (LRU thread-safe)
        with self._lock:
            if cache_key in self._internal_cache:
                # Actualizar valor existente
                self._internal_cache[cache_key] = explanation
            else:
                # Agregar nuevo elemento
                if len(self._internal_cache) >= self.max_internal_size:
                    # Eliminar el más antiguo (LRU)
                    self._internal_cache.popitem(last=False)
                self._internal_cache[cache_key] = explanation
            # Mover al final para LRU (más recientemente usado)
            self._internal_cache.move_to_end(cache_key)
        
        return True
    
    def clear(self) -> int:
        """
        Clear cache (both external and internal) - thread-safe.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        
        # Limpiar caché externo
        if self._cache is not None:
            try:
                if hasattr(self._cache, "clear"):
                    self._cache.clear()
                    count += 1
                elif hasattr(self._cache, "delete_all"):
                    deleted = self._cache.delete_all()
                    count += deleted if isinstance(deleted, int) else 1
            except Exception as e:
                logger.warning(f"Error clearing external cache: {e}")
        
        # Limpiar caché interno (thread-safe)
        with self._lock:
            count += len(self._internal_cache)
            self._internal_cache.clear()
        
        logger.debug(f"Cache cleared - {count} entries removed")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics (thread-safe).
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "enabled": self.enabled,
                "external_cache_available": self._cache is not None,
                "internal_cache_size": len(self._internal_cache),
                "internal_cache_max_size": self.max_internal_size,
                "internal_cache_usage_percent": (
                    (len(self._internal_cache) / self.max_internal_size * 100)
                    if self.max_internal_size > 0 else 0.0
                ),
                "ttl": self.ttl
            }

