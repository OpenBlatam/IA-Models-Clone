"""
MCP Request Deduplication - Deduplicación de requests
=====================================================

Evita procesar requests duplicados dentro de una ventana de tiempo.
"""

import hashlib
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """Deduplicador de requests"""
    
    def __init__(
        self,
        window_seconds: float = 60.0,
        max_cache_size: int = 10000
    ):
        self.window_seconds = window_seconds
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generar clave única para el request"""
        key_parts = [method, path]
        
        if body:
            key_parts.append(body)
        
        if headers:
            relevant_headers = {
                k: v for k, v in headers.items()
                if k.lower() not in ['x-request-id', 'x-api-key', 'authorization', 'user-agent', 'date']
            }
            if relevant_headers:
                key_parts.append(str(sorted(relevant_headers.items())))
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def is_duplicate(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[float]]:
        """Verificar si el request es duplicado"""
        key = self._generate_key(method, path, body, headers)
        now = time.time()
        
        if key in self._cache:
            cached_time = self._cache[key]
            if now - cached_time < self.window_seconds:
                self._hits += 1
                age = now - cached_time
                self._cache.move_to_end(key)
                return True, age
            else:
                del self._cache[key]
        
        self._misses += 1
        
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = now
        return False, None
    
    def clear(self):
        """Limpiar cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
            "window_seconds": self.window_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total
        }

