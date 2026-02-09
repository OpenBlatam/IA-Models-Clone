"""
Response Cache Middleware
Cache agresivo de respuestas HTTP
"""

import logging
import hashlib
import time
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """Middleware para cache agresivo de respuestas"""
    
    def __init__(self, app, ttl: int = 300, max_size: int = 1000):
        super().__init__(app)
        self.ttl = ttl
        self.max_size = max_size
        self._cache: dict = {}
        self._cache_times: dict = {}
        self._access_times: dict = {}
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con cache de respuestas"""
        # Solo cachear GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generar cache key
        cache_key = self._generate_cache_key(request)
        
        # Verificar cache
        cached_response = self._get_cached(cache_key)
        if cached_response:
            logger.debug(f"Cache hit: {request.url.path}")
            return cached_response
        
        # Procesar request
        response = await call_next(request)
        
        # Cachear si es exitoso y cacheable
        if self._is_cacheable(request, response):
            self._cache_response(cache_key, response)
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Genera clave de cache"""
        key_parts = [
            request.method,
            str(request.url.path),
            str(request.url.query),
            request.headers.get("authorization", ""),
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[Response]:
        """Obtiene respuesta del cache"""
        if cache_key in self._cache:
            # Verificar TTL
            if time.time() - self._cache_times[cache_key] < self.ttl:
                self._access_times[cache_key] = time.time()
                return self._cache[cache_key]
            else:
                # Expirar
                del self._cache[cache_key]
                del self._cache_times[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: Response):
        """Cachea una respuesta"""
        # Limpiar cache si es muy grande
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[cache_key] = response
        self._cache_times[cache_key] = time.time()
        self._access_times[cache_key] = time.time()
    
    def _evict_oldest(self):
        """Elimina la entrada más antigua (LRU)"""
        if not self._access_times:
            # Si no hay access times, eliminar la más antigua por tiempo de cache
            oldest_key = min(self._cache_times.items(), key=lambda x: x[1])[0]
        else:
            # LRU: eliminar la menos accedida recientemente
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        
        del self._cache[oldest_key]
        del self._cache_times[oldest_key]
        if oldest_key in self._access_times:
            del self._access_times[oldest_key]
    
    def _is_cacheable(self, request: Request, response: Response) -> bool:
        """Determina si una respuesta es cacheable"""
        # Solo cachear respuestas exitosas
        if not (200 <= response.status_code < 300):
            return False
        
        # No cachear si tiene headers que indican no cachear
        cache_control = response.headers.get("cache-control", "").lower()
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        # No cachear respuestas con autenticación (a menos que sea específico)
        if "authorization" in request.headers:
            # Podríamos cachear por usuario, pero por simplicidad no cacheamos
            return False
        
        return True















