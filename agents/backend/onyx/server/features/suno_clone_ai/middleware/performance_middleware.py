"""
Performance Middleware
Optimizaciones de rendimiento y caching
"""

import logging
import hashlib
import json
from typing import Optional, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware para optimizaciones de rendimiento
    - Response caching
    - Compression
    - Connection keep-alive
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.cache_ttl = kwargs.get('cache_ttl', 300)  # 5 minutos
        self.enable_compression = kwargs.get('enable_compression', True)
        self._cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self.cacheable_methods = {"GET", "HEAD"}
        self.cacheable_paths = kwargs.get('cacheable_paths', [])
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con optimizaciones"""
        # Verificar cache para GET requests
        if self.enable_caching and request.method in self.cacheable_methods:
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                logger.debug(f"Cache hit for {request.url.path}")
                return cached_response
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de performance
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Connection"] = "keep-alive"
        
        # Cachear respuesta si es cacheable
        if self.enable_caching and request.method in self.cacheable_methods:
            if self._is_cacheable(request, response):
                cache_key = self._generate_cache_key(request)
                self._cache_response(cache_key, response)
        
        # Compression (si está habilitado y el cliente lo soporta)
        if self.enable_compression:
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" in accept_encoding or "br" in accept_encoding:
                # FastAPI maneja compression automáticamente con GZipMiddleware
                pass
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Genera clave de cache basada en request"""
        key_parts = [
            request.method,
            str(request.url.path),
            str(request.url.query),
            request.headers.get("authorization", ""),
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Obtiene respuesta del cache si existe y no ha expirado"""
        import time
        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                del self._cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Response):
        """Cachea una respuesta"""
        import time
        # Solo cachear respuestas exitosas
        if 200 <= response.status_code < 300:
            self._cache[cache_key] = (response, time.time())
            # Limpiar cache viejo si es muy grande
            if len(self._cache) > 1000:
                self._cleanup_cache()
    
    def _is_cacheable(self, request: Request, response: Response) -> bool:
        """Determina si una respuesta es cacheable"""
        # No cachear si hay paths específicos configurados
        if self.cacheable_paths:
            if request.url.path not in self.cacheable_paths:
                return False
        
        # No cachear respuestas con errores
        if response.status_code >= 400:
            return False
        
        # No cachear si tiene headers que indican no cachear
        cache_control = response.headers.get("cache-control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        return True
    
    def _cleanup_cache(self):
        """Limpia entradas viejas del cache"""
        import time
        current_time = time.time()
        keys_to_remove = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in keys_to_remove:
            del self._cache[key]
        logger.debug(f"Cleaned up {len(keys_to_remove)} cache entries")















