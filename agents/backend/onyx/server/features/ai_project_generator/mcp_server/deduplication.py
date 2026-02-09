"""
MCP Request Deduplication - Deduplicación de requests
=======================================================
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """
    Deduplicador de requests
    
    Detecta y maneja requests duplicados.
    """
    
    def __init__(
        self,
        cache_ttl: int = 300,
        max_cache_size: int = 10000,
    ):
        """
        Args:
            cache_ttl: TTL del cache en segundos
            max_cache_size: Tamaño máximo del cache
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
    
    def _generate_key(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Genera clave única para request
        
        Args:
            method: Método HTTP
            path: Path
            params: Parámetros (opcional)
            body: Body (opcional)
            user_id: ID del usuario (opcional)
            
        Returns:
            Clave única
        """
        key_parts = [method, path]
        
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        
        if body:
            if isinstance(body, (dict, list)):
                key_parts.append(json.dumps(body, sort_keys=True))
            else:
                key_parts.append(str(body))
        
        if user_id:
            key_parts.append(user_id)
        
        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def check_duplicate(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Verifica si el request es duplicado
        
        Args:
            method: Método HTTP
            path: Path
            params: Parámetros (opcional)
            body: Body (opcional)
            user_id: ID del usuario (opcional)
            
        Returns:
            Response cached si es duplicado, None si no
        """
        key = self._generate_key(method, path, params, body, user_id)
        
        # Limpiar entradas expiradas
        self._clean_expired()
        
        # Verificar cache
        if key in self._cache:
            response, timestamp = self._cache[key]
            
            # Mover al final (LRU)
            self._cache.move_to_end(key)
            
            logger.info(f"Duplicate request detected: {key[:16]}...")
            return response
        
        return None
    
    def store_response(
        self,
        method: str,
        path: str,
        response: Any,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        user_id: Optional[str] = None,
    ):
        """
        Almacena response para deduplicación
        
        Args:
            method: Método HTTP
            path: Path
            response: Response a almacenar
            params: Parámetros (opcional)
            body: Body (opcional)
            user_id: ID del usuario (opcional)
        """
        key = self._generate_key(method, path, params, body, user_id)
        
        # Limpiar entradas expiradas
        self._clean_expired()
        
        # Verificar tamaño máximo
        if len(self._cache) >= self.max_cache_size:
            # Eliminar el más antiguo (FIFO)
            self._cache.popitem(last=False)
        
        # Almacenar
        self._cache[key] = (response, datetime.utcnow())
        logger.debug(f"Stored response for deduplication: {key[:16]}...")
    
    def _clean_expired(self):
        """Limpia entradas expiradas"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if (now - timestamp).total_seconds() > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def clear(self):
        """Limpia todo el cache"""
        self._cache.clear()
        logger.info("Deduplication cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del deduplicador"""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
            "cache_ttl": self.cache_ttl,
        }

