"""
MCP Cache - Sistema de cache para respuestas MCP
=================================================

Sistema de cache en memoria para respuestas MCP con soporte para TTL,
invalidación y estadísticas. Para producción, considerar usar Redis o similar.
"""

import hashlib
import json
import logging
from typing import Any, Optional, Dict, Callable, TypeVar, Awaitable
from datetime import datetime, timedelta
from functools import wraps

from .exceptions import MCPError

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class MCPCache:
    """
    Cache simple en memoria para respuestas MCP.
    
    Proporciona funcionalidad de cache con:
    - TTL (Time To Live) configurable
    - Invalidación por recurso o completa
    - Estadísticas de uso
    - Limpieza automática de entradas expiradas
    
    Nota: Para producción, usar Redis o similar para cache distribuido.
    """
    
    def __init__(self, default_ttl: int = 300) -> None:
        """
        Inicializar cache.
        
        Args:
            default_ttl: TTL por defecto en segundos (default: 300 = 5 minutos).
        
        Raises:
            ValueError: Si default_ttl es inválido.
        """
        if default_ttl < 1:
            raise ValueError("default_ttl must be at least 1 second")
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl: int = default_ttl
        logger.debug(f"MCPCache initialized with default_ttl={default_ttl}s")
    
    def _make_key(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Crea clave de cache única basada en resource_id, operation y parameters.
        
        Args:
            resource_id: ID del recurso.
            operation: Operación a realizar.
            parameters: Parámetros de la operación.
        
        Returns:
            Clave de cache (hash MD5).
        
        Raises:
            ValueError: Si resource_id u operation están vacíos.
            TypeError: Si parameters no es un diccionario.
        """
        if not resource_id or not resource_id.strip():
            raise ValueError("resource_id cannot be empty")
        
        if not operation or not operation.strip():
            raise ValueError("operation cannot be empty")
        
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be a dict, got {type(parameters)}")
        
        try:
            key_data = {
                "resource_id": resource_id.strip(),
                "operation": operation.strip(),
                "parameters": parameters,
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode('utf-8')).hexdigest()
        except (TypeError, ValueError) as e:
            logger.error(f"Error creating cache key: {e}", exc_info=True)
            raise ValueError(f"Failed to create cache key: {e}") from e
    
    def get(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Obtiene valor del cache.
        
        Args:
            resource_id: ID del recurso.
            operation: Operación.
            parameters: Parámetros de la operación.
        
        Returns:
            Valor cacheado o None si no existe o expiró.
        
        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        try:
            key = self._make_key(resource_id, operation, parameters)
            cached = self._cache.get(key)
            
            if not cached:
                return None
            
            # Verificar expiración
            expires_at = cached.get("expires_at")
            if expires_at and datetime.utcnow() > expires_at:
                del self._cache[key]
                logger.debug(f"Cache entry expired for key: {key[:8]}...")
                return None
            
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return cached.get("value")
        
        except Exception as e:
            logger.error(f"Error getting from cache: {e}", exc_info=True)
            return None
    
    def set(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Guarda valor en cache.
        
        Args:
            resource_id: ID del recurso.
            operation: Operación.
            parameters: Parámetros de la operación.
            value: Valor a cachear.
            ttl: TTL en segundos (opcional, usa default si no se especifica).
        
        Raises:
            ValueError: Si los parámetros son inválidos o ttl es inválido.
        """
        if ttl is not None and ttl < 1:
            raise ValueError("ttl must be at least 1 second")
        
        try:
            key = self._make_key(resource_id, operation, parameters)
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            self._cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=effective_ttl),
                "created_at": datetime.utcnow(),
            }
            
            logger.debug(
                f"Cache entry set for key: {key[:8]}... "
                f"(ttl={effective_ttl}s)"
            )
        
        except Exception as e:
            logger.error(f"Error setting cache: {e}", exc_info=True)
            raise ValueError(f"Failed to set cache: {e}") from e
    
    def invalidate(self, resource_id: Optional[str] = None) -> int:
        """
        Invalida cache.
        
        Args:
            resource_id: ID del recurso (opcional). Si se proporciona,
                invalida solo entradas relacionadas con ese recurso.
                Si es None, invalida todo el cache.
        
        Returns:
            Número de entradas invalidadas.
        
        Raises:
            ValueError: Si resource_id está vacío (cuando se proporciona).
        """
        if resource_id is not None:
            if not resource_id.strip():
                raise ValueError("resource_id cannot be empty when provided")
            
            # Invalidar todas las claves que contengan el resource_id
            # Nota: Esta es una implementación simple. Para mejor rendimiento,
            # considerar mantener un índice de resource_id -> keys
            keys_to_remove: list[str] = []
            resource_id_clean = resource_id.strip()
            
            for key, cached_data in list(self._cache.items()):
                # Intentar extraer resource_id de la clave o del valor
                try:
                    # La clave es un hash, necesitamos otra estrategia
                    # Por ahora, invalidamos basándonos en el contenido del valor
                    value = cached_data.get("value", {})
                    if isinstance(value, dict):
                        # Si el valor es un dict, buscar resource_id
                        if resource_id_clean in str(value):
                            keys_to_remove.append(key)
                except Exception:
                    pass
            
            for key in keys_to_remove:
                self._cache.pop(key, None)
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for resource: {resource_id_clean}")
            return len(keys_to_remove)
        else:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Invalidated all cache entries ({count} total)")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas:
            - total_entries: Total de entradas en cache
            - expired_entries: Entradas expiradas
            - active_entries: Entradas activas
        """
        try:
            now = datetime.utcnow()
            expired = sum(
                1 for cached in self._cache.values()
                if cached.get("expires_at") and now > cached["expires_at"]
            )
            
            stats = {
                "total_entries": len(self._cache),
                "expired_entries": expired,
                "active_entries": len(self._cache) - expired,
                "default_ttl": self.default_ttl,
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "active_entries": 0,
                "default_ttl": self.default_ttl,
                "error": str(e),
            }
    
    def cleanup_expired(self) -> int:
        """
        Limpia entradas expiradas del cache.
        
        Returns:
            Número de entradas eliminadas.
        """
        try:
            now = datetime.utcnow()
            keys_to_remove = [
                key for key, cached in self._cache.items()
                if cached.get("expires_at") and now > cached["expires_at"]
            ]
            
            for key in keys_to_remove:
                self._cache.pop(key, None)
            
            if keys_to_remove:
                logger.debug(f"Cleaned up {len(keys_to_remove)} expired cache entries")
            
            return len(keys_to_remove)
        
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}", exc_info=True)
            return 0


def cached(ttl: int = 300) -> Callable[[F], F]:
    """
    Decorador para cachear resultados de funciones async.
    
    Args:
        ttl: TTL en segundos (default: 300).
    
    Returns:
        Decorador que envuelve la función con lógica de cache.
    
    Raises:
        ValueError: Si ttl es inválido.
    
    Example:
        @cached(ttl=600)
        async def my_function(param1: str, param2: int) -> dict:
            # función que puede ser cacheada
            return {"result": "data"}
    """
    if ttl < 1:
        raise ValueError("ttl must be at least 1 second")
    
    def decorator(func: F) -> F:
        cache = MCPCache(default_ttl=ttl)
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Crear clave de cache basada en función y argumentos
                cache_key = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                key_str = json.dumps(cache_key, sort_keys=True, default=str)
                cache_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
                
                # Intentar obtener del cache
                cached_value = cache.get("decorated", cache_hash, {})
                if cached_value is not None:
                    logger.debug(f"Cache hit for decorated function: {func.__name__}")
                    return cached_value
                
                # Ejecutar función
                result = await func(*args, **kwargs)
                
                # Guardar en cache
                cache.set("decorated", cache_hash, {}, result, ttl=ttl)
                logger.debug(f"Cache miss for decorated function: {func.__name__}, result cached")
                
                return result
            
            except Exception as e:
                logger.error(f"Error in cached decorator for {func.__name__}: {e}", exc_info=True)
                # En caso de error, ejecutar función sin cache
                return await func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator
