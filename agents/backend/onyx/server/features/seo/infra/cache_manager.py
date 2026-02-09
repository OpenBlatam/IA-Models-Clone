from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, Optional, List
from loguru import logger
from ..core.ultra_optimized_cache import UltraOptimizedCache as CoreCache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache Manager adapter for infrastructure layer.
Wraps the core cache manager with infrastructure-specific logic.
"""




class UltraOptimizedCache:
    """Cache Manager adapter for infrastructure layer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.core_cache = CoreCache(config)
        self._setup_infrastructure()
    
    def _setup_infrastructure(self) -> Any:
        """Setup infrastructure-specific configurations."""
        # Configurar prefijos de infraestructura
        self.infrastructure_prefix = f"infra:{self.config.get('environment', 'production')}:"
        
        # Configurar TTL específicos de infraestructura
        self.infrastructure_ttls = {
            'seo_data': 3600,  # 1 hora
            'parsed_content': 1800,  # 30 minutos
            'analysis_results': 7200,  # 2 horas
            'health_checks': 300,  # 5 minutos
            'metrics': 600,  # 10 minutos
        }
        
        logger.info("Infrastructure cache manager configured")
    
    def _get_infrastructure_key(self, key: str, category: str = 'seo_data') -> str:
        """Generate infrastructure-specific cache key."""
        return f"{self.infrastructure_prefix}{category}:{key}"
    
    async def get(self, key: str, category: str = 'seo_data'):
        """Get with infrastructure key management."""
        infra_key = self._get_infrastructure_key(key, category)
        logger.debug(f"Infrastructure cache GET: {infra_key}")
        return await self.core_cache.get(infra_key)
    
    async def set(self, key: str, value: Any, category: str = 'seo_data', ttl: int = None):
        """Set with infrastructure key management."""
        infra_key = self._get_infrastructure_key(key, category)
        if ttl is None:
            ttl = self.infrastructure_ttls.get(category, 3600)
        
        logger.debug(f"Infrastructure cache SET: {infra_key} (TTL: {ttl}s)")
        return await self.core_cache.set(infra_key, value, ttl)
    
    async def delete(self, key: str, category: str = 'seo_data'):
        """Delete with infrastructure key management."""
        infra_key = self._get_infrastructure_key(key, category)
        logger.debug(f"Infrastructure cache DELETE: {infra_key}")
        return await self.core_cache.delete(infra_key)
    
    async def exists(self, key: str, category: str = 'seo_data'):
        """Exists with infrastructure key management."""
        infra_key = self._get_infrastructure_key(key, category)
        return await self.core_cache.exists(infra_key)
    
    async def clear(self, category: str = None):
        """Clear with infrastructure pattern matching."""
        if category:
            pattern = f"{self.infrastructure_prefix}{category}:*"
            logger.info(f"Infrastructure cache CLEAR category: {category}")
        else:
            pattern = f"{self.infrastructure_prefix}*"
            logger.info("Infrastructure cache CLEAR all")
        
        # Implementar limpieza por patrón
        return await self._clear_by_pattern(pattern)
    
    async def _clear_by_pattern(self, pattern: str):
        """Clear cache entries by pattern."""
        try:
            # Esta implementación dependería del backend de cache
            # Para Redis, usaría SCAN + DEL
            # Para memoria, iteraría sobre las claves
            return await self.core_cache.clear()
        except Exception as e:
            logger.error(f"Failed to clear cache by pattern {pattern}: {e}")
            return False
    
    async def get_many(self, keys: List[str], category: str = 'seo_data'):
        """Get many with infrastructure key management."""
        infra_keys = [self._get_infrastructure_key(key, category) for key in keys]
        logger.debug(f"Infrastructure cache GET_MANY: {len(infra_keys)} keys")
        
        results = await self.core_cache.get_many(infra_keys)
        
        # Mapear de vuelta a las claves originales
        return {key: results[infra_key] for key, infra_key in zip(keys, infra_keys) if infra_key in results}
    
    async def set_many(self, data: Dict[str, Any], category: str = 'seo_data', ttl: int = None):
        """Set many with infrastructure key management."""
        if ttl is None:
            ttl = self.infrastructure_ttls.get(category, 3600)
        
        infra_data = {
            self._get_infrastructure_key(key, category): value 
            for key, value in data.items()
        }
        
        logger.debug(f"Infrastructure cache SET_MANY: {len(infra_data)} items")
        return await self.core_cache.set_many(infra_data, ttl)
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get stats with infrastructure context."""
        stats = self.core_cache.get_stats()
        stats['infrastructure_layer'] = True
        stats['infrastructure_prefix'] = self.infrastructure_prefix
        stats['infrastructure_ttls'] = self.infrastructure_ttls
        return stats
    
    async def health_check(self) -> Any:
        """Health check with infrastructure context."""
        health = await self.core_cache.health_check()
        health['infrastructure'] = 'ok'
        return health
    
    async def pipeline(self) -> Any:
        """Pipeline with infrastructure context."""
        return await self.core_cache.pipeline()
    
    def cache_decorator(self, ttl: int = 300, category: str = 'seo_data', key_prefix: str = ""):
        """Cache decorator with infrastructure context."""
        def decorator(func) -> Any:
            async def wrapper(*args, **kwargs) -> Any:
                # Generar clave única con contexto de infraestructura
                key_parts = [key_prefix, func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
                
                # Intentar obtener del cache
                cached_result = await self.get(cache_key, category)
                if cached_result is not None:
                    return cached_result
                
                # Ejecutar función
                result = await func(*args, **kwargs)
                
                # Guardar en cache
                await self.set(cache_key, result, category, ttl)
                
                return result
            return wrapper
        return decorator 