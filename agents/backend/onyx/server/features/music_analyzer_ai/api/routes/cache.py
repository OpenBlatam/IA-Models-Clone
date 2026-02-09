"""
Cache management endpoints
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter
from ...utils.cache import cache_manager

logger = logging.getLogger(__name__)


class CacheRouter(BaseRouter):
    """Router for cache management endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/cache", tags=["Cache"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all cache routes"""
        
        @self.router.get("/stats", response_model=dict)
        @self.handle_exceptions
        async def get_cache_stats():
            """Obtiene estadísticas del cache"""
            stats = cache_manager.get_stats()
            return self.success_response({"cache": stats})
        
        @self.router.delete("/clear", response_model=dict)
        @self.handle_exceptions
        async def clear_cache(prefix: Optional[str] = Query(None, description="Prefijo para limpiar cache específico")):
            """Limpia el cache"""
            cache_manager.clear(prefix)
            return self.success_response(
                None,
                message=f"Cache cleared: {prefix or 'all'}"
            )


def get_cache_router() -> CacheRouter:
    """Factory function to get cache router"""
    return CacheRouter()

