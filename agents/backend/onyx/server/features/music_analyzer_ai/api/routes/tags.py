"""
Tags endpoints for resource tagging
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class TagsRouter(BaseRouter):
    """Router for tags endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/tags", tags=["Tags"])
        self._tagging_service = None
        self._register_routes()
    
    def _get_tagging_service(self):
        """Get or cache tagging service"""
        if self._tagging_service is None:
            self._tagging_service = self.get_service("tagging_service")
        return self._tagging_service
    
    def _register_routes(self):
        """Register all tags routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def add_tags(
            resource_id: str,
            resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
            tags: List[str] = Query(...),
            user_id: Optional[str] = Query(None)
        ):
            """Agrega tags a un recurso"""
            tagging_service = self._get_tagging_service()
            tagging_service.add_tags(resource_id, resource_type, tags, user_id)
            return self.success_response({
                "tags": tagging_service.get_tags(resource_id, resource_type)
            }, message="Tags agregados")
        
        @self.router.delete("", response_model=dict)
        @self.handle_exceptions
        async def remove_tags(
            resource_id: str,
            resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
            tags: List[str] = Query(...),
            user_id: Optional[str] = Query(None)
        ):
            """Elimina tags de un recurso"""
            tagging_service = self._get_tagging_service()
            tagging_service.remove_tags(resource_id, resource_type, tags, user_id)
            return self.success_response(None, message="Tags eliminados")
        
        @self.router.get("/{resource_id}", response_model=dict)
        @self.handle_exceptions
        async def get_tags(
            resource_id: str,
            resource_type: str = Query("track", regex="^(track|analysis|playlist)$")
        ):
            """Obtiene los tags de un recurso"""
            tagging_service = self._get_tagging_service()
            tags = tagging_service.get_tags(resource_id, resource_type)
            return self.success_response({"tags": tags})
        
        @self.router.get("/search", response_model=dict)
        @self.handle_exceptions
        async def search_by_tags(
            tags: List[str] = Query(...),
            resource_type: Optional[str] = Query(None, regex="^(track|analysis|playlist)$")
        ):
            """Busca recursos por tags"""
            tagging_service = self._get_tagging_service()
            results = tagging_service.search_by_tags(tags, resource_type)
            return self.list_response(results, key="results")
        
        @self.router.get("/popular", response_model=dict)
        @self.handle_exceptions
        async def get_popular_tags(
            limit: int = Query(20, ge=1, le=100)
        ):
            """Obtiene los tags más populares"""
            tagging_service = self._get_tagging_service()
            tags = tagging_service.get_popular_tags(limit)
            return self.list_response(tags, key="tags")


def get_tags_router() -> TagsRouter:
    """Factory function to get tags router"""
    return TagsRouter()

