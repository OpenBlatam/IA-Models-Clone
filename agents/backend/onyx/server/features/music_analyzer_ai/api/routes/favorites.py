"""
Favorites endpoints for user favorites
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class FavoritesRouter(BaseRouter):
    """Router for favorites endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/favorites", tags=["Favorites"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all favorites routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def add_favorite(
            track_id: str,
            track_name: str,
            artists: List[str],
            user_id: str = Query(..., description="ID del usuario"),
            notes: Optional[str] = Query(None, description="Notas opcionales")
        ):
            """Agrega una canción a favoritos"""
            favorites_service = self.get_service("favorites_service")
            success = favorites_service.add_favorite(user_id, track_id, track_name, artists, notes)
            self.require_success(success, "La canción ya está en favoritos", status_code=400)
            return self.success_response(None, message="Canción agregada a favoritos")
        
        @self.router.delete("/{track_id}", response_model=dict)
        @self.handle_exceptions
        async def remove_favorite(track_id: str, user_id: str = Query(...)):
            """Elimina una canción de favoritos"""
            favorites_service = self.get_service("favorites_service")
            success = favorites_service.remove_favorite(user_id, track_id)
            self.require_success(success, "Canción no encontrada en favoritos", status_code=404)
            return self.success_response(None, message="Canción eliminada de favoritos")
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_favorites(user_id: str = Query(...)):
            """Obtiene los favoritos de un usuario"""
            favorites_service = self.get_service("favorites_service")
            favorites = favorites_service.get_favorites(user_id)
            return self.list_response(favorites, key="favorites")
        
        @self.router.get("/stats", response_model=dict)
        @self.handle_exceptions
        async def get_favorites_stats(user_id: str = Query(...)):
            """Obtiene estadísticas de favoritos"""
            favorites_service = self.get_service("favorites_service")
            stats = favorites_service.get_stats(user_id)
            return self.success_response({"stats": stats})


def get_favorites_router() -> FavoritesRouter:
    """Factory function to get favorites router"""
    return FavoritesRouter()

