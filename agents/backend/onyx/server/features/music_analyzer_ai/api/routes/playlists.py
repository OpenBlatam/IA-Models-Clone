"""
Playlists endpoints
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class PlaylistsRouter(BaseRouter):
    """Router for playlists endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/playlists", tags=["Playlists"])
        self._playlist_service = None
        self._register_routes()
    
    def _get_playlist_service(self):
        """Get or cache playlist service"""
        if self._playlist_service is None:
            self._playlist_service = self.get_service("playlist_service")
        return self._playlist_service
    
    def _register_routes(self):
        """Register all playlists routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def create_playlist(
            name: str = Query(...),
            user_id: str = Query(...),
            is_public: bool = Query(False),
            description: Optional[str] = Query(None)
        ):
            """Crea una nueva playlist"""
            playlist_service = self._get_playlist_service()
            playlist = playlist_service.create_playlist(user_id, name, is_public, description)
            return self.success_response({"playlist": playlist}, message="Playlist creada")
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_playlists(
            user_id: Optional[str] = Query(None),
            public_only: bool = Query(False)
        ):
            """Obtiene las playlists de un usuario o públicas"""
            playlist_service = self._get_playlist_service()
            playlists = playlist_service.get_playlists(user_id, public_only)
            return self.list_response(playlists, key="playlists")
        
        @self.router.get("/{playlist_id}", response_model=dict)
        @self.handle_exceptions
        async def get_playlist(playlist_id: str):
            """Obtiene una playlist específica"""
            playlist_service = self._get_playlist_service()
            playlist = playlist_service.get_playlist(playlist_id)
            self.require_not_none(playlist, "Playlist no encontrada", status_code=404)
            return self.success_response({"playlist": playlist})
        
        @self.router.post("/{playlist_id}/tracks", response_model=dict)
        @self.handle_exceptions
        async def add_track_to_playlist(
            playlist_id: str,
            track_id: str = Query(...),
            track_name: str = Query(...),
            artists: List[str] = Query(...)
        ):
            """Agrega una canción a una playlist"""
            playlist_service = self._get_playlist_service()
            success = playlist_service.add_track(playlist_id, track_id, track_name, artists)
            self.require_success(success, "Error al agregar canción", status_code=400)
            return self.success_response(None, message="Canción agregada a la playlist")
        
        @self.router.delete("/{playlist_id}/tracks/{track_id}", response_model=dict)
        @self.handle_exceptions
        async def remove_track_from_playlist(playlist_id: str, track_id: str):
            """Elimina una canción de una playlist"""
            playlist_service = self._get_playlist_service()
            success = playlist_service.remove_track(playlist_id, track_id)
            self.require_success(success, "Canción no encontrada en la playlist", status_code=404)
            return self.success_response(None, message="Canción eliminada de la playlist")
        
        @self.router.delete("/{playlist_id}", response_model=dict)
        @self.handle_exceptions
        async def delete_playlist(
            playlist_id: str,
            user_id: str = Query(...)
        ):
            """Elimina una playlist"""
            playlist_service = self._get_playlist_service()
            success = playlist_service.delete_playlist(playlist_id, user_id)
            self.require_success(success, "Playlist no encontrada", status_code=404)
            return self.success_response(None, message="Playlist eliminada")


def get_playlists_router() -> PlaylistsRouter:
    """Factory function to get playlists router"""
    return PlaylistsRouter()

