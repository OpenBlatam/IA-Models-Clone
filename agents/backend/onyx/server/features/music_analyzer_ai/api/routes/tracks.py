"""
Track information endpoints
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter
from ..utils.response_formatters import format_tracks_response

logger = logging.getLogger(__name__)


class TracksRouter(BaseRouter):
    """Router for track information endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/track", tags=["Tracks"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all track routes"""
        
        @self.router.get("/{track_id}/info", response_model=dict)
        @self.handle_exceptions
        async def get_track_info(track_id: str):
            """Obtiene información básica de una canción"""
            spotify_service = self.get_service("spotify_service")
            track_info = spotify_service.get_track(track_id)
            return self.success_response({"track": track_info})
        
        @self.router.get("/{track_id}/audio-features", response_model=dict)
        @self.handle_exceptions
        async def get_audio_features(track_id: str):
            """Obtiene las características de audio de una canción"""
            spotify_service = self.get_service("spotify_service")
            features = spotify_service.get_track_audio_features(track_id)
            return self.success_response({"audio_features": features})
        
        @self.router.get("/{track_id}/audio-analysis", response_model=dict)
        @self.handle_exceptions
        async def get_audio_analysis(track_id: str):
            """Obtiene el análisis de audio detallado de una canción"""
            spotify_service = self.get_service("spotify_service")
            analysis = spotify_service.get_track_audio_analysis(track_id)
            return self.success_response({"audio_analysis": analysis})
        
        @self.router.get("/{track_id}/recommendations", response_model=dict)
        @self.handle_exceptions
        async def get_recommendations(
            track_id: str,
            limit: int = Query(20, ge=1, le=100, description="Número de recomendaciones")
        ):
            """
            Obtiene recomendaciones de canciones similares
            
            - **track_id**: ID de la canción en Spotify
            - **limit**: Número de recomendaciones (1-100)
            """
            spotify_service = self.get_service("spotify_service")
            recommendations = spotify_service.get_recommendations(track_id, limit)
            
            results = format_tracks_response(recommendations)
            
            return self.list_response(
                results,
                key="recommendations",
                seed_track_id=track_id
            )


def get_tracks_router() -> TracksRouter:
    """Factory function to get tracks router"""
    return TracksRouter()

