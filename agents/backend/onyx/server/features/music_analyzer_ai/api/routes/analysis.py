"""
Analysis endpoints for music tracks
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter
from ...models.schemas import TrackAnalysisRequest
from ...services.webhook_service import WebhookEvent
from ..utils.service_helpers import get_track_or_search
from ..utils.analysis_helpers import (
    perform_track_analysis,
    add_coaching_to_analysis,
    trigger_webhook_safe,
    save_analysis_to_history
)

logger = logging.getLogger(__name__)


class AnalysisRouter(BaseRouter):
    """Router for analysis endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/analyze", tags=["Analysis"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all analysis routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def analyze_track(request: TrackAnalysisRequest):
            """
            Analiza una canción completa
            
            - **track_id**: ID de la canción en Spotify (opcional si se proporciona track_name)
            - **track_name**: Nombre de la canción para buscar (opcional si se proporciona track_id)
            - **include_coaching**: Incluir análisis de coaching
            """
            spotify_service, music_analyzer, music_coach, history_service, webhook_service, analytics_service = \
                self.get_services(
                    "spotify_service",
                    "music_analyzer",
                    "music_coach",
                    "history_service",
                    "webhook_service",
                    "analytics_service"
                )
            
            track_id = get_track_or_search(
                spotify_service,
                request.track_id,
                request.track_name
            )
            
            response = await perform_track_analysis(spotify_service, music_analyzer, track_id)
            
            if request.include_coaching:
                response = add_coaching_to_analysis(response, music_coach)
            
            save_analysis_to_history(
                history_service,
                analytics_service,
                track_id,
                response
            )
            
            await trigger_webhook_safe(
                webhook_service,
                WebhookEvent.ANALYSIS_COMPLETED,
                {
                    "track_id": track_id,
                    "track_name": response["track_basic_info"]["name"],
                    "analysis_id": track_id
                }
            )
            
            return self.success_response(response)
        
        @self.router.get("/{track_id}", response_model=dict)
        @self.handle_exceptions
        async def analyze_track_by_id(
            track_id: str,
            include_coaching: bool = Query(True, description="Incluir análisis de coaching")
        ):
            """
            Analiza una canción por su ID de Spotify
            
            - **track_id**: ID de la canción en Spotify
            - **include_coaching**: Incluir análisis de coaching
            """
            spotify_service, music_analyzer, music_coach = self.get_services(
                "spotify_service",
                "music_analyzer",
                "music_coach"
            )
            
            response = await perform_track_analysis(spotify_service, music_analyzer, track_id)
            
            if include_coaching:
                response = add_coaching_to_analysis(response, music_coach)
            
            return self.success_response(response)


def get_analysis_router() -> AnalysisRouter:
    """Factory function to get analysis router"""
    return AnalysisRouter()

