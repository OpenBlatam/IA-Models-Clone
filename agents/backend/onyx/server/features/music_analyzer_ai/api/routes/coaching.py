"""
Coaching endpoints for music learning
"""

import logging

from ..base_router import BaseRouter
from ...models.schemas import TrackAnalysisRequest
from ..utils.service_helpers import get_track_or_search
from ..utils.analysis_helpers import perform_track_analysis, add_coaching_to_analysis

logger = logging.getLogger(__name__)


class CoachingRouter(BaseRouter):
    """Router for coaching endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/coaching", tags=["Coaching"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all coaching routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def get_coaching_analysis(request: TrackAnalysisRequest):
            """
            Obtiene análisis de coaching para una canción
            
            - **track_id**: ID de la canción en Spotify
            - **track_name**: Nombre de la canción para buscar
            """
            spotify_service, music_analyzer, music_coach = self.get_services(
                "spotify_service",
                "music_analyzer",
                "music_coach"
            )
            
            track_id = get_track_or_search(
                spotify_service,
                request.track_id,
                request.track_name
            )
            
            analysis = await perform_track_analysis(spotify_service, music_analyzer, track_id)
            analysis = add_coaching_to_analysis(analysis, music_coach)
            
            return self.success_response({
                "track_basic_info": analysis["track_basic_info"],
                "coaching": analysis["coaching"]
            })


def get_coaching_router() -> CoachingRouter:
    """Factory function to get coaching router"""
    return CoachingRouter()

