"""
Temporal analysis endpoints for time-based music analysis
"""

from fastapi import Query
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class TemporalRouter(BaseRouter):
    """Router for temporal analysis endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/temporal", tags=["Temporal Analysis"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all temporal routes"""
        
        @self.router.get("/structure", response_model=dict)
        @self.handle_exceptions
        async def get_temporal_structure(track_id: str = Query(...)):
            """Analiza la estructura temporal de una canción"""
            spotify_service, temporal_analyzer = self.get_services(
                "spotify_service",
                "temporal_analyzer"
            )
            
            audio_analysis = spotify_service.get_track_audio_analysis(track_id)
            structure = temporal_analyzer.analyze_temporal_structure(audio_analysis)
            
            return self.success_response({
                "track_id": track_id,
                "temporal_structure": structure
            })
        
        @self.router.get("/energy", response_model=dict)
        @self.handle_exceptions
        async def get_energy_progression(track_id: str = Query(...)):
            """Analiza la progresión de energía a lo largo del tiempo"""
            spotify_service, temporal_analyzer = self.get_services(
                "spotify_service",
                "temporal_analyzer"
            )
            
            audio_analysis = spotify_service.get_track_audio_analysis(track_id)
            progression = temporal_analyzer.analyze_energy_progression(audio_analysis)
            
            return self.success_response({
                "track_id": track_id,
                "energy_progression": progression
            })
        
        @self.router.get("/tempo", response_model=dict)
        @self.handle_exceptions
        async def get_tempo_changes(track_id: str = Query(...)):
            """Analiza cambios de tempo a lo largo del tiempo"""
            spotify_service, temporal_analyzer = self.get_services(
                "spotify_service",
                "temporal_analyzer"
            )
            
            audio_analysis = spotify_service.get_track_audio_analysis(track_id)
            changes = temporal_analyzer.analyze_tempo_changes(audio_analysis)
            
            return self.success_response({
                "track_id": track_id,
                "tempo_changes": changes
            })


def get_temporal_router() -> TemporalRouter:
    """Factory function to get temporal router"""
    return TemporalRouter()

