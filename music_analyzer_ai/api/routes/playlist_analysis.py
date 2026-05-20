"""
Playlist analysis endpoints
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter
from ..constants import MAX_TRACKS_FOR_PLAYLIST_ANALYSIS
from ..utils.router_helpers import validate_track_ids_count

logger = logging.getLogger(__name__)


class PlaylistAnalysisRouter(BaseRouter):
    """Router for playlist analysis endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/playlists", tags=["Playlist Analysis"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all playlist analysis routes"""
        
        @self.router.post("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_playlist(track_ids: List[str]):
            """Analiza una playlist completa"""
            try:
                validate_track_ids_count(
                    track_ids,
                    2,
                    MAX_TRACKS_FOR_PLAYLIST_ANALYSIS,
                    "tracks"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            playlist_analyzer = self.get_service("playlist_analyzer")
            analysis = playlist_analyzer.analyze_playlist(track_ids)
            return self.success_response({"analysis": analysis})
        
        @self.router.post("/suggest-improvements", response_model=dict)
        @self.handle_exceptions
        async def suggest_improvements(track_ids: List[str]):
            """Sugiere mejoras para una playlist"""
            try:
                validate_track_ids_count(
                    track_ids,
                    1,
                    MAX_TRACKS_FOR_PLAYLIST_ANALYSIS,
                    "tracks"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            playlist_analyzer = self.get_service("playlist_analyzer")
            suggestions = playlist_analyzer.suggest_playlist_improvements(track_ids)
            return self.success_response({"suggestions": suggestions})
        
        @self.router.post("/optimize-order", response_model=dict)
        @self.handle_exceptions
        async def optimize_order(track_ids: List[str]):
            """Optimiza el orden de tracks en una playlist"""
            try:
                validate_track_ids_count(
                    track_ids,
                    1,
                    MAX_TRACKS_FOR_PLAYLIST_ANALYSIS,
                    "tracks"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            playlist_analyzer = self.get_service("playlist_analyzer")
            optimization = playlist_analyzer.optimize_playlist_order(track_ids)
            return self.success_response({
                "optimization": optimization
            })


def get_playlist_analysis_router() -> PlaylistAnalysisRouter:
    """Factory function to get playlist analysis router"""
    return PlaylistAnalysisRouter()

