"""
Collaborations endpoints for artist collaboration analysis
"""

from fastapi import Query
from typing import List
import logging

from ..base_router import BaseRouter
from ..constants import MAX_ARTISTS_FOR_NETWORK, MIN_TRACKS_FOR_COMPARISON, MAX_TRACKS_FOR_COMPARISON
from ..utils.router_helpers import validate_track_ids_count

logger = logging.getLogger(__name__)


class CollaborationsRouter(BaseRouter):
    """Router for collaborations endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/collaborations", tags=["Collaborations"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all collaborations routes"""
        
        @self.router.get("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_collaborations(track_id: str = Query(...)):
            """Analiza las colaboraciones en un track"""
            collaboration_analyzer = self.get_service("collaboration_analyzer")
            analysis = collaboration_analyzer.analyze_track_collaborations(track_id)
            return self.success_response({"analysis": analysis})
        
        @self.router.post("/network", response_model=dict)
        @self.handle_exceptions
        async def analyze_artist_network(artist_ids: List[str]):
            """Analiza la red de colaboraciones de artistas"""
            try:
                validate_track_ids_count(
                    artist_ids,
                    1,
                    MAX_ARTISTS_FOR_NETWORK,
                    "artistas"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            collaboration_analyzer = self.get_service("collaboration_analyzer")
            network = collaboration_analyzer.analyze_artist_network(artist_ids)
            return self.success_response({"network": network})
        
        @self.router.post("/versions/compare", response_model=dict)
        @self.handle_exceptions
        async def compare_versions(track_ids: List[str]):
            """Compara diferentes versiones de una canción"""
            try:
                validate_track_ids_count(
                    track_ids,
                    MIN_TRACKS_FOR_COMPARISON,
                    MAX_TRACKS_FOR_COMPARISON,
                    "tracks"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            collaboration_analyzer = self.get_service("collaboration_analyzer")
            comparison = collaboration_analyzer.compare_versions(track_ids)
            return self.success_response({"comparison": comparison})


def get_collaborations_router() -> CollaborationsRouter:
    """Factory function to get collaborations router"""
    return CollaborationsRouter()

