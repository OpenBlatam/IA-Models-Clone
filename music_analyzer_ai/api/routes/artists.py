"""
Artist analysis endpoints
"""

from fastapi import Query
from typing import List
import logging

from ..base_router import BaseRouter
from ..constants import MIN_ARTISTS_FOR_COMPARISON, MAX_ARTISTS_FOR_COMPARISON
from ..utils.router_helpers import validate_track_ids_count

logger = logging.getLogger(__name__)


class ArtistsRouter(BaseRouter):
    """Router for artist analysis endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/artists", tags=["Artists"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all artist routes"""
        
        @self.router.post("/compare", response_model=dict)
        @self.handle_exceptions
        async def compare_artists(artist_ids: List[str]):
            """Compara múltiples artistas"""
            try:
                validate_track_ids_count(
                    artist_ids,
                    MIN_ARTISTS_FOR_COMPARISON,
                    MAX_ARTISTS_FOR_COMPARISON,
                    "artistas"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            artist_comparator = self.get_service("artist_comparator")
            comparison = artist_comparator.compare_artists(artist_ids)
            return self.success_response({"comparison": comparison})
        
        @self.router.get("/evolution", response_model=dict)
        @self.handle_exceptions
        async def get_artist_evolution(artist_id: str = Query(...)):
            """Analiza la evolución de un artista a lo largo del tiempo"""
            artist_comparator = self.get_service("artist_comparator")
            evolution = artist_comparator.analyze_evolution(artist_id)
            return self.success_response({"evolution": evolution})


def get_artists_router() -> ArtistsRouter:
    """Factory function to get artists router"""
    return ArtistsRouter()

