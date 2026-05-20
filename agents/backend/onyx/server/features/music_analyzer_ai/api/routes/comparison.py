"""
Comparison endpoints for multiple tracks
"""

from typing import List
import logging

from ..base_router import BaseRouter
from ..constants import MIN_TRACKS_FOR_COMPARISON, MAX_TRACKS_FOR_COMPARISON
from ..utils.router_helpers import validate_track_ids_count

logger = logging.getLogger(__name__)


class ComparisonRouter(BaseRouter):
    """Router for comparison endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/compare", tags=["Comparison"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all comparison routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def compare_tracks(track_ids: List[str]):
            """
            Compara múltiples canciones
            
            - **track_ids**: Lista de IDs de canciones de Spotify (mínimo 2, máximo 10)
            """
            try:
                validate_track_ids_count(
                    track_ids,
                    MIN_TRACKS_FOR_COMPARISON,
                    MAX_TRACKS_FOR_COMPARISON,
                    "canciones"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            comparison_service = self.get_service("comparison_service")
            comparison = comparison_service.compare_tracks(track_ids)
            
            return self.success_response(comparison)


def get_comparison_router() -> ComparisonRouter:
    """Factory function to get comparison router"""
    return ComparisonRouter()

