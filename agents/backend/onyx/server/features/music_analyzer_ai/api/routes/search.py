"""
Search endpoints for music tracks
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter
from ...models.schemas import TrackSearchRequest
from ...utils.exceptions import SpotifyAPIException
from ..utils.response_formatters import format_tracks_response
from ..constants import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT
from ..validators.request_validators import validate_limit

logger = logging.getLogger(__name__)


class SearchRouter(BaseRouter):
    """Router for search endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/search", tags=["Search"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all search routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def search_track(request: TrackSearchRequest):
            """
            Busca canciones en Spotify
            
            - **query**: Nombre de la canción o artista
            - **limit**: Número máximo de resultados (1-50)
            """
            spotify_service = self.get_service("spotify_service")
            limit = validate_limit(request.limit, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT)
            tracks = spotify_service.search_track(request.query, limit)
            
            results = format_tracks_response(tracks)
            
            return self.list_response(
                results,
                key="results",
                query=request.query
            )


def get_search_router() -> SearchRouter:
    """Factory function to get search router"""
    return SearchRouter()

