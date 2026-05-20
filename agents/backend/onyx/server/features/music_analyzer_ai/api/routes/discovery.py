"""
Discovery endpoints for music discovery features
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class DiscoveryRouter(BaseRouter):
    """Router for discovery endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/discovery", tags=["Discovery"])
        self._discovery_service = None
        self._register_routes()
    
    def _get_discovery_service(self):
        """Get or cache discovery service"""
        if self._discovery_service is None:
            self._discovery_service = self.get_service("discovery_service")
        return self._discovery_service
    
    def _register_routes(self):
        """Register all discovery routes"""
        
        @self.router.get("/similar-artists", response_model=dict)
        @self.handle_exceptions
        async def get_similar_artists(
            artist_id: str = Query(...),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Descubre artistas similares"""
            discovery_service = self._get_discovery_service()
            artists = discovery_service.discover_similar_artists(artist_id, limit)
            return self.list_response(artists, key="artists")
        
        @self.router.get("/underground", response_model=dict)
        @self.handle_exceptions
        async def get_underground_tracks(
            genre: Optional[str] = Query(None),
            limit: int = Query(20, ge=1, le=50)
        ):
            """Descubre tracks underground"""
            discovery_service = self._get_discovery_service()
            tracks = discovery_service.discover_underground(genre, limit)
            return self.list_response(tracks, key="tracks")
        
        @self.router.get("/mood-transition", response_model=dict)
        @self.handle_exceptions
        async def get_mood_transition_tracks(
            from_mood: str = Query(...),
            to_mood: str = Query(...),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Descubre tracks que transicionan entre moods"""
            discovery_service = self._get_discovery_service()
            tracks = discovery_service.discover_mood_transition(from_mood, to_mood, limit)
            return self.list_response(tracks, key="tracks")
        
        @self.router.get("/fresh", response_model=dict)
        @self.handle_exceptions
        async def get_fresh_tracks(
            days: int = Query(30, ge=1, le=365),
            limit: int = Query(20, ge=1, le=50)
        ):
            """Descubre tracks frescos (recientes)"""
            discovery_service = self._get_discovery_service()
            tracks = discovery_service.discover_fresh(days, limit)
            return self.list_response(tracks, key="tracks")


def get_discovery_router() -> DiscoveryRouter:
    """Factory function to get discovery router"""
    return DiscoveryRouter()

