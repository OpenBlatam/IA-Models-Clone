"""
Covers and remixes endpoints
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class CoversRemixesRouter(BaseRouter):
    """Router for covers and remixes endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/covers", tags=["Covers & Remixes"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all covers/remixes routes"""
        
        @self.router.get("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_cover(track_id: str = Query(...)):
            """Analiza un cover y lo compara con el original"""
            cover_remix_analyzer = self.get_service("cover_remix_analyzer")
            analysis = cover_remix_analyzer.analyze_cover(track_id)
            return self.success_response({"analysis": analysis})
        
        @self.router.get("/find", response_model=dict)
        @self.handle_exceptions
        async def find_versions(
            track_id: str = Query(...),
            version_type: Optional[str] = Query(None, regex="^(cover|remix|both)$")
        ):
            """Encuentra covers y remixes de un track"""
            cover_remix_analyzer = self.get_service("cover_remix_analyzer")
            versions = cover_remix_analyzer.find_versions(track_id, version_type)
            return self.list_response(versions, key="versions")


def get_covers_remixes_router() -> CoversRemixesRouter:
    """Factory function to get covers/remixes router"""
    return CoversRemixesRouter()

