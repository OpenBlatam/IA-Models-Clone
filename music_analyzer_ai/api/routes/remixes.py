"""
Remixes endpoints
"""

from fastapi import Query
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class RemixesRouter(BaseRouter):
    """Router for remixes endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/remixes", tags=["Remixes"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all remixes routes"""
        
        @self.router.get("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_remix(track_id: str = Query(...)):
            """Analiza un remix y lo compara con el original"""
            cover_remix_analyzer = self.get_service("cover_remix_analyzer")
            analysis = cover_remix_analyzer.analyze_remix(track_id)
            return self.success_response({"analysis": analysis})


def get_remixes_router() -> RemixesRouter:
    """Factory function to get remixes router"""
    return RemixesRouter()

