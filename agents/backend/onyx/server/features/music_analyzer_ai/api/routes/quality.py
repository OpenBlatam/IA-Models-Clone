"""
Quality analysis endpoints for production quality assessment
"""

from fastapi import Query
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class QualityRouter(BaseRouter):
    """Router for quality analysis endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/quality", tags=["Quality Analysis"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all quality routes"""
        
        @self.router.get("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_quality(track_id: str = Query(...)):
            """Analiza la calidad de producción de un track"""
            quality_analyzer = self.get_service("quality_analyzer")
            quality = quality_analyzer.analyze_production_quality(track_id)
            return self.success_response({"quality_analysis": quality})


def get_quality_router() -> QualityRouter:
    """Factory function to get quality router"""
    return QualityRouter()

