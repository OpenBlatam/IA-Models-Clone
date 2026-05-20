"""
Instrumentation analysis endpoints
"""

from fastapi import Query
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class InstrumentationRouter(BaseRouter):
    """Router for instrumentation endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/instrumentation", tags=["Instrumentation"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all instrumentation routes"""
        
        @self.router.get("/analyze", response_model=dict)
        @self.handle_exceptions
        async def analyze_instrumentation(track_id: str = Query(...)):
            """Analiza la instrumentación de un track"""
            instrumentation_analyzer = self.get_service("instrumentation_analyzer")
            analysis = instrumentation_analyzer.analyze_instrumentation(track_id)
            return self.success_response({"analysis": analysis})


def get_instrumentation_router() -> InstrumentationRouter:
    """Factory function to get instrumentation router"""
    return InstrumentationRouter()

