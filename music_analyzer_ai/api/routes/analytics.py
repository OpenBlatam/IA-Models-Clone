"""
Analytics endpoints for system metrics
"""

import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class AnalyticsRouter(BaseRouter):
    """Router for analytics endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/analytics", tags=["Analytics"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all analytics routes"""
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_analytics():
            """Obtiene estadísticas y métricas del sistema"""
            analytics_service = self.get_service("analytics_service")
            stats = analytics_service.get_stats()
            return self.success_response({"analytics": stats})
        
        @self.router.post("/reset", response_model=dict)
        @self.handle_exceptions
        async def reset_analytics():
            """Resetea las estadísticas de analytics"""
            analytics_service = self.get_service("analytics_service")
            analytics_service.reset_stats()
            return self.success_response(None, message="Analytics reset successfully")


def get_analytics_router() -> AnalyticsRouter:
    """Factory function to get analytics router"""
    return AnalyticsRouter()

