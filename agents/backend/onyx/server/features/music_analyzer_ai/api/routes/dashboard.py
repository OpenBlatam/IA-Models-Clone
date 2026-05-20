"""
Dashboard endpoints
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class DashboardRouter(BaseRouter):
    """Router for dashboard endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/dashboard", tags=["Dashboard"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all dashboard routes"""
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_dashboard(user_id: Optional[str] = Query(None)):
            """Obtiene el dashboard completo con métricas del sistema y usuario"""
            dashboard_service = self.get_service("dashboard_service")
            dashboard = dashboard_service.get_dashboard(user_id)
            return self.success_response({"dashboard": dashboard})


def get_dashboard_router() -> DashboardRouter:
    """Factory function to get dashboard router"""
    return DashboardRouter()

