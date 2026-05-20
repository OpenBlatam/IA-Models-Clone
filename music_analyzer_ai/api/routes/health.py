"""
Health check endpoints
"""

import logging

from ..base_router import BaseRouter
from ..health import health_checker

logger = logging.getLogger(__name__)


class HealthRouter(BaseRouter):
    """Router for health check endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/health", tags=["Health"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all health routes"""
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def health_check():
            """Comprehensive health check"""
            try:
                from ...config.service_registry import get_service
                spotify_service = get_service("spotify_service")
                spotify_service._get_access_token()
                
                return self.success_response({
                    "status": "healthy",
                    "spotify_connection": "ok"
                })
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return self.success_response({
                    "status": "unhealthy",
                    "spotify_connection": "error",
                    "error": str(e)
                })
        
        @self.router.get("/detailed", response_model=dict)
        @self.handle_exceptions
        async def detailed_health_check():
            """Detailed health check with all services"""
            results = await health_checker.run_checks()
            return self.success_response(results)
        
        @self.router.get("/spotify", response_model=dict)
        @self.handle_exceptions
        async def spotify_health():
            """Spotify service health check"""
            result = await health_checker.check_spotify()
            return self.success_response(result)
        
        @self.router.get("/services", response_model=dict)
        @self.handle_exceptions
        async def services_health():
            """All services health check"""
            result = await health_checker.check_services()
            return self.success_response(result)


def get_health_router() -> HealthRouter:
    """Factory function to get health router"""
    return HealthRouter()

