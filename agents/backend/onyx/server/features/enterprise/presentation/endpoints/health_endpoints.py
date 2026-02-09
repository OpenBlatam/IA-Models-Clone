from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter
from ...core.interfaces.health_interface import IHealthService
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Health Check Endpoints
=====================

Health check API endpoints.
"""



class HealthEndpoints:
    """Health check endpoints."""
    
    def __init__(self, health_service: IHealthService):
        
    """__init__ function."""
self.health_service = health_service
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self) -> Any:
        """Setup health check routes."""
        
        @self.router.get("/health")
        async def health_check():
            """Comprehensive health check."""
            health_status = await self.health_service.get_health_status()
            
            status_code = 200 if health_status.is_ready() else 503
            
            return {
                "status": health_status.overall_state.value,
                "timestamp": health_status.timestamp.isoformat(),
                "version": health_status.version,
                "checks": {
                    name: {
                        "status": comp.state.value,
                        "message": comp.message,
                        "last_check": comp.last_check.isoformat(),
                        "details": comp.details
                    }
                    for name, comp in health_status.components.items()
                }
            }
        
        @self.router.get("/health/live")
        async def liveness_check():
            """Kubernetes liveness probe."""
            is_alive = await self.health_service.check_liveness()
            
            if is_alive:
                return {"status": "alive"}
            else:
                return {"status": "dead"}
        
        @self.router.get("/health/ready")
        async def readiness_check():
            """Kubernetes readiness probe."""
            is_ready = await self.health_service.check_readiness()
            
            status_code = 200 if is_ready else 503
            
            return {
                "status": "ready" if is_ready else "not_ready"
            } 