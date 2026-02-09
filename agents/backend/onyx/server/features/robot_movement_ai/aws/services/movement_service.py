"""
Movement Service
================

Microservice for robot movement operations (stateless).
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_client import ServiceClientFactory

logger = logging.getLogger(__name__)


class MoveRequest(BaseModel):
    """Move request model."""
    x: float
    y: float
    z: float
    orientation: Optional[list] = None


class MovementService(BaseMicroservice):
    """Movement microservice."""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        if config is None:
            config = ServiceConfig(
                service_name="movement-service",
                service_version="1.0.0",
                port=8001
            )
        super().__init__(config)
        self._trajectory_client = None
        self._chat_client = None
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for movement service."""
        app = FastAPI(
            title="Movement Service",
            description="Robot movement microservice",
            version=self.config.service_version
        )
        return app
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies."""
        return {
            "trajectory_client": ServiceClientFactory.get_client("trajectory-service"),
            "chat_client": ServiceClientFactory.get_client("chat-service"),
        }
    
    def _setup_routes(self):
        """Setup movement service routes."""
        trajectory_client = ServiceClientFactory.get_client("trajectory-service")
        
        @self.app.post("/api/v1/move/to")
        async def move_to(request: MoveRequest):
            """Move robot to position."""
            try:
                # Get optimized trajectory from trajectory service
                trajectory_response = await trajectory_client.post(
                    "/api/v1/trajectory/optimize",
                    json={
                        "target": {"x": request.x, "y": request.y, "z": request.z},
                        "orientation": request.orientation
                    }
                )
                
                # Execute movement (would call robot driver)
                # For now, return success
                return {
                    "success": True,
                    "message": f"Moving to ({request.x}, {request.y}, {request.z})",
                    "trajectory": trajectory_response.get("trajectory")
                }
            except Exception as e:
                logger.error(f"Movement failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/move/stop")
        async def stop_movement():
            """Stop robot movement."""
            return {"success": True, "message": "Movement stopped"}
        
        @self.app.get("/api/v1/movement/status")
        async def get_movement_status():
            """Get movement status."""
            return {
                "status": "idle",
                "current_position": {"x": 0, "y": 0, "z": 0}
            }















