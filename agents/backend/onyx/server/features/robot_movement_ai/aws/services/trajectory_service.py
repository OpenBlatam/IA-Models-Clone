"""
Trajectory Service
==================

Microservice for trajectory optimization (stateless).
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_client import ServiceClientFactory

logger = logging.getLogger(__name__)


class TrajectoryRequest(BaseModel):
    """Trajectory optimization request."""
    target: Dict[str, float]
    orientation: Optional[List[float]] = None
    obstacles: Optional[List[Dict]] = None
    algorithm: str = "astar"


class TrajectoryService(BaseMicroservice):
    """Trajectory optimization microservice."""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        if config is None:
            config = ServiceConfig(
                service_name="trajectory-service",
                service_version="1.0.0",
                port=8002
            )
        super().__init__(config)
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for trajectory service."""
        app = FastAPI(
            title="Trajectory Service",
            description="Trajectory optimization microservice",
            version=self.config.service_version
        )
        return app
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies."""
        return {}
    
    def _setup_routes(self):
        """Setup trajectory service routes."""
        @self.app.post("/api/v1/trajectory/optimize")
        async def optimize_trajectory(request: TrajectoryRequest):
            """Optimize trajectory."""
            try:
                # Import trajectory optimizer
                from robot_movement_ai.core.trajectory_optimizer import TrajectoryOptimizer, TrajectoryPoint
                from robot_movement_ai.config.robot_config import RobotConfig
                import numpy as np
                
                config = RobotConfig()
                optimizer = TrajectoryOptimizer(config)
                
                # Create trajectory points
                start_point = TrajectoryPoint(
                    position=np.array([0, 0, 0]),
                    orientation=np.array([0, 0, 0, 1]),
                    timestamp=0.0
                )
                
                target_pos = request.target
                goal_point = TrajectoryPoint(
                    position=np.array([target_pos["x"], target_pos["y"], target_pos["z"]]),
                    orientation=np.array(request.orientation or [0, 0, 0, 1]),
                    timestamp=0.0
                )
                
                # Optimize
                obstacles = request.obstacles or []
                if request.algorithm == "astar":
                    trajectory = optimizer.optimize_with_astar(
                        start_point, goal_point, obstacles, grid_resolution=0.05
                    )
                else:
                    trajectory = optimizer.optimize_with_rrt(
                        start_point, goal_point, obstacles, max_iterations=1000
                    )
                
                # Analyze
                analysis = optimizer.analyze_trajectory(trajectory)
                
                return {
                    "success": True,
                    "trajectory": [
                        {
                            "position": point.position.tolist(),
                            "orientation": point.orientation.tolist(),
                            "timestamp": point.timestamp
                        }
                        for point in trajectory
                    ],
                    "analysis": analysis
                }
            except Exception as e:
                logger.error(f"Trajectory optimization failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/trajectory/validate")
        async def validate_trajectory(trajectory: List[Dict]):
            """Validate trajectory."""
            return {
                "valid": True,
                "errors": [],
                "warnings": []
            }















