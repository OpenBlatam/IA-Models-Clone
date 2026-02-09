"""
Universal Robot API
===================

API para controlar cualquier tipo de robot.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.robot_types import RobotType, RobotConfig
from ..core.universal_controller import get_universal_engine
from ..core.adaptive_models import get_adaptive_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/universal-robots", tags=["Universal Robots"])


# Request/Response Models
class RegisterRobotRequest(BaseModel):
    """Request para registrar robot."""
    robot_id: str = Field(..., description="ID único del robot")
    robot_type: str = Field(..., description="Tipo de robot")
    name: str = Field(..., description="Nombre del robot")
    dof: int = Field(..., description="Grados de libertad")
    joint_limits: Dict[str, List[float]] = Field(default_factory=dict, description="Límites de articulaciones")
    link_lengths: List[float] = Field(default_factory=list, description="Longitudes de eslabones")
    base_position: List[float] = Field(default=[0.0, 0.0, 0.0], description="Posición base")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class MoveToRequest(BaseModel):
    """Request para mover robot."""
    robot_id: str = Field(..., description="ID del robot")
    target_position: List[float] = Field(..., description="Posición objetivo [x, y, z]")
    target_orientation: Optional[List[float]] = Field(None, description="Orientación objetivo (quaternion)")
    optimize_trajectory: bool = Field(default=True, description="Optimizar trayectoria")


class MovePathRequest(BaseModel):
    """Request para mover a lo largo de ruta."""
    robot_id: str = Field(..., description="ID del robot")
    waypoints: List[List[float]] = Field(..., description="Lista de waypoints")
    orientations: Optional[List[List[float]]] = Field(None, description="Lista de orientaciones")


# Endpoints
@router.post("/register", response_model=Dict[str, Any])
async def register_robot(request: RegisterRobotRequest):
    """Registrar nuevo robot."""
    try:
        engine = get_universal_engine()
        
        # Convertir joint_limits
        joint_limits = {}
        for key, value in request.joint_limits.items():
            if len(value) == 2:
                joint_limits[key] = (float(value[0]), float(value[1]))
        
        config = RobotConfig(
            robot_type=RobotType(request.robot_type),
            name=request.name,
            dof=request.dof,
            joint_limits=joint_limits,
            link_lengths=request.link_lengths,
            base_position=request.base_position,
            metadata=request.metadata
        )
        
        success = engine.register_robot(request.robot_id, config)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register robot")
        
        return {
            "robot_id": request.robot_id,
            "robot_type": request.robot_type,
            "status": "registered"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connect/{robot_id}", response_model=Dict[str, Any])
async def connect_robot(robot_id: str):
    """Conectar con robot."""
    try:
        engine = get_universal_engine()
        success = engine.connect_robot(robot_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to connect robot")
        
        return {
            "robot_id": robot_id,
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Error connecting robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move", response_model=Dict[str, Any])
async def move_robot(request: MoveToRequest):
    """Mover robot a posición objetivo."""
    try:
        engine = get_universal_engine()
        
        success = engine.move_robot(
            request.robot_id,
            request.target_position,
            request.target_orientation
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to move robot")
        
        return {
            "robot_id": request.robot_id,
            "target_position": request.target_position,
            "status": "moved"
        }
    except Exception as e:
        logger.error(f"Error moving robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move-path", response_model=Dict[str, Any])
async def move_along_path(request: MovePathRequest):
    """Mover robot a lo largo de ruta."""
    try:
        engine = get_universal_engine()
        
        controller = engine.robots.get(request.robot_id)
        if not controller:
            raise HTTPException(status_code=404, detail="Robot not found")
        
        success = controller.move_along_path(
            request.waypoints,
            request.orientations
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to move along path")
        
        return {
            "robot_id": request.robot_id,
            "waypoints": len(request.waypoints),
            "status": "completed"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving along path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=Dict[str, Any])
async def list_robots():
    """Listar robots registrados."""
    try:
        engine = get_universal_engine()
        robot_ids = engine.list_robots()
        
        robots = []
        for robot_id in robot_ids:
            robot_type = engine.get_robot_type(robot_id)
            state = engine.get_robot_state(robot_id)
            
            robots.append({
                "robot_id": robot_id,
                "robot_type": robot_type.value if robot_type else "unknown",
                "connected": state is not None and engine.robots[robot_id].robot.is_connected()
            })
        
        return {
            "robots": robots,
            "total": len(robots)
        }
    except Exception as e:
        logger.error(f"Error listing robots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{robot_id}", response_model=Dict[str, Any])
async def get_robot_state(robot_id: str):
    """Obtener estado del robot."""
    try:
        engine = get_universal_engine()
        state = engine.get_robot_state(robot_id)
        
        if state is None:
            raise HTTPException(status_code=404, detail="Robot not found")
        
        return {
            "robot_id": robot_id,
            "joint_positions": state.joint_positions,
            "joint_velocities": state.joint_velocities,
            "end_effector_position": state.end_effector_position,
            "base_position": state.base_position,
            "timestamp": state.timestamp
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting robot state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-model/{robot_id}", response_model=Dict[str, Any])
async def create_adaptive_model(robot_id: str, robot_type: str):
    """Crear modelo adaptativo para robot."""
    try:
        manager = get_adaptive_model_manager()
        model_id = manager.create_model_for_robot(
            RobotType(robot_type),
            robot_id
        )
        
        return {
            "robot_id": robot_id,
            "model_id": model_id,
            "robot_type": robot_type,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating adaptive model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

