"""
Robot API - FastAPI endpoints
==============================

API RESTful para control de robots mediante chat y comandos directos.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..config.robot_config import RobotConfig
from ..core.robot import RobotMovementEngine
from ..chat.chat_controller import ChatRobotController
from .middleware import RequestLoggingMiddleware, PerformanceMonitoringMiddleware
from .security_middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware
)
from ..core.exceptions import (
    BaseRobotException,
    RobotMovementError,
    TrajectoryError,
    IKError,
    RobotConnectionError,
    RobotNotConnectedError,
    RobotMovementInProgressError,
    ValidationError,
    ConfigurationError,
    SafetyError,
    CollisionDetectedError,
    SafetyLimitExceededError,
    EmergencyStopError
)
from .router_registry import get_router_registry

logger = logging.getLogger(__name__)


# Pydantic models with validation
class MoveToRequest(BaseModel):
    """Request para mover a posición."""
    x: float = Field(..., description="X coordinate", ge=-10.0, le=10.0)
    y: float = Field(..., description="Y coordinate", ge=-10.0, le=10.0)
    z: float = Field(..., description="Z coordinate", ge=-10.0, le=10.0)
    orientation: Optional[List[float]] = Field(
        None,
        description="Orientation quaternion [qx, qy, qz, qw]",
        min_length=4,
        max_length=4
    )
    
    @classmethod
    def validate_orientation(cls, v):
        """Validar quaternion."""
        if v is not None:
            from ..core.validation_utils import validate_quaternion
            validate_quaternion(v, name="orientation", tolerance=0.1)
        return v


class ChatMessage(BaseModel):
    """Mensaje de chat."""
    message: str = Field(
        ...,
        description="Chat message or command",
        min_length=1,
        max_length=10000
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context",
        max_items=50
    )
    
    @classmethod
    def validate_message(cls, v):
        """Validar mensaje."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Respuesta de chat."""
    success: bool
    message: str
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class WaypointRequest(BaseModel):
    """Request para waypoint."""
    x: float = Field(..., ge=-10.0, le=10.0)
    y: float = Field(..., ge=-10.0, le=10.0)
    z: float = Field(..., ge=-10.0, le=10.0)
    orientation: Optional[List[float]] = Field(None, min_length=4, max_length=4)


class PathRequest(BaseModel):
    """Request para ruta con múltiples waypoints."""
    waypoints: List[WaypointRequest] = Field(..., min_items=2, max_items=100)


class ObstaclesRequest(BaseModel):
    """Request para actualizar obstáculos."""
    obstacles: List[List[float]] = Field(
        ...,
        description="List of obstacles, each as [min_x, min_y, min_z, max_x, max_y, max_z]",
        max_items=1000
    )
    
    @classmethod
    def validate_obstacles(cls, v):
        """Validar formato de obstáculos."""
        if not v:
            return v
        for obs in v:
            if len(obs) != 6:
                raise ValueError("Each obstacle must have 6 coordinates [min_x, min_y, min_z, max_x, max_y, max_z]")
            min_coords = obs[:3]
            max_coords = obs[3:]
            if any(max_coords[i] < min_coords[i] for i in range(3)):
                raise ValueError("Max coordinates must be >= min coordinates for each axis")
        return v


def create_robot_app(config: RobotConfig) -> FastAPI:
    """Crear aplicación FastAPI."""
    app = FastAPI(
        title="Robot Movement AI API",
        description="Plataforma IA de Movimiento Robótico - Control mediante chat",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "movement", "description": "Robot movement operations"},
            {"name": "chat", "description": "Chat-based robot control"},
            {"name": "trajectory", "description": "Trajectory planning and optimization"},
            {"name": "health", "description": "Health checks and monitoring"},
            {"name": "status", "description": "Robot status and statistics"}
        ]
    )
    
    # Global exception handler
    @app.exception_handler(BaseRobotException)
    async def robot_exception_handler(request, exc: BaseRobotException):
        """Manejar excepciones del sistema de robots."""
        status_code = 500
        if isinstance(exc, (RobotNotConnectedError, RobotConnectionError)):
            status_code = 503
        elif isinstance(exc, RobotMovementInProgressError):
            status_code = 409
        elif isinstance(exc, (TrajectoryError, IKError, ValidationError)):
            status_code = 400
        elif isinstance(exc, SafetyError):
            status_code = 423  # Locked - safety issue
        elif isinstance(exc, ConfigurationError):
            status_code = 500
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": exc.to_dict(),
                "timestamp": asyncio.get_event_loop().time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Manejar excepciones generales."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                    "error_code": "INTERNAL_SERVER_ERROR"
                },
                "timestamp": asyncio.get_event_loop().time()
            }
        )
    
    # Middleware (orden importante - se ejecutan en orden inverso)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_request_size=10 * 1024 * 1024)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_size=10
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold_ms=1000.0)
    
    # CORS
    if config.api_cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Inicializar componentes
    movement_engine = RobotMovementEngine(config)
    chat_controller = ChatRobotController(movement_engine, config)
    
    # Estado global (en producción usaría dependency injection)
    app.state.movement_engine = movement_engine
    app.state.chat_controller = chat_controller
    app.state.config = config
    app.state.initialized = False
    
    # Registrar e incluir routers usando el registro
    router_registry = get_router_registry()
    
    # Routers principales (requeridos) - ya tienen prefix y tags definidos
    router_registry.register_lazy("api.metrics_api", required=True)
    router_registry.register_lazy("api.system_api", required=True)
    router_registry.register_lazy("api.health_api", required=True)
    
    # Routers opcionales - los prefix y tags ya están definidos en cada router
    optional_routers = [
        "api.resources_api",
        "api.analytics_api",
        "api.tasks_api",
        "api.notifications_api",
        "api.monitoring_api",
        "api.export_api",
        "api.security_api",
        "api.reports_api",
        "api.dashboards_api",
        "api.features_api",
        "api.knowledge_api",
        "api.collaboration_api",
        "api.version_api",
        "api.webhook_api",
        "api.graphql_api",
        "api.streaming_api",
        "api.cache_api",
        "api.load_balancer_api",
        "api.service_api",
        "api.queue_api",
        "api.rate_limit_api",
        "api.retry_api",
        "api.resource_api",
        "api.observability_api",
        "api.distributed_api",
        "api.task_api",
        "api.pattern_api",
        "api.microservices_api",
        "api.data_sync_api",
        "api.deep_learning_api",
        "api.llm_api",
        "api.universal_robot_api",
    ]
    
    for module_path in optional_routers:
        router_registry.register_lazy(module_path, required=False)
    
    # Incluir todos los routers registrados (prefix y tags ya están en los routers)
    for router_info in router_registry.get_routers():
        app.include_router(router_info["router"])
    
    # Log routers fallidos (si los hay)
    failed = router_registry.get_failed_routers()
    if failed:
        logger.warning(f"Some optional routers failed to load: {failed}")
    
    @app.on_event("startup")
    async def startup():
        """Inicializar al arrancar."""
        try:
            await movement_engine.initialize()
            app.state.initialized = True
            logger.info("API started and robot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            app.state.initialized = False
            raise
    
    @app.on_event("shutdown")
    async def shutdown():
        """Limpiar al apagar."""
        await movement_engine.shutdown()
        logger.info("API shut down")
    
    @app.get("/")
    async def root():
        """Endpoint raíz."""
        return {
            "name": "Robot Movement AI API",
            "version": "1.0.0",
            "status": "running",
            "robot_brand": config.robot_brand.value,
            "ros_enabled": config.ros_enabled
        }
    
    @app.get("/health")
    async def health():
        """Health check mejorado."""
        from ..core.health_check import get_health_check_system
        
        health_system = get_health_check_system()
        health_report = health_system.get_health_report()
        
        # Agregar información adicional
        status = movement_engine.get_status()
        feedback_stats = movement_engine.feedback_system.get_statistics()
        
        health_report["robot"] = {
            "initialized": app.state.initialized,
            "status": status,
            "feedback_stats": feedback_stats
        }
        
        return health_report
    
    @app.post("/api/v1/move/to", response_model=Dict[str, Any])
    async def move_to(request: MoveToRequest):
        """Mover robot a posición absoluta."""
        if not app.state.initialized:
            raise RobotNotConnectedError(
                "Robot not initialized",
                error_code="ROBOT_NOT_INITIALIZED",
                details={"endpoint": "/api/v1/move/to"}
            )
        
        # Validate input using validation utilities
        from ..core.validation_utils import validate_position, validate_quaternion
        import numpy as np
        
        validate_position(
            [request.x, request.y, request.z],
            name="position",
            min_val=-10.0,
            max_val=10.0
        )
        
        orientation = None
        if request.orientation:
            validate_quaternion(request.orientation, name="orientation", tolerance=0.1)
            orientation = np.array(request.orientation)
        
        try:
            
            from ..core.inverse_kinematics import EndEffectorPose
            target_pose = EndEffectorPose(
                position=[request.x, request.y, request.z],
                orientation=orientation or [0.0, 0.0, 0.0, 1.0]
            )
            
            success = await movement_engine.move_to_pose(target_pose)
            
            return {
                "success": success,
                "message": f"Moving to ({request.x}, {request.y}, {request.z})",
                "target": {"x": request.x, "y": request.y, "z": request.z},
                "timestamp": asyncio.get_event_loop().time()
            }
        except BaseRobotException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in move_to: {e}", exc_info=True)
            raise RobotMovementError(
                f"Unexpected error during movement: {str(e)}",
                error_code="MOVEMENT_ERROR",
                cause=e
            )
    
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(message: ChatMessage):
        """Procesar mensaje de chat."""
        if not message.message or not message.message.strip():
            raise ValidationError(
                "Message cannot be empty",
                error_code="EMPTY_MESSAGE",
                details={"message_length": len(message.message) if message.message else 0}
            )
        
        if len(message.message) > 10000:
            raise ValidationError(
                "Message too long (max 10000 characters)",
                error_code="MESSAGE_TOO_LONG",
                details={"message_length": len(message.message)}
            )
        
        try:
            result = await chat_controller.process_chat_message(
                message.message,
                message.context
            )
            return ChatResponse(**result)
        except BaseRobotException:
            raise
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise RobotMovementError(
                f"Error processing chat message: {str(e)}",
                error_code="CHAT_ERROR",
                cause=e,
                details={"message_preview": message.message[:100]}
            )
    
    @app.post("/api/v1/stop")
    async def stop():
        """Detener movimiento del robot."""
        try:
            await movement_engine.stop_movement()
            return {
                "success": True,
                "message": "Robot stopped"
            }
        except Exception as e:
            logger.error(f"Error in stop: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/status")
    async def status():
        """Obtener estado del robot."""
        try:
            robot_status = movement_engine.get_status()
            feedback_stats = movement_engine.feedback_system.get_statistics()
            
            return {
                "robot_status": robot_status,
                "feedback_stats": feedback_stats,
                "config": config.to_dict()
            }
        except Exception as e:
            logger.error(f"Error in status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/statistics")
    async def statistics():
        """Obtener estadísticas detalladas del sistema."""
        try:
            engine_stats = movement_engine.get_statistics()
            chat_stats = chat_controller.get_statistics()
            optimizer_stats = movement_engine.trajectory_optimizer.get_statistics()
            
            return {
                "engine_statistics": engine_stats,
                "chat_statistics": chat_stats,
                "optimizer_statistics": optimizer_stats
            }
        except Exception as e:
            logger.error(f"Error in statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/move/path")
    async def move_path(request: PathRequest):
        """Mover robot a lo largo de una ruta con múltiples waypoints."""
        try:
            from ..core.inverse_kinematics import EndEffectorPose
            import numpy as np
            
            waypoints = []
            for wp in request.waypoints:
                orientation = np.array(wp.orientation) if wp.orientation else np.array([0.0, 0.0, 0.0, 1.0])
                waypoints.append(EndEffectorPose(
                    position=np.array([wp.x, wp.y, wp.z]),
                    orientation=orientation
                ))
            
            success = await movement_engine.move_along_path(waypoints)
            
            return {
                "success": success,
                "message": f"Moving along path with {len(waypoints)} waypoints",
                "waypoints_count": len(waypoints)
            }
        except Exception as e:
            logger.error(f"Error in move_path: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/obstacles/update")
    async def update_obstacles(request: ObstaclesRequest):
        """Actualizar lista de obstáculos conocidos."""
        try:
            import numpy as np
            
            obstacles = [np.array(obs) for obs in request.obstacles]
            movement_engine.update_obstacles(obstacles)
            
            return {
                "success": True,
                "message": f"Updated {len(obstacles)} obstacles",
                "obstacles_count": len(obstacles)
            }
        except Exception as e:
            logger.error(f"Error in update_obstacles: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/obstacles")
    async def get_obstacles():
        """Obtener lista de obstáculos conocidos."""
        try:
            obstacles = movement_engine.known_obstacles
            return {
                "obstacles": [obs.tolist() for obs in obstacles],
                "count": len(obstacles)
            }
        except Exception as e:
            logger.error(f"Error in get_obstacles: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/replanification/enable")
    async def enable_replanification(enabled: bool = True):
        """Habilitar/deshabilitar replanificación automática."""
        try:
            movement_engine.replanification_enabled = enabled
            return {
                "success": True,
                "message": f"Replanification {'enabled' if enabled else 'disabled'}",
                "replanification_enabled": enabled
            }
        except Exception as e:
            logger.error(f"Error in enable_replanification: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/movement/history")
    async def movement_history(limit: int = 10):
        """Obtener historial de movimientos."""
        try:
            history = movement_engine.movement_history[-limit:]
            return {
                "history": history,
                "total_movements": movement_engine.total_movements,
                "returned_count": len(history)
            }
        except Exception as e:
            logger.error(f"Error in movement_history: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/trajectory/optimize/astar")
    async def optimize_astar(request: MoveToRequest, grid_resolution: float = 0.05):
        """Optimizar trayectoria usando algoritmo A*."""
        try:
            from ..core.trajectory_optimizer import TrajectoryPoint
            from ..core.inverse_kinematics import EndEffectorPose
            import numpy as np
            
            start_feedback = movement_engine.feedback_system.get_latest_feedback()
            if not start_feedback:
                raise HTTPException(status_code=400, detail="No current position available")
            
            start_point = TrajectoryPoint(
                position=start_feedback.end_effector_position,
                orientation=start_feedback.end_effector_orientation,
                timestamp=0.0
            )
            
            orientation = np.array(request.orientation) if request.orientation else np.array([0.0, 0.0, 0.0, 1.0])
            goal_point = TrajectoryPoint(
                position=np.array([request.x, request.y, request.z]),
                orientation=orientation,
                timestamp=0.0
            )
            
            trajectory = movement_engine.trajectory_optimizer.optimize_with_astar(
                start_point, goal_point, movement_engine.known_obstacles, grid_resolution
            )
            
            analysis = movement_engine.trajectory_optimizer.analyze_trajectory(trajectory)
            
            return {
                "success": True,
                "trajectory_points": len(trajectory),
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in optimize_astar: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/trajectory/optimize/rrt")
    async def optimize_rrt(request: MoveToRequest, max_iterations: int = 1000, step_size: float = 0.1):
        """Optimizar trayectoria usando algoritmo RRT."""
        try:
            from ..core.trajectory_optimizer import TrajectoryPoint
            import numpy as np
            
            start_feedback = movement_engine.feedback_system.get_latest_feedback()
            if not start_feedback:
                raise HTTPException(status_code=400, detail="No current position available")
            
            start_point = TrajectoryPoint(
                position=start_feedback.end_effector_position,
                orientation=start_feedback.end_effector_orientation,
                timestamp=0.0
            )
            
            orientation = np.array(request.orientation) if request.orientation else np.array([0.0, 0.0, 0.0, 1.0])
            goal_point = TrajectoryPoint(
                position=np.array([request.x, request.y, request.z]),
                orientation=orientation,
                timestamp=0.0
            )
            
            trajectory = movement_engine.trajectory_optimizer.optimize_with_rrt(
                start_point, goal_point, movement_engine.known_obstacles, max_iterations, step_size
            )
            
            analysis = movement_engine.trajectory_optimizer.analyze_trajectory(trajectory)
            
            return {
                "success": True,
                "trajectory_points": len(trajectory),
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in optimize_rrt: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/trajectory/analyze")
    async def analyze_trajectory(request: PathRequest):
        """Analizar trayectoria generada desde waypoints."""
        try:
            from ..core.trajectory_optimizer import TrajectoryPoint
            from ..core.inverse_kinematics import EndEffectorPose
            import numpy as np
            
            if len(request.waypoints) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 waypoints")
            
            # Generar trayectoria desde waypoints
            trajectory_points = []
            for i, wp in enumerate(request.waypoints):
                orientation = np.array(wp.orientation) if wp.orientation else np.array([0.0, 0.0, 0.0, 1.0])
                point = TrajectoryPoint(
                    position=np.array([wp.x, wp.y, wp.z]),
                    orientation=orientation,
                    timestamp=i * 0.01
                )
                trajectory_points.append(point)
            
            analysis = movement_engine.trajectory_optimizer.analyze_trajectory(trajectory_points)
            
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in analyze_trajectory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/trajectory/export")
    async def export_trajectory(request: PathRequest, format: str = "json"):
        """Exportar trayectoria a archivo."""
        try:
            from ..core.trajectory_optimizer import TrajectoryPoint
            import numpy as np
            from pathlib import Path
            
            if format not in ["json", "csv"]:
                raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
            
            # Generar trayectoria desde waypoints
            trajectory_points = []
            for i, wp in enumerate(request.waypoints):
                orientation = np.array(wp.orientation) if wp.orientation else np.array([0.0, 0.0, 0.0, 1.0])
                point = TrajectoryPoint(
                    position=np.array([wp.x, wp.y, wp.z]),
                    orientation=orientation,
                    timestamp=i * 0.01
                )
                trajectory_points.append(point)
            
            # Generar nombre de archivo
            filepath = f"trajectory_{int(asyncio.get_event_loop().time())}.{format}"
            filepath = Path(movement_engine.config.storage_path) / filepath
            
            success = movement_engine.trajectory_optimizer.export_trajectory(
                trajectory_points, str(filepath), format
            )
            
            if success:
                return {
                    "success": True,
                    "filepath": str(filepath),
                    "format": format
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to export trajectory")
        except Exception as e:
            logger.error(f"Error in export_trajectory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket para chat en tiempo real."""
        await websocket.accept()
        logger.info("WebSocket connection established")
        
        try:
            while True:
                data = await websocket.receive_text()
                
                # Procesar mensaje
                result = await chat_controller.process_chat_message(data)
                
                # Enviar respuesta
                await websocket.send_json(result)
        
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket: {e}", exc_info=True)
            await websocket.close()
    
    return app


class RobotAPI:
    """Wrapper para la API de robots."""
    
    def __init__(self, config: RobotConfig):
        """Inicializar API."""
        self.config = config
        self.app = create_robot_app(config)
    
    def run(self, host: str = "0.0.0.0", port: int = 8010, reload: bool = False):
        """Ejecutar servidor API."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level=self.config.log_level.lower()
        )

