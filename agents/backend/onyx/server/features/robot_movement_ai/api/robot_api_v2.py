"""
Robot API v2 - Nueva Arquitectura
==================================

API RESTful refactorizada usando la nueva arquitectura mejorada.
Implementa Clean Architecture con Use Cases y Dependency Injection.
"""

import logging
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Nueva arquitectura
from ..core.architecture.di_setup import (
    setup_dependency_injection,
    create_dependency,
    get_move_robot_use_case,
    get_robot_status_use_case,
    get_movement_history_use_case
)
from ..core.architecture.application_layer import (
    MoveRobotUseCase,
    GetRobotStatusUseCase,
    GetMovementHistoryUseCase,
    MoveRobotCommand,
    GetRobotStatusQuery,
    GetMovementHistoryQuery,
    ApplicationError,
    ErrorCode
)
from ..core.architecture.error_handling import (
    handle_error,
    ErrorContext,
    ErrorHandler,
    get_error_handler
)
from ..core.architecture.circuit_breaker import (
    circuit_breaker,
    CircuitBreakerConfig,
    get_circuit_breaker_manager
)

logger = logging.getLogger(__name__)

# Crear app FastAPI
app = FastAPI(
    title="Robot Movement AI API v2",
    description="API RESTful con arquitectura mejorada",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class MoveRobotRequest(BaseModel):
    """Request para mover robot."""
    robot_id: str = Field(..., description="ID del robot")
    target_x: float = Field(..., ge=-10.0, le=10.0, description="Coordenada X")
    target_y: float = Field(..., ge=-10.0, le=10.0, description="Coordenada Y")
    target_z: float = Field(..., ge=-10.0, le=10.0, description="Coordenada Z")
    target_qx: Optional[float] = Field(None, description="Orientación QX")
    target_qy: Optional[float] = Field(None, description="Orientación QY")
    target_qz: Optional[float] = Field(None, description="Orientación QZ")
    target_qw: Optional[float] = Field(None, description="Orientación QW")
    user_id: Optional[str] = Field(None, description="ID del usuario")


class MoveRobotResponse(BaseModel):
    """Response de movimiento."""
    success: bool
    movement_id: str
    robot_id: str
    status: str
    message: Optional[str] = None


class RobotStatusResponse(BaseModel):
    """Response de estado del robot."""
    robot_id: str
    brand: str
    model: str
    is_connected: bool
    current_position: Optional[dict] = None
    has_active_movements: bool


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Configurar sistema al iniciar."""
    logger.info("Inicializando Robot Movement AI API v2...")
    
    # Configurar Dependency Injection
    repo_type = os.getenv("REPOSITORY_TYPE", "in_memory")
    await setup_dependency_injection({
        'repository_type': repo_type,
        'enable_event_bus': os.getenv("ENABLE_EVENT_BUS", "true").lower() == "true"
    })
    
    logger.info(f"Dependency Injection configurado (repository_type: {repo_type})")
    logger.info("API v2 lista")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar."""
    logger.info("Cerrando Robot Movement AI API v2...")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ApplicationError)
async def application_error_handler(request, exc: ApplicationError):
    """Manejar errores de aplicación."""
    error_handler = get_error_handler()
    error_details = error_handler.handle_error(exc)
    
    status_code = 500
    if exc.code == ErrorCode.APPLICATION_NOT_FOUND:
        status_code = 404
    elif exc.code == ErrorCode.APPLICATION_UNAUTHORIZED:
        status_code = 401
    elif exc.code == ErrorCode.APPLICATION_FORBIDDEN:
        status_code = 403
    elif exc.code == ErrorCode.ROBOT_NOT_CONNECTED:
        status_code = 400
    
    return JSONResponse(
        status_code=status_code,
        content=error_handler.create_error_response(error_details)
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception):
    """Manejar errores generales."""
    error_handler = get_error_handler()
    context = ErrorContext(
        operation=request.url.path,
        request_id=getattr(request.state, 'request_id', None)
    )
    error_details = error_handler.handle_error(exc, context)
    
    return JSONResponse(
        status_code=500,
        content=error_handler.create_error_response(error_details, include_stack_trace=False)
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/api/v2/robots/{robot_id}/move", response_model=MoveRobotResponse)
async def move_robot_v2(
    robot_id: str,
    request: MoveRobotRequest,
    use_case: MoveRobotUseCase = Depends(create_dependency(MoveRobotUseCase))
):
    """
    Mover robot a posición específica.
    
    Usa la nueva arquitectura con Use Cases y Dependency Injection.
    """
    try:
        command = MoveRobotCommand(
            robot_id=robot_id,
            target_x=request.target_x,
            target_y=request.target_y,
            target_z=request.target_z,
            target_qx=request.target_qx,
            target_qy=request.target_qy,
            target_qz=request.target_qz,
            target_qw=request.target_qw,
            user_id=request.user_id
        )
        
        result = await use_case.execute(command)
        
        return MoveRobotResponse(
            success=True,
            movement_id=result.movement_id,
            robot_id=result.robot_id,
            status=result.status,
            message="Movement initiated successfully"
        )
    
    except ApplicationError as e:
        # Re-lanzar para que el handler lo maneje
        raise
    except Exception as e:
        context = ErrorContext(
            operation="move_robot",
            robot_id=robot_id,
            user_id=request.user_id
        )
        error_details = handle_error(e, context)
        raise HTTPException(
            status_code=500,
            detail=error_details.message
        )


@app.get("/api/v2/robots/{robot_id}/status", response_model=RobotStatusResponse)
async def get_robot_status_v2(
    robot_id: str,
    use_case: GetRobotStatusUseCase = Depends(create_dependency(GetRobotStatusUseCase))
):
    """
    Obtener estado del robot.
    
    Usa la nueva arquitectura con Use Cases.
    """
    try:
        query = GetRobotStatusQuery(robot_id=robot_id)
        status = await use_case.execute(query)
        
        return RobotStatusResponse(
            robot_id=status.robot_id,
            brand=status.brand,
            model=status.model,
            is_connected=status.is_connected,
            current_position=status.current_position,
            has_active_movements=status.has_active_movements
        )
    
    except ApplicationError as e:
        raise
    except Exception as e:
        context = ErrorContext(
            operation="get_robot_status",
            robot_id=robot_id
        )
        error_details = handle_error(e, context)
        raise HTTPException(
            status_code=500,
            detail=error_details.message
        )


@app.get("/api/v2/robots/{robot_id}/movements")
async def get_movement_history_v2(
    robot_id: str,
    limit: int = 100,
    use_case: GetMovementHistoryUseCase = Depends(create_dependency(GetMovementHistoryUseCase))
):
    """
    Obtener historial de movimientos del robot.
    
    Usa la nueva arquitectura con Use Cases.
    """
    try:
        query = GetMovementHistoryQuery(
            robot_id=robot_id,
            limit=limit
        )
        movements = await use_case.execute(query)
        
        return {
            "robot_id": robot_id,
            "count": len(movements),
            "movements": [movement.__dict__ for movement in movements]
        }
    
    except ApplicationError as e:
        raise
    except Exception as e:
        context = ErrorContext(
            operation="get_movement_history",
            robot_id=robot_id
        )
        error_details = handle_error(e, context)
        raise HTTPException(
            status_code=500,
            detail=error_details.message
        )


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "improved"
    }


# ============================================================================
# Circuit Breaker Status
# ============================================================================

@app.get("/api/v2/circuit-breakers")
async def get_circuit_breakers():
    """Obtener estado de todos los circuit breakers."""
    manager = get_circuit_breaker_manager()
    circuits = await manager.list_all()
    
    return {
        "count": len(circuits),
        "circuits": [circuit.get_state_info() for circuit in circuits]
    }


# ============================================================================
# Main
# ============================================================================

def run_api_v2(host: str = "0.0.0.0", port: int = 8011, reload: bool = False):
    """
    Ejecutar API v2.
    
    Args:
        host: Host del servidor
        port: Puerto (default: 8011 para no conflictar con v1)
        reload: Habilitar auto-reload
    """
    uvicorn.run(
        "robot_movement_ai.api.robot_api_v2:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_api_v2()




