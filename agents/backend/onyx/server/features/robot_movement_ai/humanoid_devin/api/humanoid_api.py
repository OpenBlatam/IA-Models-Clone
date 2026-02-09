"""
Humanoid API (optimizado)
==========================

API RESTful para control de robot humanoide mediante chat natural.
Incluye validaciones, manejo de errores robusto, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn

from ..drivers.humanoid_devin_driver import HumanoidDevinDriver
from ..core.humanoid_chat_controller import HumanoidChatController
from ..core.humanoid_movement_engine import HumanoidMovementEngine
from ..config import settings
from ..exceptions import (
    RobotConnectionError,
    RobotNotInitializedError,
    InvalidCommandError,
    MovementError
)

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """
    Mensaje de chat (optimizado).
    
    Incluye validaciones y sanitización.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Chat message or command",
        examples=["Walk forward 2 meters"]
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Valida y sanitiza el mensaje"""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v.strip()) > 5000:
            raise ValueError("Message cannot exceed 5000 characters")
        return v.strip()


class ChatResponse(BaseModel):
    """
    Respuesta de chat (optimizado).
    
    Incluye información detallada sobre la acción ejecutada.
    """
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    action: Optional[str] = Field(None, description="Acción ejecutada")
    data: Optional[Dict[str, Any]] = Field(None, description="Datos adicionales")


def create_humanoid_app(
    robot_ip: str = "192.168.1.100",
    robot_port: int = 30001,
    dof: int = 32,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4"
) -> FastAPI:
    """
    Crear aplicación FastAPI para robot humanoide.
    
    Args:
        robot_ip: IP del robot humanoide
        robot_port: Puerto del robot
        dof: Grados de libertad
        llm_provider: Proveedor de LLM
        llm_api_key: API key para LLM
        llm_model: Modelo de LLM
    """
    app = FastAPI(
        title="Humanoid Devin Robot API",
        description="Control de robot humanoide mediante chat natural",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    driver = HumanoidDevinDriver(robot_ip=robot_ip, robot_port=robot_port, dof=dof)
    movement_engine = HumanoidMovementEngine(driver)
    chat_controller = HumanoidChatController(
        driver=driver,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model
    )
    
    app.state.driver = driver
    app.state.movement_engine = movement_engine
    app.state.chat_controller = chat_controller
    app.state.initialized = False
    
    @app.on_event("startup")
    async def startup():
        """Inicializar al arrancar."""
        try:
            await movement_engine.initialize()
            app.state.initialized = True
            logger.info("Humanoid API started and robot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize humanoid robot: {e}")
            app.state.initialized = False
    
    @app.on_event("shutdown")
    async def shutdown():
        """Limpiar al apagar."""
        await movement_engine.shutdown()
        logger.info("Humanoid API shut down")
    
    @app.get("/")
    async def root():
        """Endpoint raíz."""
        return {
            "name": "Humanoid Devin Robot API",
            "version": "1.0.0",
            "status": "running",
            "dof": dof,
            "robot_ip": robot_ip
        }
    
    @app.get("/health")
    async def health():
        """Health check."""
        status = await driver.get_status() if driver.connected else None
        return {
            "status": "healthy" if app.state.initialized else "unhealthy",
            "robot_connected": driver.connected,
            "robot_status": status
        }
    
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(message: ChatMessage):
        """Procesar mensaje de chat."""
        try:
            result = await chat_controller.process_chat_message(
                message.message,
                message.context
            )
            return ChatResponse(**result)
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/stop")
    async def stop():
        """Detener movimiento del robot."""
        try:
            await driver.stop()
            return {
                "success": True,
                "message": "Humanoid robot stopped"
            }
        except Exception as e:
            logger.error(f"Error in stop: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/status")
    async def status():
        """Obtener estado del robot."""
        try:
            robot_status = await driver.get_status()
            engine_status = movement_engine.get_status()
            chat_stats = chat_controller.get_statistics()
            
            return {
                "robot_status": robot_status,
                "engine_status": engine_status,
                "chat_statistics": chat_stats
            }
        except Exception as e:
            logger.error(f"Error in status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/walk")
    async def walk(direction: str = "forward", distance: float = 1.0, speed: float = 0.5):
        """Caminar en dirección especificada."""
        try:
            success = await driver.walk(direction=direction, distance=distance, speed=speed)
            return {
                "success": success,
                "message": f"Walking {direction} for {distance}m",
                "direction": direction,
                "distance": distance
            }
        except Exception as e:
            logger.error(f"Error in walk: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/stand")
    async def stand():
        """Ponerse de pie."""
        try:
            success = await driver.stand()
            return {
                "success": success,
                "message": "Standing up"
            }
        except Exception as e:
            logger.error(f"Error in stand: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/sit")
    async def sit():
        """Sentarse."""
        try:
            success = await driver.sit()
            return {
                "success": success,
                "message": "Sitting down"
            }
        except Exception as e:
            logger.error(f"Error in sit: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/grasp")
    async def grasp(hand: str = "right"):
        """Agarrar objeto."""
        try:
            success = await driver.grasp(hand=hand)
            return {
                "success": success,
                "message": f"Grasping with {hand} hand",
                "hand": hand
            }
        except Exception as e:
            logger.error(f"Error in grasp: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/wave")
    async def wave(hand: str = "right"):
        """Saludar con la mano."""
        try:
            success = await driver.wave(hand=hand)
            return {
                "success": success,
                "message": f"Waving with {hand} hand",
                "hand": hand
            }
        except Exception as e:
            logger.error(f"Error in wave: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket para chat en tiempo real."""
        await websocket.accept()
        logger.info("WebSocket connection established for humanoid")
        
        try:
            while True:
                data = await websocket.receive_text()
                result = await chat_controller.process_chat_message(data)
                await websocket.send_json(result)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket: {e}", exc_info=True)
            await websocket.close()
    
    return app


class HumanoidAPI:
    """Wrapper para la API de robot humanoide."""
    
    def __init__(
        self,
        robot_ip: str = "192.168.1.100",
        robot_port: int = 30001,
        dof: int = 32,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4"
    ):
        """Inicializar API."""
        self.app = create_humanoid_app(
            robot_ip=robot_ip,
            robot_port=robot_port,
            dof=dof,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8020, reload: bool = False):
        """Ejecutar servidor API."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

