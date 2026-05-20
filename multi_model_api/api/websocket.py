"""
WebSocket endpoints for streaming responses
"""

import asyncio
import logging
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..core.models import ModelRegistry, get_registry

registry = get_registry()
from ..api.schemas import ModelConfig, ModelType
from ..core.consensus import apply_consensus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model/ws", tags=["WebSocket"])


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming multi-model responses
    
    Expected message format:
    {
        "prompt": "your prompt here",
        "models": [
            {"model_type": "gpt_5.1", "is_enabled": true, "temperature": 0.7}
        ],
        "strategy": "parallel",
        "consensus_method": "majority"
    }
    """
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            prompt = data.get("prompt")
            models_data = data.get("models", [])
            strategy = data.get("strategy", "parallel")
            consensus_method = data.get("consensus_method", "majority")
            
            if not prompt:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Prompt is required"
                }, websocket)
                continue
            
            if not models_data:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "At least one model is required"
                }, websocket)
                continue
            
            enabled_models = []
            for model_data in models_data:
                if model_data.get("is_enabled", True):
                    try:
                        model_type = ModelType(model_data["model_type"])
                        enabled_models.append(ModelConfig(
                            model_type=model_type,
                            is_enabled=True,
                            temperature=model_data.get("temperature"),
                            max_tokens=model_data.get("max_tokens"),
                            multiplier=model_data.get("multiplier", 1)
                        ))
                    except (ValueError, KeyError) as e:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Invalid model config: {e}"
                        }, websocket)
                        continue
            
            if not enabled_models:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "No enabled models found"
                }, websocket)
                continue
            
            await manager.send_personal_message({
                "type": "start",
                "message": f"Processing {len(enabled_models)} models with {strategy} strategy"
            }, websocket)
            
            tasks = []
            for model in enabled_models:
                task = registry.execute_model(
                    model.model_type,
                    prompt,
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                    timeout=30.0
                )
                tasks.append((model, task))
            
            responses = []
            for model, task in tasks:
                try:
                    response = await task
                    responses.append(response)
                    await manager.send_personal_message({
                        "type": "model_response",
                        "model_type": model.model_type.value,
                        "response": response.response if response.success else None,
                        "success": response.success,
                        "error": response.error,
                        "latency_ms": response.latency_ms,
                        "tokens_used": response.tokens_used
                    }, websocket)
                except Exception as e:
                    from ..api.schemas import ModelResponse
                    error_response = ModelResponse(
                        model_type=model.model_type,
                        response="",
                        success=False,
                        error=str(e),
                        latency_ms=None,
                        tokens_used=None
                    )
                    responses.append(error_response)
                    await manager.send_personal_message({
                        "type": "model_error",
                        "model_type": model.model_type.value,
                        "error": str(e)
                    }, websocket)
            
            if strategy == "consensus":
                successful_responses = [r for r in responses if getattr(r, 'success', False) and getattr(r, 'response', None)]
                if successful_responses:
                    weights = {m.model_type.value: m.multiplier for m, _ in tasks}
                    consensus_result = apply_consensus(
                        successful_responses,
                        consensus_method,
                        weights
                    )
                    await manager.send_personal_message({
                        "type": "consensus",
                        "result": consensus_result,
                        "method": consensus_method
                    }, websocket)
            
            await manager.send_personal_message({
                "type": "complete",
                "message": "All models processed"
            }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "message": str(e)
        }, websocket)
        manager.disconnect(websocket)

