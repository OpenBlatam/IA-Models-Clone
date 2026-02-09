"""
Chat Service
============

Microservice for chat-based robot control (stateless).
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_client import ServiceClientFactory

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatService(BaseMicroservice):
    """Chat microservice."""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        if config is None:
            config = ServiceConfig(
                service_name="chat-service",
                service_version="1.0.0",
                port=8003
            )
        super().__init__(config)
        self._movement_client = None
        self._trajectory_client = None
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for chat service."""
        app = FastAPI(
            title="Chat Service",
            description="Chat-based robot control microservice",
            version=self.config.service_version
        )
        return app
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies."""
        return {
            "movement_client": ServiceClientFactory.get_client("movement-service"),
            "trajectory_client": ServiceClientFactory.get_client("trajectory-service"),
        }
    
    def _setup_routes(self):
        """Setup chat service routes."""
        movement_client = ServiceClientFactory.get_client("movement-service")
        
        @self.app.post("/api/v1/chat")
        async def process_chat(message: ChatMessage):
            """Process chat message."""
            try:
                # Parse command
                command = message.message.lower()
                
                if "move to" in command or "move" in command:
                    # Extract coordinates (simplified)
                    # In production, use NLP/LLM
                    coords = self._parse_coordinates(command)
                    
                    if coords:
                        # Call movement service
                        result = await movement_client.post(
                            "/api/v1/move/to",
                            json={"x": coords[0], "y": coords[1], "z": coords[2]}
                        )
                        return {
                            "success": True,
                            "message": f"Moving to {coords}",
                            "action": "move",
                            "data": result
                        }
                
                return {
                    "success": True,
                    "message": "Command processed",
                    "action": "unknown"
                }
            except Exception as e:
                logger.error(f"Chat processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket chat endpoint."""
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    # Process message
                    result = await process_chat(ChatMessage(message=data))
                    await websocket.send_json(result)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
    
    def _parse_coordinates(self, command: str) -> Optional[list]:
        """Parse coordinates from command (simplified)."""
        # Simplified parsing - in production use NLP
        import re
        pattern = r'\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)'
        match = re.search(pattern, command)
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        return None















