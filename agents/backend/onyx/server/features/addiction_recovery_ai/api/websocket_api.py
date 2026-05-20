"""
WebSocket API for Real-time Recovery AI
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
import json
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

app = FastAPI(title="Recovery AI WebSocket API")

# Connection manager
class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    await manager.connect(websocket)
    
    try:
        from addiction_recovery_ai import create_ultra_fast_engine
        engine = create_ultra_fast_engine(use_gpu=False)  # Use CPU for WebSocket
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "predict_progress":
                # Predict progress
                features = message.get("features", {})
                start = time.time()
                progress = engine.predict_progress(features)
                elapsed = (time.time() - start) * 1000
                
                response = {
                    "type": "progress_result",
                    "progress": progress,
                    "percentage": f"{progress * 100:.1f}%",
                    "inference_time_ms": elapsed
                }
                
                await manager.send_personal_message(
                    json.dumps(response),
                    websocket
                )
            
            elif message_type == "analyze_sentiment":
                # Analyze sentiment
                text = message.get("text", "")
                start = time.time()
                sentiment = engine.analyze_sentiment(text)
                elapsed = (time.time() - start) * 1000
                
                response = {
                    "type": "sentiment_result",
                    "label": sentiment.get("label", "NEUTRAL"),
                    "score": sentiment.get("score", 0.5),
                    "inference_time_ms": elapsed
                }
                
                await manager.send_personal_message(
                    json.dumps(response),
                    websocket
                )
            
            elif message_type == "ping":
                # Ping/Pong
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}),
                    websocket
                )
            
            else:
                # Unknown message type
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

