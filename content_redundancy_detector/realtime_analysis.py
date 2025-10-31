"""
Real-time Analysis Engine with WebSocket Support
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import redis.asyncio as redis
from ai_ml_enhanced import ai_ml_engine
from config import settings

logger = logging.getLogger(__name__)


class RealTimeAnalysisRequest(BaseModel):
    """Request model for real-time analysis"""
    content: str = Field(..., min_length=1, max_length=10000, description="Content to analyze")
    analysis_types: List[str] = Field(default=["sentiment", "entities", "language"], description="Types of analysis to perform")
    stream_interval: float = Field(default=1.0, ge=0.1, le=10.0, description="Streaming interval in seconds")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class RealTimeAnalysisResult(BaseModel):
    """Result model for real-time analysis"""
    session_id: str
    timestamp: str
    analysis_type: str
    result: Dict[str, Any]
    progress: float = Field(ge=0.0, le=1.0, description="Analysis progress")
    status: str = Field(..., description="Analysis status")


class StreamingAnalysisRequest(BaseModel):
    """Request model for streaming analysis"""
    content_stream: List[str] = Field(..., min_items=1, description="Stream of content chunks")
    analysis_types: List[str] = Field(default=["sentiment", "entities"], description="Types of analysis")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for processing")
    real_time: bool = Field(default=True, description="Enable real-time streaming")


class ConnectionManager:
    """Manages WebSocket connections for real-time analysis"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analysis_sessions: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.analysis_sessions[session_id] = {
            "started_at": datetime.now(),
            "status": "connected",
            "analyses_completed": 0,
            "total_analyses": 0
        }
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.analysis_sessions:
            del self.analysis_sessions[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_personal_message(self, message: str, session_id: str):
        """Send message to specific WebSocket connection"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        for session_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_analysis_result(self, result: RealTimeAnalysisResult, session_id: str):
        """Send analysis result to specific session"""
        message = result.model_dump_json()
        await self.send_personal_message(message, session_id)
    
    async def update_session_progress(self, session_id: str, progress: float, status: str):
        """Update session progress"""
        if session_id in self.analysis_sessions:
            self.analysis_sessions[session_id]["progress"] = progress
            self.analysis_sessions[session_id]["status"] = status
            
            # Send progress update
            progress_message = {
                "type": "progress_update",
                "session_id": session_id,
                "progress": progress,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            await self.send_personal_message(json.dumps(progress_message), session_id)


# Global connection manager
manager = ConnectionManager()


class RealTimeAnalysisEngine:
    """Engine for real-time content analysis"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.analysis_queue: asyncio.Queue = asyncio.Queue()
        self.running_analyses: Dict[str, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize the real-time analysis engine"""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.info("Real-time analysis engine initialized with Redis")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    async def start_analysis_session(self, request: RealTimeAnalysisRequest) -> str:
        """Start a new real-time analysis session"""
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store session in Redis
        if self.redis_client:
            session_data = {
                "content": request.content,
                "analysis_types": request.analysis_types,
                "stream_interval": request.stream_interval,
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }
            await self.redis_client.hset(f"session:{session_id}", mapping=session_data)
            await self.redis_client.expire(f"session:{session_id}", 3600)  # 1 hour TTL
        
        # Start analysis task
        analysis_task = asyncio.create_task(
            self._run_analysis_session(session_id, request)
        )
        self.running_analyses[session_id] = analysis_task
        
        logger.info(f"Started analysis session: {session_id}")
        return session_id
    
    async def _run_analysis_session(self, session_id: str, request: RealTimeAnalysisRequest):
        """Run analysis session with streaming results"""
        try:
            total_analyses = len(request.analysis_types)
            completed_analyses = 0
            
            for analysis_type in request.analysis_types:
                # Perform analysis
                result = await self._perform_analysis(request.content, analysis_type)
                
                # Create result object
                analysis_result = RealTimeAnalysisResult(
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    analysis_type=analysis_type,
                    result=result,
                    progress=(completed_analyses + 1) / total_analyses,
                    status="completed"
                )
                
                # Send result via WebSocket
                await manager.send_analysis_result(analysis_result, session_id)
                
                # Update progress
                await manager.update_session_progress(
                    session_id, 
                    analysis_result.progress, 
                    f"Completed {analysis_type} analysis"
                )
                
                completed_analyses += 1
                
                # Wait for stream interval
                if completed_analyses < total_analyses:
                    await asyncio.sleep(request.stream_interval)
            
            # Mark session as completed
            await manager.update_session_progress(session_id, 1.0, "completed")
            
            # Store final results in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    f"session:{session_id}", 
                    "status", 
                    "completed"
                )
                await self.redis_client.hset(
                    f"session:{session_id}", 
                    "completed_at", 
                    datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.error(f"Error in analysis session {session_id}: {e}")
            await manager.update_session_progress(session_id, 0.0, f"error: {str(e)}")
        finally:
            # Clean up
            if session_id in self.running_analyses:
                del self.running_analyses[session_id]
    
    async def _perform_analysis(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Perform specific type of analysis"""
        try:
            if analysis_type == "sentiment":
                return await ai_ml_engine.analyze_sentiment(content)
            elif analysis_type == "language":
                return await ai_ml_engine.detect_language(content)
            elif analysis_type == "entities":
                return await ai_ml_engine.extract_entities(content)
            elif analysis_type == "readability":
                return await ai_ml_engine.analyze_readability(content)
            elif analysis_type == "summary":
                return await ai_ml_engine.generate_summary(content)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            logger.error(f"Error performing {analysis_type} analysis: {e}")
            return {"error": str(e)}
    
    async def stop_analysis_session(self, session_id: str) -> bool:
        """Stop a running analysis session"""
        if session_id in self.running_analyses:
            task = self.running_analyses[session_id]
            task.cancel()
            del self.running_analyses[session_id]
            
            # Update status in Redis
            if self.redis_client:
                await self.redis_client.hset(f"session:{session_id}", "status", "cancelled")
            
            logger.info(f"Stopped analysis session: {session_id}")
            return True
        return False
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of analysis session"""
        if self.redis_client:
            session_data = await self.redis_client.hgetall(f"session:{session_id}")
            if session_data:
                return {k.decode(): v.decode() for k, v in session_data.items()}
        return None
    
    async def stream_analysis(self, request: StreamingAnalysisRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream analysis results for content chunks"""
        try:
            total_chunks = len(request.content_stream)
            processed_chunks = 0
            
            for i in range(0, total_chunks, request.batch_size):
                batch = request.content_stream[i:i + request.batch_size]
                batch_results = []
                
                # Process batch
                for content in batch:
                    chunk_result = {}
                    for analysis_type in request.analysis_types:
                        result = await self._perform_analysis(content, analysis_type)
                        chunk_result[analysis_type] = result
                    
                    batch_results.append({
                        "content": content,
                        "results": chunk_result,
                        "chunk_index": processed_chunks,
                        "timestamp": datetime.now().isoformat()
                    })
                    processed_chunks += 1
                
                # Yield batch results
                yield {
                    "batch_index": i // request.batch_size,
                    "batch_results": batch_results,
                    "progress": processed_chunks / total_chunks,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Real-time delay if enabled
                if request.real_time:
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in stream analysis: {e}")
            yield {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active analysis sessions"""
        active_sessions = []
        
        for session_id, task in self.running_analyses.items():
            session_info = {
                "session_id": session_id,
                "status": "running" if not task.done() else "completed",
                "started_at": datetime.now().isoformat()
            }
            
            # Get additional info from Redis
            if self.redis_client:
                redis_data = await self.redis_client.hgetall(f"session:{session_id}")
                if redis_data:
                    session_info.update({k.decode(): v.decode() for k, v in redis_data.items()})
            
            active_sessions.append(session_info)
        
        return active_sessions
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old analysis sessions"""
        if not self.redis_client:
            return
        
        try:
            # Get all session keys
            session_keys = await self.redis_client.keys("session:*")
            
            for key in session_keys:
                session_data = await self.redis_client.hgetall(key)
                if session_data:
                    started_at_str = session_data.get(b"started_at", b"").decode()
                    if started_at_str:
                        started_at = datetime.fromisoformat(started_at_str)
                        age_hours = (datetime.now() - started_at).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            await self.redis_client.delete(key)
                            logger.info(f"Cleaned up old session: {key.decode()}")
        
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")


# Global real-time analysis engine
realtime_engine = RealTimeAnalysisEngine()


async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    """WebSocket endpoint for real-time analysis"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "start_analysis":
                # Start real-time analysis
                request_data = message.get("data", {})
                request = RealTimeAnalysisRequest(**request_data)
                request.session_id = session_id
                
                analysis_session_id = await realtime_engine.start_analysis_session(request)
                
                # Send confirmation
                response = {
                    "type": "analysis_started",
                    "session_id": analysis_session_id,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(response))
            
            elif message_type == "stop_analysis":
                # Stop analysis
                analysis_session_id = message.get("analysis_session_id")
                if analysis_session_id:
                    stopped = await realtime_engine.stop_analysis_session(analysis_session_id)
                    response = {
                        "type": "analysis_stopped",
                        "session_id": analysis_session_id,
                        "success": stopped,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(response))
            
            elif message_type == "get_status":
                # Get session status
                analysis_session_id = message.get("analysis_session_id")
                if analysis_session_id:
                    status = await realtime_engine.get_session_status(analysis_session_id)
                    response = {
                        "type": "status_response",
                        "session_id": analysis_session_id,
                        "status": status,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(response))
            
            elif message_type == "ping":
                # Heartbeat
                response = {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(response))
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


async def initialize_realtime_engine():
    """Initialize the real-time analysis engine"""
    await realtime_engine.initialize()
    
    # Start cleanup task
    asyncio.create_task(periodic_cleanup())


async def periodic_cleanup():
    """Periodic cleanup of old sessions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await realtime_engine.cleanup_old_sessions()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
















