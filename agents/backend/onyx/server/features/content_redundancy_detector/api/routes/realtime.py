"""
Realtime Router - Real-time analysis and streaming endpoints
"""

import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path, Query, WebSocket
from fastapi.responses import JSONResponse

try:
    from real_time_engine import real_time_engine, StreamType
    from realtime_analysis import realtime_engine, RealTimeAnalysisRequest
    from websocket_endpoint import websocket_endpoint
except ImportError:
    logging.warning("realtime modules not available")
    real_time_engine = None
    StreamType = None
    realtime_engine = None
    RealTimeAnalysisRequest = None
    websocket_endpoint = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.websocket("/ws/{session_id}")
async def websocket_realtime(websocket: WebSocket, session_id: str = Path(...)):
    """WebSocket endpoint for real-time analysis"""
    if not websocket_endpoint:
        await websocket.close(code=1003, reason="WebSocket endpoint not available")
        return
    await websocket_endpoint(websocket, session_id)


@router.post("/start", response_model=Dict[str, Any])
async def start_realtime_analysis(request: Dict[str, Any]) -> JSONResponse:
    """Start real-time analysis session"""
    logger.info("Starting real-time analysis session")
    
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime engine not available")
    
    try:
        if RealTimeAnalysisRequest:
            analysis_request = RealTimeAnalysisRequest(**request)
            session_id = await realtime_engine.start_analysis_session(analysis_request)
        else:
            session_id = await realtime_engine.start_analysis_session(request)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "session_id": session_id,
                "status": "started",
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error starting real-time analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stop/{session_id}", response_model=Dict[str, Any])
async def stop_realtime_analysis(session_id: str = Path(...)) -> JSONResponse:
    """Stop real-time analysis session"""
    logger.info(f"Stopping real-time analysis session: {session_id}")
    
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime engine not available")
    
    try:
        stopped = await realtime_engine.stop_analysis_session(session_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "session_id": session_id,
                "stopped": stopped,
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error stopping real-time analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions", response_model=Dict[str, Any])
async def get_realtime_sessions() -> JSONResponse:
    """Get active real-time analysis sessions"""
    logger.info("Getting active real-time sessions")
    
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime engine not available")
    
    try:
        sessions = await realtime_engine.get_active_sessions()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "sessions": sessions,
                "count": len(sessions),
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error getting real-time sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream/create", response_model=Dict[str, Any])
async def create_realtime_stream(stream_data: Dict[str, Any]) -> JSONResponse:
    """Create real-time stream"""
    logger.info("Real-time stream creation requested")
    
    if not real_time_engine or not StreamType:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        stream_id = stream_data.get("stream_id", f"stream_{int(time.time())}")
        name = stream_data.get("name", "Unnamed Stream")
        stream_type = stream_data.get("type", "content_analysis")
        
        type_mapping = {
            "content_analysis": StreamType.CONTENT_ANALYSIS,
            "similarity_detection": StreamType.SIMILARITY_DETECTION,
            "quality_assessment": StreamType.QUALITY_ASSESSMENT,
            "batch_processing": StreamType.BATCH_PROCESSING,
            "system_monitoring": StreamType.SYSTEM_MONITORING,
            "ai_ml_processing": StreamType.AI_ML_PROCESSING
        }
        
        stream_type_enum = type_mapping.get(stream_type, StreamType.CONTENT_ANALYSIS)
        stream = await real_time_engine.create_stream(stream_id, name, stream_type_enum)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "stream_id": stream.id,
                "name": stream.name,
                "type": stream.stream_type.value if hasattr(stream.stream_type, 'value') else str(stream.stream_type),
                "status": stream.status.value if hasattr(stream.status, 'value') else str(stream.status),
                "created_at": stream.created_at
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Create real-time stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stream/{stream_id}", response_model=Dict[str, Any])
async def get_realtime_stream(stream_id: str = Path(...)) -> JSONResponse:
    """Get real-time stream status"""
    logger.info(f"Real-time stream status requested: {stream_id}")
    
    if not real_time_engine:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        stats = await real_time_engine.get_stream_stats(stream_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get real-time stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/streams", response_model=Dict[str, Any])
async def get_all_realtime_streams() -> JSONResponse:
    """Get all real-time streams"""
    logger.info("All real-time streams requested")
    
    if not real_time_engine:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        streams = await real_time_engine.get_all_streams()
        return JSONResponse(content={
            "success": True,
            "data": {
                "streams": streams
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get real-time streams error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream/{stream_id}/subscribe", response_model=Dict[str, Any])
async def subscribe_to_stream(
    stream_id: str = Path(...),
    subscription_data: Dict[str, Any] = None
) -> JSONResponse:
    """Subscribe to real-time stream"""
    logger.info(f"Stream subscription requested: {stream_id}")
    
    if not real_time_engine:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        if not subscription_data:
            subscription_data = {}
        
        subscriber_id = subscription_data.get("subscriber_id", f"sub_{int(time.time())}")
        filters = subscription_data.get("filters", {})
        
        def callback(event):
            logger.info(f"Event received: {event.event_type} - {event.data}")
        
        success = await real_time_engine.subscribe_to_stream(
            subscriber_id, stream_id, callback, filters
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "subscriber_id": subscriber_id,
                "stream_id": stream_id,
                "subscribed": True
            },
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subscribe to stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/events/{stream_id}", response_model=Dict[str, Any])
async def get_stream_events(
    stream_id: str = Path(...),
    limit: int = Query(default=100, ge=1, le=1000)
) -> JSONResponse:
    """Get stream events"""
    logger.info(f"Stream events requested: {stream_id}")
    
    if not real_time_engine:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        events = await real_time_engine.get_stream_events(stream_id, limit)
        return JSONResponse(content={
            "success": True,
            "data": {
                "stream_id": stream_id,
                "events": [
                    {
                        "id": event.id,
                        "event_type": event.event_type,
                        "data": event.data,
                        "timestamp": event.timestamp,
                        "metadata": event.metadata
                    }
                    for event in events
                ]
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get stream events error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/engine/stats", response_model=Dict[str, Any])
async def get_realtime_engine_stats() -> JSONResponse:
    """Get real-time engine statistics"""
    logger.info("Real-time engine stats requested")
    
    if not real_time_engine:
        raise HTTPException(status_code=503, detail="Real-time engine not available")
    
    try:
        stats = real_time_engine.get_engine_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get real-time engine stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






