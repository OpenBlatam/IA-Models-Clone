from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from typing import Dict, Any
import time
import structlog
from ..models import KeyMessageRequest, KeyMessageResponse
from ..service import OptimizedKeyMessageService
from ..utils import is_valid_message, calculate_processing_time, monitor_performance
    from ..api import service
    from ..models import MessageType
    from ..models import MessageTone
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Message routes for Key Messages feature with concise conditionals.
"""


logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency
async def get_service() -> OptimizedKeyMessageService:
    """Get service instance."""
    if not service:
        raise HTTPException(status_code=503, detail="Service not available")
    return service

@router.post("/generate", response_model=KeyMessageResponse)
@monitor_performance
async def generate_message(
    request: Request,
    key_message_request: KeyMessageRequest,
    background_tasks: BackgroundTasks,
    svc: OptimizedKeyMessageService = Depends(get_service)
) -> KeyMessageResponse:
    """Generate a key message response."""
    start_time = time.perf_counter()
    
    # Guard clauses with concise conditionals
    if not is_valid_message(key_message_request.message):
        raise HTTPException(status_code=400, detail="Invalid message")
    
    if not svc.is_healthy():
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    try:
        # Add background task
        background_tasks.add_task(log_analytics, "generate", key_message_request)
        
        # Generate response
        response = await svc.generate_response(key_message_request)
        processing_time = calculate_processing_time(start_time)
        
        return response
        
    except Exception as e:
        logger.error("Error generating message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=KeyMessageResponse)
@monitor_performance
async def analyze_message(
    request: Request,
    key_message_request: KeyMessageRequest,
    background_tasks: BackgroundTasks,
    svc: OptimizedKeyMessageService = Depends(get_service)
) -> KeyMessageResponse:
    """Analyze a key message."""
    start_time = time.perf_counter()
    
    # Guard clauses with concise conditionals
    if not is_valid_message(key_message_request.message):
        raise HTTPException(status_code=400, detail="Invalid message")
    
    if not svc.is_healthy():
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    try:
        # Add background task
        background_tasks.add_task(log_analytics, "analyze", key_message_request)
        
        # Analyze message
        response = await svc.analyze_message(key_message_request)
        processing_time = calculate_processing_time(start_time)
        
        return response
        
    except Exception as e:
        logger.error("Error analyzing message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_message_types() -> Dict[str, Any]:
    """Get available message types."""
    
    return {
        "success": True,
        "data": [msg_type.value for msg_type in MessageType],
        "processing_time": 0.0
    }

@router.get("/tones")
async def get_tones() -> Dict[str, Any]:
    """Get available tones."""
    
    return {
        "success": True,
        "data": [tone.value for tone in MessageTone],
        "processing_time": 0.0
    }

# Helper function
async def log_analytics(operation: str, request: KeyMessageRequest) -> None:
    """Log analytics data."""
    if not operation or not request:
        return
    
    logger.info("Analytics logged", 
               operation=operation,
               message_length=len(request.message),
               message_type=request.message_type.value,
               tone=request.tone.value) 