"""
Mobile-Optimized API
====================

Lightweight, mobile-optimized API endpoints for the BUL system.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import io

from ..core.bul_engine import get_global_bul_engine
from ..core.continuous_processor import get_global_continuous_processor
from ..ml.document_optimizer import get_global_document_optimizer
from ..voice.voice_processor import get_global_voice_processor
from ..collaboration.realtime_editor import get_global_realtime_editor

logger = logging.getLogger(__name__)

# Mobile API router
mobile_router = APIRouter(prefix="/mobile", tags=["Mobile API"])

# Pydantic models for mobile API
class MobileQueryRequest(BaseModel):
    """Mobile-optimized query request."""
    query: str = Field(..., max_length=500, description="User query")
    business_area: Optional[str] = Field(None, description="Business area")
    document_type: Optional[str] = Field(None, description="Document type")
    user_id: Optional[str] = Field(None, description="User ID")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")

class MobileDocumentResponse(BaseModel):
    """Mobile-optimized document response."""
    id: str
    title: str
    content: str
    business_area: str
    document_type: str
    created_at: str
    word_count: int
    reading_time: int  # Estimated reading time in minutes
    quality_score: float
    summary: str

class MobileStatusResponse(BaseModel):
    """Mobile-optimized status response."""
    status: str
    active_queries: int
    total_documents: int
    system_health: str
    last_activity: str
    mobile_optimized: bool = True

class MobileVoiceRequest(BaseModel):
    """Mobile voice request."""
    audio_data: str  # Base64 encoded audio
    language: str = "en"
    user_id: Optional[str] = None

class MobileVoiceResponse(BaseModel):
    """Mobile voice response."""
    text: str
    confidence: float
    processing_time: float
    language_detected: Optional[str] = None

class MobileCollaborationRequest(BaseModel):
    """Mobile collaboration request."""
    session_id: str
    user_id: str
    user_name: str
    action: str  # join, leave, update_cursor, add_comment
    data: Optional[Dict[str, Any]] = None

class MobileCollaborationResponse(BaseModel):
    """Mobile collaboration response."""
    success: bool
    message: str
    session_info: Optional[Dict[str, Any]] = None

# Mobile API endpoints
@mobile_router.post("/query", response_model=MobileDocumentResponse)
async def mobile_submit_query(request: MobileQueryRequest):
    """Submit query from mobile device."""
    try:
        bul_engine = get_global_bul_engine()
        
        # Process query
        result = await bul_engine.process_query(
            query=request.query,
            business_area=request.business_area,
            document_type=request.document_type,
            user_id=request.user_id
        )
        
        # Calculate reading time (average 200 words per minute)
        word_count = len(result.content.split())
        reading_time = max(1, word_count // 200)
        
        # Get quality score
        optimizer = get_global_document_optimizer()
        metrics = await optimizer.analyze_document_performance(result.content)
        
        # Create summary (first 100 words)
        summary_words = result.content.split()[:100]
        summary = " ".join(summary_words) + ("..." if len(result.content.split()) > 100 else "")
        
        return MobileDocumentResponse(
            id=result.id,
            title=result.title,
            content=result.content,
            business_area=result.business_area,
            document_type=result.document_type,
            created_at=result.created_at.isoformat(),
            word_count=word_count,
            reading_time=reading_time,
            quality_score=metrics.quality_score,
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Error in mobile query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.get("/status", response_model=MobileStatusResponse)
async def mobile_get_status():
    """Get mobile-optimized system status."""
    try:
        processor = get_global_continuous_processor()
        bul_engine = get_global_bul_engine()
        
        # Get system statistics
        stats = await processor.get_statistics()
        
        return MobileStatusResponse(
            status="active",
            active_queries=stats.get("active_queries", 0),
            total_documents=stats.get("total_documents", 0),
            system_health="healthy",
            last_activity=datetime.now().isoformat(),
            mobile_optimized=True
        )
    
    except Exception as e:
        logger.error(f"Error getting mobile status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.post("/voice/transcribe", response_model=MobileVoiceResponse)
async def mobile_voice_transcribe(request: MobileVoiceRequest):
    """Transcribe voice input from mobile device."""
    try:
        voice_processor = get_global_voice_processor()
        
        # Decode base64 audio data
        audio_data = base64.b64decode(request.audio_data)
        
        # Transcribe
        result = await voice_processor.speech_to_text(audio_data)
        
        return MobileVoiceResponse(
            text=result.text,
            confidence=result.confidence,
            processing_time=result.processing_time,
            language_detected=result.language_detected
        )
    
    except Exception as e:
        logger.error(f"Error in mobile voice transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.post("/voice/synthesize")
async def mobile_voice_synthesize(text: str = Form(...), language: str = Form("en")):
    """Synthesize text to speech for mobile device."""
    try:
        voice_processor = get_global_voice_processor()
        
        # Generate speech
        audio_data = await voice_processor.text_to_speech(text)
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    
    except Exception as e:
        logger.error(f"Error in mobile voice synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.post("/collaboration", response_model=MobileCollaborationResponse)
async def mobile_collaboration(request: MobileCollaborationRequest):
    """Handle mobile collaboration actions."""
    try:
        editor = get_global_realtime_editor()
        
        if request.action == "join":
            success = await editor.join_session(
                session_id=request.session_id,
                user_id=request.user_id,
                user_name=request.user_name,
                user_email=f"{request.user_id}@mobile.user",
                role="editor"
            )
            
            if success:
                session_info = editor.get_session_info(request.session_id)
                return MobileCollaborationResponse(
                    success=True,
                    message="Successfully joined session",
                    session_info=session_info
                )
            else:
                return MobileCollaborationResponse(
                    success=False,
                    message="Failed to join session"
                )
        
        elif request.action == "leave":
            await editor.leave_session(request.session_id, request.user_id)
            return MobileCollaborationResponse(
                success=True,
                message="Successfully left session"
            )
        
        elif request.action == "update_cursor":
            position = request.data.get("position", 0) if request.data else 0
            await editor.update_cursor_position(
                session_id=request.session_id,
                user_id=request.user_id,
                position=position
            )
            return MobileCollaborationResponse(
                success=True,
                message="Cursor position updated"
            )
        
        elif request.action == "add_comment":
            content = request.data.get("content", "") if request.data else ""
            position = request.data.get("position", 0) if request.data else 0
            
            comment_id = await editor.add_comment(
                session_id=request.session_id,
                user_id=request.user_id,
                content=content,
                position=position
            )
            
            if comment_id:
                return MobileCollaborationResponse(
                    success=True,
                    message="Comment added successfully"
                )
            else:
                return MobileCollaborationResponse(
                    success=False,
                    message="Failed to add comment"
                )
        
        else:
            return MobileCollaborationResponse(
                success=False,
                message=f"Unknown action: {request.action}"
            )
    
    except Exception as e:
        logger.error(f"Error in mobile collaboration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.get("/documents/recent")
async def mobile_get_recent_documents(user_id: Optional[str] = None, limit: int = 10):
    """Get recent documents for mobile device."""
    try:
        bul_engine = get_global_bul_engine()
        
        # Get recent documents (simplified for mobile)
        recent_docs = []
        
        # In a real implementation, this would query the database
        # For now, return a simplified response
        return {
            "documents": recent_docs,
            "total": len(recent_docs),
            "mobile_optimized": True
        }
    
    except Exception as e:
        logger.error(f"Error getting recent documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.get("/documents/{document_id}")
async def mobile_get_document(document_id: str):
    """Get specific document for mobile device."""
    try:
        bul_engine = get_global_bul_engine()
        
        # Get document (simplified for mobile)
        # In a real implementation, this would query the database
        return {
            "id": document_id,
            "title": "Sample Document",
            "content": "This is a sample document for mobile viewing.",
            "mobile_optimized": True
        }
    
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.post("/upload")
async def mobile_upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload file from mobile device."""
    try:
        # Read file content
        content = await file.read()
        
        # Process file based on type
        if file.content_type.startswith("audio/"):
            # Handle audio file
            voice_processor = get_global_voice_processor()
            result = await voice_processor.speech_to_text(content)
            
            return {
                "success": True,
                "type": "audio",
                "transcription": result.text,
                "confidence": result.confidence
            }
        
        elif file.content_type.startswith("text/"):
            # Handle text file
            text_content = content.decode("utf-8")
            
            return {
                "success": True,
                "type": "text",
                "content": text_content,
                "word_count": len(text_content.split())
            }
        
        else:
            return {
                "success": False,
                "message": f"Unsupported file type: {file.content_type}"
            }
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.get("/analytics/summary")
async def mobile_get_analytics_summary(user_id: Optional[str] = None):
    """Get analytics summary for mobile device."""
    try:
        optimizer = get_global_document_optimizer()
        stats = optimizer.get_optimization_statistics()
        
        return {
            "documents_analyzed": stats.get("total_documents_analyzed", 0),
            "users_analyzed": stats.get("total_users_analyzed", 0),
            "ml_enabled": stats.get("ml_enabled", False),
            "mobile_optimized": True
        }
    
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mobile_router.get("/health")
async def mobile_health_check():
    """Mobile health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mobile_optimized": True,
        "version": "1.0.0"
    }

# Mobile-specific utility functions
def optimize_for_mobile(data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize data for mobile consumption."""
    # Remove large fields, compress data, etc.
    optimized = {}
    
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 1000:
            # Truncate long strings for mobile
            optimized[key] = value[:1000] + "..."
        elif isinstance(value, dict):
            # Recursively optimize nested dictionaries
            optimized[key] = optimize_for_mobile(value)
        else:
            optimized[key] = value
    
    return optimized

def get_mobile_headers() -> Dict[str, str]:
    """Get mobile-optimized response headers."""
    return {
        "X-Mobile-Optimized": "true",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Cache-Control": "public, max-age=300",  # 5 minutes cache
        "Content-Encoding": "gzip"
    }

