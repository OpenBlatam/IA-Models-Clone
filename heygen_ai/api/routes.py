from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from .models import *
from ..core import HeyGenAI
from ..config.settings import get_settings
from typing import Any, List, Dict, Optional
"""
API Routes for HeyGen AI equivalent.
FastAPI endpoints for video generation and management with LangChain integration.
"""

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["heygen-ai"])

# Get settings
settings = get_settings()

# Initialize HeyGen AI system with OpenRouter API key
heygen_ai = HeyGenAI(openrouter_api_key=settings.openrouter_api_key)

# Dependency to get HeyGen AI instance
def get_heygen_ai() -> HeyGenAI:
    return heygen_ai

# =============================================================================
# Core Video Generation Endpoints (Enhanced)
# =============================================================================

@router.post("/videos/create", response_model=VideoResponse)
async def create_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Create a new video using the enhanced HeyGen AI core."""
    try:
        # Generate video using enhanced core
        response = await heygen.create_video(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/videos/batch", response_model=List[VideoResponse])
async def batch_create_videos(
    requests: List[VideoRequest],
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Create multiple videos in batch using the enhanced core."""
    try:
        responses = await heygen.batch_create_videos(requests)
        return responses
        
    except Exception as e:
        logger.error(f"Failed to create batch videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Voice Generation Endpoints (Enhanced)
# =============================================================================

@router.post("/voice/generate")
async def generate_voice(
    request: VoiceGenerationRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Generate speech from text using the enhanced voice engine."""
    try:
        audio_path = await heygen.voice_engine.synthesize_speech(request)
        
        return {
            "status": "success",
            "audio_path": audio_path,
            "voice_id": request.voice_id,
            "text_length": len(request.text),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/available")
async def get_available_voices(heygen: HeyGenAI = Depends(get_heygen_ai)):
    """Get list of available voices from the enhanced voice engine."""
    try:
        voices = await heygen.voice_engine.get_available_voices()
        return {
            "voices": voices,
            "total_count": len(voices),
            "engines": list(heygen.voice_engine.tts_engines.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get available voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Avatar Generation Endpoints (Enhanced)
# =============================================================================

@router.post("/avatar/generate")
async def generate_avatar_video(
    request: AvatarGenerationRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Generate avatar video with lip-sync using the enhanced avatar manager."""
    try:
        video_path = await heygen.avatar_manager.generate_avatar_video(request)
        
        return {
            "status": "success",
            "video_path": video_path,
            "avatar_id": request.avatar_id,
            "resolution": request.resolution,
            "quality_preset": request.quality_preset,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate avatar video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/avatar/available")
async def get_available_avatars(heygen: HeyGenAI = Depends(get_heygen_ai)):
    """Get list of available avatars from the enhanced avatar manager."""
    try:
        avatars = await heygen.avatar_manager.get_available_avatars()
        return {
            "avatars": avatars,
            "total_count": len(avatars),
            "models": list(heygen.avatar_manager.generation_pipelines.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get available avatars: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# System Health and Status Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(heygen: HeyGenAI = Depends(get_heygen_ai)):
    """Check system health including all enhanced components."""
    try:
        # Get health from enhanced core
        health_status = heygen.health_check()
        
        # Get component statuses
        voice_health = heygen.voice_engine.health_check()
        avatar_health = heygen.avatar_manager.health_check()
        
        components = {
            **health_status,
            "voice_engine": voice_health["status"] == "healthy",
            "avatar_manager": avatar_health["status"] == "healthy"
        }
        
        overall_status = "healthy" if all(components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components=components,
            version="2.0.0",
            uptime=0.0,  # Would calculate actual uptime
            metadata={
                "enhanced_core": True,
                "voice_engines": list(heygen.voice_engine.tts_engines.keys()),
                "avatar_models": list(heygen.avatar_manager.generation_pipelines.keys())
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/status")
async def get_system_status(heygen: HeyGenAI = Depends(get_heygen_ai)):
    """Get detailed system status and performance metrics."""
    try:
        # Get performance metrics from enhanced components
        voice_metrics = heygen.voice_engine.get_performance_metrics()
        avatar_metrics = heygen.avatar_manager.get_performance_metrics()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "voice_engine": {
                    "status": "operational",
                    "request_count": voice_metrics["request_count"],
                    "success_rate": voice_metrics["success_rate"],
                    "avg_processing_time": voice_metrics["avg_processing_time"]
                },
                "avatar_manager": {
                    "status": "operational",
                    "request_count": avatar_metrics["request_count"],
                    "success_rate": avatar_metrics["success_rate"],
                    "avg_processing_time": avatar_metrics["avg_processing_time"]
                }
            },
            "performance": {
                "total_requests": voice_metrics["request_count"] + avatar_metrics["request_count"],
                "overall_success_rate": (voice_metrics["success_rate"] + avatar_metrics["success_rate"]) / 2
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Legacy Endpoints (Maintained for Backward Compatibility)
# =============================================================================

@router.post("/videos/create-legacy", response_model=VideoResponse)
async def create_video_legacy(
    request: CreateVideoRequest,
    background_tasks: BackgroundTasks,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Legacy endpoint for creating videos (maintained for compatibility)."""
    try:
        # Convert legacy request to core request
        core_request = VideoRequest(
            script=request.script,
            avatar_id=request.avatar_id,
            voice_id=request.voice_id,
            language=request.language.value,
            output_format=request.output_format.value,
            resolution=request.resolution.value,
            duration=request.duration,
            background=str(request.background) if request.background else None,
            custom_settings=request.custom_settings,
            quality_preset=request.quality_preset,
            enable_expressions=request.enable_expressions,
            enable_effects=request.enable_effects
        )
        
        # Generate video using enhanced core
        response = await heygen.create_video(core_request)
        return response
        
    except Exception as e:
        logger.error(f"Failed to create video (legacy): {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Script Generation Endpoints with LangChain
@router.post("/scripts/generate", response_model=ScriptResponse)
async def generate_script(
    request: GenerateScriptRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Generate a script using LangChain and OpenRouter."""
    try:
        script = await heygen.generate_script(
            topic=request.topic,
            language=request.language.value,
            style=request.style.value,
            duration=request.duration,
            context=request.additional_context or ""
        )
        
        return ScriptResponse(
            script_id=f"script_{datetime.now().timestamp()}",
            script=script,
            word_count=len(script.split()),
            estimated_duration=len(script.split()) / 150.0,  # Rough estimate
            language=request.language,
            style=request.style,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scripts/optimize", response_model=ScriptResponse)
async def optimize_script(
    request: OptimizeScriptRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Optimize a script using LangChain."""
    try:
        optimized_script = await heygen.optimize_script(
            script=request.script,
            duration=request.duration,
            style=request.style.value,
            language=request.language.value
        )
        
        return ScriptResponse(
            script_id=f"optimized_{datetime.now().timestamp()}",
            script=optimized_script,
            word_count=len(optimized_script.split()),
            estimated_duration=len(optimized_script.split()) / 150.0,
            language=request.language,
            style=request.style,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to optimize script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scripts/analyze", response_model=ScriptAnalysisResponse)
async def analyze_script(
    request: AnalyzeScriptRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Analyze a script using LangChain."""
    try:
        analysis = await heygen.analyze_script(request.script)
        
        return ScriptAnalysisResponse(
            script_id=f"analysis_{datetime.now().timestamp()}",
            word_count=analysis["word_count"],
            estimated_duration=analysis["estimated_duration"],
            readability_score=analysis["readability_score"],
            sentiment=analysis["sentiment"],
            complexity=analysis["complexity"],
            suggestions=analysis["suggestions"]
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scripts/translate", response_model=TranslationResponse)
async def translate_script(
    request: TranslateScriptRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Translate a script using LangChain."""
    try:
        translated_script = await heygen.translate_script(
            script=request.script,
            target_language=request.target_language.value,
            source_language=request.source_language.value,
            preserve_style=request.preserve_style
        )
        
        return TranslationResponse(
            translation_id=f"translation_{datetime.now().timestamp()}",
            original_script=request.script,
            translated_script=translated_script,
            source_language=request.source_language,
            target_language=request.target_language,
            word_count=len(translated_script.split()),
            confidence_score=0.95  # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Failed to translate script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# LangChain Agent Endpoints
@router.post("/langchain/chat")
async def chat_with_agent(
    request: ChatRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Chat with LangChain agent for complex workflows."""
    try:
        response = await heygen.chat_with_agent(request.message)
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "agent_used": "langchain_agent"
        }
        
    except Exception as e:
        logger.error(f"Failed to chat with agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge Base Endpoints
@router.post("/knowledge-base/create")
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Create a knowledge base using LangChain."""
    try:
        await heygen.create_knowledge_base(
            documents=request.documents,
            name=request.name
        )
        
        return {
            "status": "success",
            "message": f"Knowledge base '{request.name}' created successfully",
            "document_count": len(request.documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-base/search")
async def search_knowledge_base(
    request: SearchKnowledgeBaseRequest,
    heygen: HeyGenAI = Depends(get_heygen_ai)
):
    """Search knowledge base using LangChain."""
    try:
        results = await heygen.search_knowledge_base(
            query=request.query,
            name=request.name,
            k=request.max_results
        )
        
        return {
            "query": request.query,
            "results": results,
            "result_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to search knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handling
@router.exception_handler(Exception)
async def global_exception_handler(request, exc) -> Any:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_ERROR",
        details={"message": str(exc)},
        timestamp=datetime.now().isoformat()
    ) 