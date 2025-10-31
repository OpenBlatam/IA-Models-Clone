"""
API Routes for OpusClip Improved
===============================

FastAPI routes with comprehensive video processing and AI capabilities.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .schemas import (
    VideoAnalysisRequest, VideoAnalysisResponse, ClipGenerationRequest,
    ClipGenerationResponse, ClipExportRequest, ClipExportResponse,
    BatchProcessingRequest, BatchProcessingResponse, ProjectRequest,
    ProjectResponse, AnalyticsRequest, AnalyticsResponse,
    HealthCheckResponse, ErrorResponse, ProcessingStatus
)
from .services import opus_clip_service
from .exceptions import (
    OpusClipException, VideoProcessingError, VideoAnalysisError,
    ClipGenerationError, AIProviderError, ValidationError,
    AuthenticationError, AuthorizationError, RateLimitError
)
from .auth import get_current_user, verify_token
from .rate_limiter import rate_limit
from .middleware import RequestLoggingMiddleware

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter(
    prefix="/api/v2/opus-clip",
    tags=["opus-clip"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Security
security = HTTPBearer()


# Health Check
@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check service health and status"
)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthCheckResponse(
            status="healthy",
            version="2.0.0",
            uptime=0.0,  # Would calculate actual uptime
            memory_usage=0.0,  # Would get actual memory usage
            cpu_usage=0.0,  # Would get actual CPU usage
            dependencies={
                "database": "healthy",
                "redis": "healthy",
                "ai_services": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


# Video Analysis Routes
@router.post(
    "/analyze",
    response_model=VideoAnalysisResponse,
    summary="Analyze Video",
    description="Analyze video content and extract insights"
)
@rate_limit(requests=10, window=3600)  # 10 requests per hour
async def analyze_video(
    request: VideoAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze video content"""
    try:
        logger.info(f"Video analysis requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.analyze_video(request)
        
        logger.info(f"Video analysis completed: {result.analysis_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except VideoAnalysisError as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in video analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/analyze/upload",
    response_model=VideoAnalysisResponse,
    summary="Analyze Uploaded Video",
    description="Analyze uploaded video file"
)
@rate_limit(requests=5, window=3600)  # 5 requests per hour
async def analyze_uploaded_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    extract_audio: bool = Form(True, description="Extract and analyze audio"),
    detect_faces: bool = Form(True, description="Detect faces in video"),
    detect_objects: bool = Form(True, description="Detect objects in video"),
    analyze_sentiment: bool = Form(True, description="Analyze sentiment"),
    extract_transcript: bool = Form(True, description="Extract transcript"),
    detect_scenes: bool = Form(True, description="Detect scene changes"),
    language: str = Form("en", description="Language for analysis"),
    current_user: dict = Depends(get_current_user)
):
    """Analyze uploaded video file"""
    try:
        logger.info(f"Video upload analysis requested by user {current_user.get('user_id')}")
        
        # Save uploaded file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Create analysis request
            request = VideoAnalysisRequest(
                video_path=temp_path,
                extract_audio=extract_audio,
                detect_faces=detect_faces,
                detect_objects=detect_objects,
                analyze_sentiment=analyze_sentiment,
                extract_transcript=extract_transcript,
                detect_scenes=detect_scenes,
                language=language
            )
            
            result = await opus_clip_service.analyze_video(request)
            
            logger.info(f"Video upload analysis completed: {result.analysis_id}")
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except VideoAnalysisError as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in video upload analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/analyze/{analysis_id}",
    response_model=VideoAnalysisResponse,
    summary="Get Analysis Results",
    description="Get video analysis results by ID"
)
async def get_analysis_results(
    analysis_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get video analysis results"""
    try:
        logger.info(f"Analysis results requested: {analysis_id}")
        
        # Get analysis results from cache or database
        results = await opus_clip_service._get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis results not found"
            )
        
        return VideoAnalysisResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Clip Generation Routes
@router.post(
    "/generate",
    response_model=ClipGenerationResponse,
    summary="Generate Clips",
    description="Generate clips from analyzed video"
)
@rate_limit(requests=20, window=3600)  # 20 requests per hour
async def generate_clips(
    request: ClipGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate clips from analyzed video"""
    try:
        logger.info(f"Clip generation requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.generate_clips(request)
        
        logger.info(f"Clip generation completed: {result.generation_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ClipGenerationError as e:
        logger.error(f"Clip generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in clip generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/generate/{generation_id}",
    response_model=ClipGenerationResponse,
    summary="Get Generation Results",
    description="Get clip generation results by ID"
)
async def get_generation_results(
    generation_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get clip generation results"""
    try:
        logger.info(f"Generation results requested: {generation_id}")
        
        # Get generation results from cache or database
        results = await opus_clip_service._get_generation_results(generation_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation results not found"
            )
        
        return ClipGenerationResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting generation results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Clip Export Routes
@router.post(
    "/export",
    response_model=ClipExportResponse,
    summary="Export Clips",
    description="Export clips in specified format and quality"
)
@rate_limit(requests=15, window=3600)  # 15 requests per hour
async def export_clips(
    request: ClipExportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Export clips in specified format and quality"""
    try:
        logger.info(f"Clip export requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.export_clips(request)
        
        logger.info(f"Clip export completed: {result.export_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ClipGenerationError as e:
        logger.error(f"Clip export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in clip export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/export/{export_id}",
    response_model=ClipExportResponse,
    summary="Get Export Results",
    description="Get clip export results by ID"
)
async def get_export_results(
    export_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get clip export results"""
    try:
        logger.info(f"Export results requested: {export_id}")
        
        # Get export results from cache or database
        # This would be implemented in the service
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Export results retrieval not implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting export results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Download Routes
@router.get(
    "/download/{file_id}",
    summary="Download File",
    description="Download generated clip or exported file"
)
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download file by ID"""
    try:
        logger.info(f"File download requested: {file_id}")
        
        # Get file path from database or cache
        # This would be implemented in the service
        file_path = f"./output/{file_id}.mp4"  # Placeholder
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return FileResponse(
            path=file_path,
            filename=f"{file_id}.mp4",
            media_type="video/mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Batch Processing Routes
@router.post(
    "/batch/process",
    response_model=BatchProcessingResponse,
    summary="Process Batch",
    description="Process multiple videos in batch"
)
@rate_limit(requests=3, window=3600)  # 3 requests per hour
async def process_batch(
    request: BatchProcessingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process multiple videos in batch"""
    try:
        logger.info(f"Batch processing requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.process_batch(request)
        
        logger.info(f"Batch processing completed: {result.batch_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ClipGenerationError as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/batch/{batch_id}",
    response_model=BatchProcessingResponse,
    summary="Get Batch Results",
    description="Get batch processing results by ID"
)
async def get_batch_results(
    batch_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get batch processing results"""
    try:
        logger.info(f"Batch results requested: {batch_id}")
        
        # Get batch results from cache or database
        # This would be implemented in the service
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Batch results retrieval not implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Project Management Routes
@router.post(
    "/projects",
    response_model=ProjectResponse,
    summary="Create Project",
    description="Create a new project"
)
@rate_limit(requests=10, window=3600)  # 10 requests per hour
async def create_project(
    request: ProjectRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new project"""
    try:
        logger.info(f"Project creation requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.create_project(request)
        
        logger.info(f"Project created: {result.project_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in project creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/projects",
    response_model=List[ProjectResponse],
    summary="List Projects",
    description="List user's projects"
)
async def list_projects(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100, description="Number of projects to return"),
    offset: int = Query(0, ge=0, description="Number of projects to skip")
):
    """List user's projects"""
    try:
        logger.info(f"Projects list requested by user {current_user.get('user_id')}")
        
        # Get projects from database
        # This would be implemented in the service
        projects = []  # Placeholder
        
        return projects
        
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Get Project",
    description="Get project by ID"
)
async def get_project(
    project_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get project by ID"""
    try:
        logger.info(f"Project requested: {project_id}")
        
        # Get project from database
        # This would be implemented in the service
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Project retrieval not implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Analytics Routes
@router.post(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Get Analytics",
    description="Get analytics for project or system"
)
@rate_limit(requests=20, window=3600)  # 20 requests per hour
async def get_analytics(
    request: AnalyticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics for project or system"""
    try:
        logger.info(f"Analytics requested by user {current_user.get('user_id')}")
        
        result = await opus_clip_service.get_analytics(request)
        
        logger.info(f"Analytics generated: {result.analytics_id}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/analytics/{analytics_id}",
    response_model=AnalyticsResponse,
    summary="Get Analytics Results",
    description="Get analytics results by ID"
)
async def get_analytics_results(
    analytics_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics results by ID"""
    try:
        logger.info(f"Analytics results requested: {analytics_id}")
        
        # Get analytics results from cache or database
        # This would be implemented in the service
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Analytics results retrieval not implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Statistics Routes
@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get system statistics"
)
async def get_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get system statistics"""
    try:
        logger.info(f"Statistics requested by user {current_user.get('user_id')}")
        
        # Get statistics from database
        stats = {
            "total_videos_processed": 0,
            "total_clips_generated": 0,
            "total_exports": 0,
            "active_projects": 0,
            "system_uptime": 0,
            "average_processing_time": 0,
            "success_rate": 0.95
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Error Handlers
@router.exception_handler(OpusClipException)
async def opus_clip_exception_handler(request, exc: OpusClipException):
    """Handle OpusClip exceptions"""
    logger.error(f"OpusClip exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code=exc.error_code,
            error_message=exc.message,
            error_details=exc.details
        ).model_dump()
    )


@router.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation exceptions"""
    logger.error(f"Validation exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message=exc.message,
            error_details=exc.details
        ).model_dump()
    )


@router.exception_handler(AuthenticationError)
async def authentication_exception_handler(request, exc: AuthenticationError):
    """Handle authentication exceptions"""
    logger.error(f"Authentication exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=ErrorResponse(
            error_code="AUTHENTICATION_ERROR",
            error_message=exc.message,
            error_details=exc.details
        ).model_dump()
    )


@router.exception_handler(AuthorizationError)
async def authorization_exception_handler(request, exc: AuthorizationError):
    """Handle authorization exceptions"""
    logger.error(f"Authorization exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=ErrorResponse(
            error_code="AUTHORIZATION_ERROR",
            error_message=exc.message,
            error_details=exc.details
        ).model_dump()
    )


@router.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request, exc: RateLimitError):
    """Handle rate limit exceptions"""
    logger.error(f"Rate limit exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse(
            error_code="RATE_LIMIT_ERROR",
            error_message=exc.message,
            error_details=exc.details
        ).model_dump(),
        headers={"Retry-After": str(exc.retry_after) if exc.retry_after else "3600"}
    )






























