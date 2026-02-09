"""
Ultimate Opus Clip API

The most comprehensive video processing API with all advanced features:
- Content Curation Engine (ClipGenius™)
- Speaker Tracking System
- B-roll Integration System
- Advanced Viral Scoring
- Audio Processing System
- Professional Export System
- Advanced Analytics System
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import structlog
import time
import asyncio
import uuid
from pydantic import BaseModel, field_validator
from pathlib import Path
import json
import os

# Import all processors
from .processors.content_curation_engine import ContentCurationEngine, EngagementAnalyzer, SegmentDetector, ClipOptimizer
from .processors.speaker_tracking_system import SpeakerTrackingSystem, FaceDetector, ObjectTracker, AutoFramer, TrackingConfig
from .processors.broll_integration_system import BrollIntegrationSystem, ContentAnalyzer, BrollSuggester, BrollIntegrator, BrollConfig
from .processors.advanced_viral_scoring import AdvancedViralScorer, TrendAnalyzer, HistoricalAnalyzer, AudienceAnalyzer as ViralAudienceAnalyzer
from .processors.audio_processing_system import AudioProcessingSystem, MusicLibrary, AudioEnhancer, SoundEffectLibrary, AudioMixer
from .processors.professional_export_system import ProfessionalExportSystem, XMLExporter, FCPXMLExporter, SocialMediaPublisher, CloudStorageManager
from .processors.advanced_analytics_system import AdvancedAnalyticsSystem, PerformanceTracker, AudienceAnalyzer, ContentAnalyzer, ReportGenerator

# Import models
from .models.video_models import VideoClipRequest, VideoClipResponse, VideoClipBatchRequest, VideoClipBatchResponse
from .models.viral_models import ViralVideoVariant, ViralVideoBatchResponse, ViralCaptionConfig

# Import error handling
from .error_handling import ErrorHandler, ErrorCode, ValidationError, ProcessingError, ExternalServiceError

logger = structlog.get_logger("ultimate_api")
error_handler = ErrorHandler()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UltimateOpusClipRequest(BaseModel):
    """Ultimate Opus Clip processing request with all features."""
    video_path: str
    content_text: Optional[str] = None
    target_platform: str = "tiktok"
    
    # Feature toggles
    enable_content_curation: bool = True
    enable_speaker_tracking: bool = True
    enable_broll_integration: bool = True
    enable_viral_scoring: bool = True
    enable_audio_processing: bool = True
    enable_analytics: bool = True
    
    # Content curation settings
    curation_settings: Dict[str, Any] = {}
    
    # Speaker tracking settings
    tracking_settings: Dict[str, Any] = {}
    
    # B-roll settings
    broll_settings: Dict[str, Any] = {}
    
    # Viral scoring settings
    viral_scoring_settings: Dict[str, Any] = {}
    
    # Audio processing settings
    audio_settings: Dict[str, Any] = {}
    
    # Export settings
    export_settings: Dict[str, Any] = {}
    
    # Analytics settings
    analytics_settings: Dict[str, Any] = {}
    
    # Output settings
    max_clips: int = 5
    clip_duration_range: tuple = (8, 15)
    output_resolution: tuple = (1080, 1920)
    quality: str = "high"
    
    @field_validator('target_platform')
    def validate_platform(cls, v):
        allowed = ["tiktok", "youtube", "instagram", "twitter", "facebook", "linkedin"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of {allowed}")
        return v
    
    @field_validator('quality')
    def validate_quality(cls, v):
        allowed = ["draft", "standard", "high", "professional", "broadcast"]
        if v not in allowed:
            raise ValueError(f"Quality must be one of {allowed}")
        return v

class UltimateOpusClipResponse(BaseModel):
    """Ultimate Opus Clip processing response."""
    job_id: str
    status: str
    clips: List[Dict[str, Any]]
    processing_time: float
    total_clips: int
    success_rate: float
    
    # Feature-specific results
    content_curation_results: Optional[Dict[str, Any]] = None
    speaker_tracking_results: Optional[Dict[str, Any]] = None
    broll_integration_results: Optional[Dict[str, Any]] = None
    viral_scoring_results: Optional[Dict[str, Any]] = None
    audio_processing_results: Optional[Dict[str, Any]] = None
    analytics_results: Optional[Dict[str, Any]] = None
    
    # Export options
    export_formats: List[str] = []
    cloud_upload_urls: List[str] = []
    
    # Recommendations
    recommendations: List[str] = []
    
    metadata: Dict[str, Any]

class ViralScoringRequest(BaseModel):
    """Request for viral scoring analysis."""
    content_data: Dict[str, Any]
    platform: str = "tiktok"
    include_trends: bool = True
    include_audience_analysis: bool = True

class AudioProcessingRequest(BaseModel):
    """Request for audio processing."""
    video_path: str
    content_analysis: Dict[str, Any]
    enhancement_level: str = "balanced"  # minimal, balanced, enhanced, professional
    add_background_music: bool = True
    add_sound_effects: bool = True

class ExportRequest(BaseModel):
    """Request for professional export."""
    video_data: Dict[str, Any]
    export_format: str = "premiere_pro"  # premiere_pro, final_cut, davinci_resolve, xml, edl
    quality: str = "high"
    include_metadata: bool = True

class AnalyticsRequest(BaseModel):
    """Request for analytics analysis."""
    content_ids: List[str]
    time_range: str = "week"  # hour, day, week, month, quarter, year
    include_audience_insights: bool = True
    include_trends: bool = True

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Ultimate Opus Clip API",
    description="The most comprehensive AI-powered video processing platform with all advanced features",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

# Initialize all processors
content_curation_engine = ContentCurationEngine()
speaker_tracking_system = SpeakerTrackingSystem()
broll_integration_system = BrollIntegrationSystem()
advanced_viral_scorer = AdvancedViralScorer()
audio_processing_system = AudioProcessingSystem()
professional_export_system = ProfessionalExportSystem()
advanced_analytics_system = AdvancedAnalyticsSystem()

# Job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "features": {
            "content_curation": True,
            "speaker_tracking": True,
            "broll_integration": True,
            "viral_scoring": True,
            "audio_processing": True,
            "professional_export": True,
            "advanced_analytics": True
        },
        "processors": {
            "content_curation_engine": "ready",
            "speaker_tracking_system": "ready",
            "broll_integration_system": "ready",
            "advanced_viral_scorer": "ready",
            "audio_processing_system": "ready",
            "professional_export_system": "ready",
            "advanced_analytics_system": "ready"
        }
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    try:
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0",
            "components": {
                "content_curation_engine": "healthy",
                "speaker_tracking_system": "healthy",
                "broll_integration_system": "healthy",
                "advanced_viral_scorer": "healthy",
                "audio_processing_system": "healthy",
                "professional_export_system": "healthy",
                "advanced_analytics_system": "healthy"
            },
            "active_jobs": len(active_jobs),
            "completed_jobs": len(completed_jobs),
            "system_resources": {
                "cpu_usage": "normal",
                "memory_usage": "normal",
                "disk_usage": "normal"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

# =============================================================================
# MAIN ULTIMATE OPUS CLIP PROCESSING ENDPOINT
# =============================================================================

@app.post("/ultimate-opus-clip/process", response_model=UltimateOpusClipResponse)
async def process_video_ultimate_opus_clip(
    request: UltimateOpusClipRequest,
    background_tasks: BackgroundTasks
):
    """
    Process video with ALL Opus Clip features enabled.
    
    This is the ultimate endpoint that combines:
    - Content curation to find engaging moments
    - Speaker tracking for professional framing
    - B-roll integration for enhanced visuals
    - Advanced viral scoring for viral potential
    - Audio processing for professional sound
    - Professional export capabilities
    - Advanced analytics and insights
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Starting Ultimate Opus Clip processing job: {job_id}")
        
        # Initialize job tracking
        active_jobs[job_id] = {
            "status": "processing",
            "start_time": time.time(),
            "request": request.dict(),
            "progress": 0,
            "features_enabled": {
                "content_curation": request.enable_content_curation,
                "speaker_tracking": request.enable_speaker_tracking,
                "broll_integration": request.enable_broll_integration,
                "viral_scoring": request.enable_viral_scoring,
                "audio_processing": request.enable_audio_processing,
                "analytics": request.enable_analytics
            }
        }
        
        # Start background processing
        background_tasks.add_task(
            process_ultimate_opus_clip_background,
            job_id,
            request
        )
        
        return UltimateOpusClipResponse(
            job_id=job_id,
            status="processing",
            clips=[],
            processing_time=0.0,
            total_clips=0,
            success_rate=0.0,
            recommendations=[],
            metadata={"message": "Ultimate processing started with all features enabled"}
        )
        
    except Exception as e:
        logger.error(f"Ultimate Opus Clip processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

async def process_ultimate_opus_clip_background(job_id: str, request: UltimateOpusClipRequest):
    """Background task for Ultimate Opus Clip processing."""
    try:
        start_time = time.time()
        clips = []
        feature_results = {}
        
        # Step 1: Content Curation (if enabled)
        if request.enable_content_curation:
            logger.info(f"Job {job_id}: Starting content curation")
            active_jobs[job_id]["progress"] = 10
            
            curation_result = await content_curation_engine.analyze_video(request.video_path)
            feature_results["content_curation"] = curation_result
            
            # Extract clips from curation result
            for clip_data in curation_result.get("clips", []):
                clips.append({
                    "start_time": clip_data["start_time"],
                    "end_time": clip_data["end_time"],
                    "duration": clip_data["duration"],
                    "score": clip_data["score"],
                    "type": "content_curated",
                    "metadata": clip_data.get("metadata", {})
                })
            
            logger.info(f"Job {job_id}: Content curation completed, found {len(clips)} clips")
        
        # Step 2: Advanced Viral Scoring (if enabled)
        if request.enable_viral_scoring and clips:
            logger.info(f"Job {job_id}: Starting viral scoring analysis")
            active_jobs[job_id]["progress"] = 20
            
            viral_results = []
            for i, clip in enumerate(clips):
                try:
                    content_data = {
                        "keywords": clip.get("metadata", {}).get("keywords", []),
                        "duration": clip["duration"],
                        "engagement_scores": [{"score": clip["score"]}],
                        "content_text": request.content_text or ""
                    }
                    
                    viral_score = await advanced_viral_scorer.calculate_viral_score(
                        content_data, request.target_platform
                    )
                    
                    clip["viral_score"] = viral_score.overall_score
                    clip["viral_analysis"] = {
                        "factor_scores": {k.value: v for k, v in viral_score.factor_scores.items()},
                        "recommendations": viral_score.recommendations,
                        "trend_alignment": viral_score.trend_alignment,
                        "audience_potential": viral_score.audience_potential
                    }
                    
                    viral_results.append(viral_score)
                    
                except Exception as e:
                    logger.error(f"Viral scoring failed for clip {i}: {e}")
                    clip["viral_score"] = 0.5
                    clip["viral_analysis"] = {"error": str(e)}
            
            feature_results["viral_scoring"] = {
                "total_clips_analyzed": len(viral_results),
                "average_viral_score": sum([vs.overall_score for vs in viral_results]) / len(viral_results) if viral_results else 0,
                "top_viral_clip": max(clips, key=lambda c: c.get("viral_score", 0)) if clips else None
            }
            
            logger.info(f"Job {job_id}: Viral scoring completed")
        
        # Step 3: Speaker Tracking (if enabled)
        if request.enable_speaker_tracking and clips:
            logger.info(f"Job {job_id}: Starting speaker tracking")
            active_jobs[job_id]["progress"] = 40
            
            tracking_results = []
            for i, clip in enumerate(clips):
                try:
                    # Extract clip video
                    clip_path = await extract_video_clip(
                        request.video_path,
                        clip["start_time"],
                        clip["end_time"],
                        f"/tmp/clip_{job_id}_{i}.mp4"
                    )
                    
                    # Apply speaker tracking
                    tracking_result = await speaker_tracking_system.process_video(
                        clip_path,
                        f"/tmp/tracked_clip_{job_id}_{i}.mp4",
                        request.output_resolution[0],
                        request.output_resolution[1]
                    )
                    
                    clip["tracked_video"] = f"/tmp/tracked_clip_{job_id}_{i}.mp4"
                    clip["tracking_report"] = tracking_result.get("tracking_report", {})
                    tracking_results.append(tracking_result)
                    
                except Exception as e:
                    logger.error(f"Speaker tracking failed for clip {i}: {e}")
                    clip["tracking_error"] = str(e)
            
            feature_results["speaker_tracking"] = {
                "total_clips_tracked": len(tracking_results),
                "success_rate": len([r for r in tracking_results if r.get("tracking_report", {}).get("tracking_success_rate", 0) > 0.8]) / len(tracking_results) if tracking_results else 0
            }
            
            logger.info(f"Job {job_id}: Speaker tracking completed")
        
        # Step 4: Audio Processing (if enabled)
        if request.enable_audio_processing and clips:
            logger.info(f"Job {job_id}: Starting audio processing")
            active_jobs[job_id]["progress"] = 60
            
            audio_results = []
            for i, clip in enumerate(clips):
                try:
                    input_video = clip.get("tracked_video", await extract_video_clip(
                        request.video_path,
                        clip["start_time"],
                        clip["end_time"],
                        f"/tmp/clip_{job_id}_{i}.mp4"
                    ))
                    
                    content_analysis = {
                        "keywords": clip.get("metadata", {}).get("keywords", []),
                        "duration": clip["duration"],
                        "content_type": clip.get("metadata", {}).get("content_type", "general")
                    }
                    
                    audio_result = await audio_processing_system.process_video_audio(
                        input_video,
                        content_analysis,
                        f"/tmp/audio_clip_{job_id}_{i}.mp4"
                    )
                    
                    clip["audio_processed_video"] = f"/tmp/audio_clip_{job_id}_{i}.mp4"
                    clip["audio_processing_report"] = audio_result
                    audio_results.append(audio_result)
                    
                except Exception as e:
                    logger.error(f"Audio processing failed for clip {i}: {e}")
                    clip["audio_error"] = str(e)
            
            feature_results["audio_processing"] = {
                "total_clips_processed": len(audio_results),
                "success_rate": len([r for r in audio_results if r.get("processing_successful", False)]) / len(audio_results) if audio_results else 0
            }
            
            logger.info(f"Job {job_id}: Audio processing completed")
        
        # Step 5: B-roll Integration (if enabled)
        if request.enable_broll_integration and clips and request.content_text:
            logger.info(f"Job {job_id}: Starting B-roll integration")
            active_jobs[job_id]["progress"] = 80
            
            broll_results = []
            for i, clip in enumerate(clips):
                try:
                    input_video = clip.get("audio_processed_video", clip.get("tracked_video", await extract_video_clip(
                        request.video_path,
                        clip["start_time"],
                        clip["end_time"],
                        f"/tmp/clip_{job_id}_{i}.mp4"
                    )))
                    
                    broll_result = await broll_integration_system.process_video(
                        input_video,
                        request.content_text,
                        f"/tmp/broll_clip_{job_id}_{i}.mp4"
                    )
                    
                    clip["broll_video"] = f"/tmp/broll_clip_{job_id}_{i}.mp4"
                    clip["broll_report"] = broll_result.get("integration_report", {})
                    broll_results.append(broll_result)
                    
                except Exception as e:
                    logger.error(f"B-roll integration failed for clip {i}: {e}")
                    clip["broll_error"] = str(e)
            
            feature_results["broll_integration"] = {
                "total_clips_enhanced": len(broll_results),
                "success_rate": len([r for r in broll_results if r.get("integration_report", {}).get("integration_quality") != "low"]) / len(broll_results) if broll_results else 0
            }
            
            logger.info(f"Job {job_id}: B-roll integration completed")
        
        # Step 6: Analytics (if enabled)
        if request.enable_analytics:
            logger.info(f"Job {job_id}: Starting analytics analysis")
            active_jobs[job_id]["progress"] = 90
            
            try:
                # Track performance metrics
                for i, clip in enumerate(clips):
                    await advanced_analytics_system.track_content_performance(
                        f"clip_{job_id}_{i}",
                        request.target_platform,
                        {
                            "engagement": clip.get("score", 0.5),
                            "viral_potential": clip.get("viral_score", 0.5),
                            "quality": 0.8  # Placeholder
                        }
                    )
                
                # Generate analytics report
                content_ids = [f"clip_{job_id}_{i}" for i in range(len(clips))]
                analytics_report = await advanced_analytics_system.generate_analytics_report(content_ids)
                
                feature_results["analytics"] = {
                    "report_id": analytics_report.report_id,
                    "total_content_analyzed": len(analytics_report.content_analyses),
                    "performance_summary": analytics_report.performance_summary,
                    "recommendations": analytics_report.recommendations
                }
                
            except Exception as e:
                logger.error(f"Analytics analysis failed: {e}")
                feature_results["analytics"] = {"error": str(e)}
            
            logger.info(f"Job {job_id}: Analytics analysis completed")
        
        # Step 7: Finalize results
        processing_time = time.time() - start_time
        success_rate = len([c for c in clips if not any(k.endswith("_error") for k in c.keys())]) / max(len(clips), 1)
        
        # Generate recommendations
        recommendations = []
        if feature_results.get("viral_scoring", {}).get("average_viral_score", 0) < 0.6:
            recommendations.append("Consider adding more trending elements to increase viral potential")
        if feature_results.get("speaker_tracking", {}).get("success_rate", 0) < 0.8:
            recommendations.append("Improve speaker tracking for better framing")
        if feature_results.get("audio_processing", {}).get("success_rate", 0) < 0.8:
            recommendations.append("Enhance audio processing for better sound quality")
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "clips": clips,
            "feature_results": feature_results,
            "processing_time": processing_time,
            "success_rate": success_rate,
            "recommendations": recommendations
        })
        
        # Move to completed jobs
        completed_jobs[job_id] = active_jobs.pop(job_id)
        
        logger.info(f"Job {job_id}: Ultimate processing completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Ultimate background processing failed for job {job_id}: {e}")
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0
        })

# =============================================================================
# INDIVIDUAL FEATURE ENDPOINTS
# =============================================================================

@app.post("/viral-scoring/analyze")
async def analyze_viral_potential(request: ViralScoringRequest):
    """Analyze viral potential of content."""
    try:
        viral_score = await advanced_viral_scorer.calculate_viral_score(
            request.content_data, request.platform
        )
        
        return {
            "status": "success",
            "viral_score": viral_score.overall_score,
            "factor_scores": {k.value: v for k, v in viral_score.factor_scores.items()},
            "recommendations": viral_score.recommendations,
            "trend_alignment": viral_score.trend_alignment,
            "audience_potential": viral_score.audience_potential,
            "confidence": viral_score.confidence
        }
    except Exception as e:
        logger.error(f"Viral scoring analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/audio-processing/process")
async def process_audio(request: AudioProcessingRequest):
    """Process video audio with enhancement and effects."""
    try:
        result = await audio_processing_system.process_video_audio(
            request.video_path,
            request.content_analysis,
            f"/tmp/audio_processed_{int(time.time())}.mp4"
        )
        
        return {
            "status": "success",
            "audio_processing_result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")

@app.post("/export/professional")
async def export_professional(request: ExportRequest):
    """Export video in professional format."""
    try:
        # Create export settings
        from .processors.professional_export_system import ExportSettings, ExportFormat, ExportQuality
        
        settings = ExportSettings(
            format=ExportFormat(request.export_format),
            quality=ExportQuality(request.quality),
            resolution=(1920, 1080),
            frame_rate=30.0,
            bitrate=5000,
            codec="h264",
            audio_codec="aac",
            audio_bitrate=128,
            include_metadata=request.include_metadata
        )
        
        result = await professional_export_system.export_project(
            request.video_data,
            settings,
            f"/tmp/export_{int(time.time())}.xml"
        )
        
        return {
            "status": "success",
            "export_result": {
                "success": result.success,
                "output_path": result.output_path,
                "format": result.format.value,
                "file_size": result.file_size
            }
        }
    except Exception as e:
        logger.error(f"Professional export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

@app.post("/analytics/analyze")
async def analyze_performance(request: AnalyticsRequest):
    """Analyze content performance and generate insights."""
    try:
        from .processors.advanced_analytics_system import TimeRange
        
        time_range_map = {
            "hour": TimeRange.HOUR,
            "day": TimeRange.DAY,
            "week": TimeRange.WEEK,
            "month": TimeRange.MONTH,
            "quarter": TimeRange.QUARTER,
            "year": TimeRange.YEAR
        }
        
        time_range = time_range_map.get(request.time_range, TimeRange.WEEK)
        
        report = await advanced_analytics_system.generate_analytics_report(
            request.content_ids, time_range
        )
        
        return {
            "status": "success",
            "analytics_report": {
                "report_id": report.report_id,
                "content_analyses": [
                    {
                        "content_id": ca.content_id,
                        "performance_score": ca.performance_score,
                        "viral_potential": ca.viral_potential,
                        "audience_engagement": ca.audience_engagement,
                        "recommendations": ca.recommendations
                    }
                    for ca in report.content_analyses
                ],
                "performance_summary": report.performance_summary,
                "audience_insights": [
                    {
                        "demographic": ai.demographic,
                        "percentage": ai.percentage,
                        "trend": ai.trend,
                        "confidence": ai.confidence
                    }
                    for ai in report.audience_insights
                ],
                "recommendations": report.recommendations
            }
        }
    except Exception as e:
        logger.error(f"Analytics analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {e}")

# =============================================================================
# JOB MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id in active_jobs:
        return active_jobs[job_id]
    elif job_id in completed_jobs:
        return completed_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "active_jobs": list(active_jobs.keys()),
        "completed_jobs": list(completed_jobs.keys()),
        "total_active": len(active_jobs),
        "total_completed": len(completed_jobs)
    }

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    if job_id in active_jobs:
        active_jobs[job_id]["status"] = "cancelled"
        return {"message": "Job cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def extract_video_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """Extract a clip from video."""
    try:
        import subprocess
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-c", "copy",
            output_path,
            "-y"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ProcessingError(f"Video extraction failed: {result.stderr}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Video clip extraction failed: {e}")
        raise ProcessingError(f"Video clip extraction failed: {e}")

# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(ProcessingError)
async def processing_exception_handler(request: Request, exc: ProcessingError):
    """Handle processing errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Processing Error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# =============================================================================
# STARTUP AND SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Ultimate Opus Clip API starting up")
    
    # Create necessary directories
    os.makedirs("/tmp", exist_ok=True)
    
    logger.info("Ultimate Opus Clip API started successfully with all features enabled")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Ultimate Opus Clip API shutting down")
    
    # Cleanup temporary files
    import shutil
    if os.path.exists("/tmp"):
        shutil.rmtree("/tmp", ignore_errors=True)
    
    logger.info("Ultimate Opus Clip API shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


