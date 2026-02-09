"""
API routes for Social Video Transcriber AI
Enhanced with authentication, batch processing, webhooks, and advanced analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header, Query, Depends
from fastapi.security import APIKeyHeader
from typing import Optional, List, Dict, Any
import logging
from uuid import UUID
from datetime import datetime
import time

from ..core.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionStatus,
    VariantRequest,
    VariantResponse,
    AnalysisRequest,
    AnalysisResponse,
    QuickVariantRequest,
    SupportedPlatform,
    TextVariant,
)
from ..services.video_downloader import get_video_downloader
from ..services.transcription_service import get_transcription_service
from ..services.ai_analyzer import get_ai_analyzer
from ..services.variant_generator import get_variant_generator
from ..services.cache_service import get_cache_service
from ..services.advanced_analyzer import get_advanced_analyzer
from ..services.batch_processor import get_batch_processor, BatchStatus
from ..services.webhook_service import get_webhook_service, WebhookEvent
from ..services.auth_service import get_auth_service, APIKey, UserTier, TIER_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter()

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory storage for jobs (use database in production)
transcription_jobs: Dict[UUID, TranscriptionResponse] = {}


async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[APIKey]:
    """Dependency to validate API key"""
    if not api_key:
        return None
    
    auth_service = get_auth_service()
    return auth_service.validate_api_key(api_key)


async def require_api_key(
    api_key: Optional[APIKey] = Depends(get_api_key),
) -> APIKey:
    """Dependency that requires a valid API key"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Valid API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


async def check_rate_limit(api_key: APIKey = Depends(require_api_key)) -> APIKey:
    """Dependency to check rate limits"""
    auth_service = get_auth_service()
    allowed, rate_info = auth_service.check_rate_limit(api_key)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "exceeded": rate_info.get("exceeded"),
                "limits": rate_info,
            },
            headers={
                "Retry-After": str(60),  # Suggest retry after 60s
                "X-RateLimit-Limit": str(rate_info["minute"]["limit"]),
                "X-RateLimit-Remaining": "0",
            },
        )
    
    return api_key


async def process_transcription_job(job_id: UUID, request: TranscriptionRequest):
    """Background task to process transcription"""
    job = transcription_jobs.get(job_id)
    if not job:
        return
    
    start_time = time.time()
    webhook_service = get_webhook_service()
    
    try:
        downloader = get_video_downloader()
        transcriber = get_transcription_service()
        analyzer = get_ai_analyzer()
        cache = get_cache_service()
        
        # Trigger job started webhook
        await webhook_service.trigger(
            WebhookEvent.JOB_STARTED,
            {"job_id": str(job_id), "url": request.video_url},
            job_id=job_id,
        )
        
        # Check cache first
        cache_key_options = {
            "timestamps": request.include_timestamps,
            "analysis": request.include_analysis,
            "language": request.language,
        }
        cached = await cache.get(request.video_url, cache_key_options)
        
        if cached:
            logger.info(f"Using cached transcription for {job_id}")
            # Update job from cache
            for key, value in cached.items():
                if hasattr(job, key) and key != 'job_id':
                    setattr(job, key, value)
            job.status = TranscriptionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.processing_time = time.time() - start_time
            
            await webhook_service.trigger(
                WebhookEvent.JOB_COMPLETED,
                {"job_id": str(job_id), "cached": True},
                job_id=job_id,
            )
            return
        
        # Step 1: Detect platform
        job.status = TranscriptionStatus.DOWNLOADING
        platform = (
            request.platform 
            if request.platform != SupportedPlatform.AUTO 
            else downloader.detect_platform(request.video_url)
        )
        job.platform_detected = platform
        
        # Step 2: Download video and extract audio
        audio_path, video_info = await downloader.download_video(
            url=request.video_url,
            job_id=job_id,
            extract_audio=True,
        )
        
        job.video_title = video_info.get('title')
        job.video_duration = video_info.get('duration')
        job.video_author = video_info.get('author')
        
        # Progress webhook
        await webhook_service.trigger(
            WebhookEvent.JOB_PROGRESS,
            {"job_id": str(job_id), "status": "transcribing", "progress": 50},
            job_id=job_id,
        )
        
        # Step 3: Transcribe audio
        job.status = TranscriptionStatus.TRANSCRIBING
        transcription_result = await transcriber.transcribe(
            audio_path=audio_path,
            language=request.language,
            include_timestamps=request.include_timestamps,
        )
        
        job.full_text = transcription_result['full_text']
        job.full_text_with_timestamps = transcription_result['full_text_with_timestamps']
        job.segments = transcription_result['segments']
        
        # Step 4: AI Analysis (optional)
        if request.include_analysis and job.full_text:
            job.status = TranscriptionStatus.ANALYZING
            analysis = await analyzer.analyze_content(job.full_text)
            job.analysis = analysis
            
            await webhook_service.trigger(
                WebhookEvent.ANALYSIS_COMPLETED,
                {"job_id": str(job_id), "framework": analysis.framework.value},
                job_id=job_id,
            )
        
        # Complete
        job.status = TranscriptionStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.processing_time = time.time() - start_time
        
        # Cache result
        await cache.set(
            request.video_url,
            {
                "full_text": job.full_text,
                "full_text_with_timestamps": job.full_text_with_timestamps,
                "segments": [s.to_dict() for s in job.segments] if job.segments else [],
                "analysis": job.analysis.to_dict() if job.analysis else None,
                "video_title": job.video_title,
                "video_duration": job.video_duration,
                "video_author": job.video_author,
                "platform_detected": job.platform_detected.value if job.platform_detected else None,
            },
            cache_key_options,
        )
        
        # Cleanup downloaded files
        await downloader.cleanup(job_id)
        
        logger.info(f"Transcription completed: {job_id} ({job.processing_time:.2f}s)")
        
        # Trigger completion webhook
        await webhook_service.trigger(
            WebhookEvent.JOB_COMPLETED,
            {
                "job_id": str(job_id),
                "video_title": job.video_title,
                "duration": job.video_duration,
                "processing_time": job.processing_time,
            },
            job_id=job_id,
        )
        
    except Exception as e:
        logger.error(f"Transcription failed for {job_id}: {e}", exc_info=True)
        job.status = TranscriptionStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        job.processing_time = time.time() - start_time
        
        # Trigger failure webhook
        await webhook_service.trigger(
            WebhookEvent.JOB_FAILED,
            {"job_id": str(job_id), "error": str(e)},
            job_id=job_id,
        )


# ============== Transcription Endpoints ==============

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[APIKey] = Depends(get_api_key),
):
    """
    Transcribe a video from TikTok, Instagram, or YouTube
    
    - **video_url**: URL of the video to transcribe
    - **platform**: Platform (auto-detect if not specified)
    - **include_timestamps**: Include timestamps in transcription
    - **include_analysis**: Include AI framework analysis
    - **language**: Language code (auto-detect if not specified)
    """
    try:
        # Check rate limit if API key provided
        if api_key:
            auth_service = get_auth_service()
            allowed, rate_info = auth_service.check_rate_limit(api_key)
            if not allowed:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Register webhook if provided
        webhook_service = get_webhook_service()
        
        # Create job
        response = TranscriptionResponse(
            status=TranscriptionStatus.PENDING,
        )
        
        # Register job webhook
        if request.webhook_url:
            webhook_service.register_job_webhook(response.job_id, request.webhook_url)
        
        # Store job
        transcription_jobs[response.job_id] = response
        
        # Process in background
        background_tasks.add_task(
            process_transcription_job,
            response.job_id,
            request,
        )
        
        logger.info(f"Started transcription job: {response.job_id}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcribe/{job_id}", response_model=TranscriptionResponse)
async def get_transcription_status(job_id: UUID):
    """Get status of a transcription job"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return transcription_jobs[job_id]


@router.get("/transcribe/{job_id}/text")
async def get_transcription_text(
    job_id: UUID,
    with_timestamps: bool = Query(True, description="Include timestamps"),
    format: str = Query("text", description="Output format: text, srt, vtt"),
):
    """Get transcription text in various formats"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = transcription_jobs[job_id]
    
    if job.status != TranscriptionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Transcription not ready. Status: {job.status.value}"
        )
    
    transcriber = get_transcription_service()
    
    if format == "srt":
        return {"format": "srt", "content": transcriber.segments_to_srt(job.segments)}
    elif format == "vtt":
        return {"format": "vtt", "content": transcriber.segments_to_vtt(job.segments)}
    else:
        return {
            "format": "text",
            "content": job.full_text_with_timestamps if with_timestamps else job.full_text,
        }


# ============== Advanced Analysis Endpoints ==============

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text to determine framework and structure"""
    try:
        analyzer = get_ai_analyzer()
        
        analysis = await analyzer.analyze_content(
            text=request.text,
            include_structure=request.analyze_structure,
        )
        
        improvements = None
        if request.suggest_improvements:
            improvements = await analyzer.suggest_improvements(request.text, analysis)
        
        return AnalysisResponse(analysis=analysis, improvements=improvements)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/full")
async def full_analysis(text: str = Query(..., min_length=20)):
    """
    Perform comprehensive analysis including:
    - Keywords extraction
    - Summary generation
    - Sentiment analysis
    - Framework detection
    """
    try:
        advanced_analyzer = get_advanced_analyzer()
        result = await advanced_analyzer.full_analysis(text)
        
        # Also get framework analysis
        ai_analyzer = get_ai_analyzer()
        framework = await ai_analyzer.analyze_content(text)
        
        return {
            **result,
            "framework_analysis": framework.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Full analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/keywords")
async def extract_keywords(
    text: str = Query(..., min_length=20),
    max_keywords: int = Query(15, ge=5, le=50),
):
    """Extract keywords from text"""
    try:
        analyzer = get_advanced_analyzer()
        keywords = await analyzer.extract_keywords(text, max_keywords)
        
        return {
            "keywords": [
                {
                    "keyword": kw.keyword,
                    "relevance_score": kw.relevance_score,
                    "category": kw.category,
                }
                for kw in keywords
            ],
            "count": len(keywords),
        }
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/summary")
async def generate_summary(text: str = Query(..., min_length=50)):
    """Generate a comprehensive summary"""
    try:
        analyzer = get_advanced_analyzer()
        summary = await analyzer.generate_summary(text)
        
        if not summary:
            raise HTTPException(status_code=400, detail="Could not generate summary")
        
        return {
            "brief_summary": summary.brief_summary,
            "detailed_summary": summary.detailed_summary,
            "bullet_points": summary.bullet_points,
            "main_topic": summary.main_topic,
            "subtopics": summary.subtopics,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/sentiment")
async def analyze_sentiment(text: str = Query(..., min_length=20)):
    """Analyze sentiment and emotions"""
    try:
        analyzer = get_advanced_analyzer()
        result = await analyzer.analyze_sentiment(text)
        
        if not result:
            raise HTTPException(status_code=400, detail="Could not analyze sentiment")
        
        return {
            "overall_sentiment": result.overall_sentiment.value,
            "sentiment_score": result.sentiment_score,
            "emotions_detected": result.emotions_detected,
            "tone_descriptors": result.tone_descriptors,
            "confidence": result.confidence,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/speakers")
async def detect_speakers(job_id: UUID):
    """Detect multiple speakers in a transcription"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = transcription_jobs[job_id]
    
    if job.status != TranscriptionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Transcription not ready")
    
    if not job.full_text_with_timestamps:
        raise HTTPException(status_code=400, detail="No timestamped transcription available")
    
    try:
        analyzer = get_advanced_analyzer()
        result = await analyzer.detect_speakers(job.full_text_with_timestamps)
        return result
        
    except Exception as e:
        logger.error(f"Speaker detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============== Variant Endpoints ==============

@router.post("/variants", response_model=VariantResponse)
async def generate_variants(request: VariantRequest):
    """Generate text variants maintaining context, structure, and length"""
    start_time = time.time()
    
    try:
        generator = get_variant_generator()
        analyzer = get_ai_analyzer()
        webhook_service = get_webhook_service()
        
        text = request.text
        analysis = None
        
        if not text and request.job_id:
            if request.job_id not in transcription_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = transcription_jobs[request.job_id]
            
            if job.status != TranscriptionStatus.COMPLETED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Transcription not ready. Status: {job.status.value}"
                )
            
            text = job.full_text
            analysis = job.analysis
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        if not analysis:
            analysis = await analyzer.analyze_content(text)
        
        variants = await generator.generate_variants(
            text=text,
            num_variants=request.num_variants,
            preserve_structure=request.preserve_structure,
            preserve_length=request.preserve_length,
            target_tone=request.target_tone,
            custom_instructions=request.custom_instructions,
            analysis=analysis,
        )
        
        processing_time = time.time() - start_time
        
        # Trigger webhook
        await webhook_service.trigger(
            WebhookEvent.VARIANTS_GENERATED,
            {"num_variants": len(variants), "processing_time": processing_time},
        )
        
        return VariantResponse(
            original_text=text,
            variants=variants,
            analysis=analysis,
            processing_time=processing_time,
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Variant generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/variants/quick", response_model=VariantResponse)
async def generate_quick_variants(request: QuickVariantRequest):
    """Quick one-click variant generation from a previous transcription"""
    start_time = time.time()
    
    try:
        if request.job_id not in transcription_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = transcription_jobs[request.job_id]
        
        if job.status != TranscriptionStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Transcription not ready. Status: {job.status.value}"
            )
        
        generator = get_variant_generator()
        variants = await generator.generate_quick_variant(job.full_text)
        
        return VariantResponse(
            original_text=job.full_text,
            variants=variants,
            analysis=job.analysis,
            processing_time=time.time() - start_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick variant generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/variants/text", response_model=VariantResponse)
async def generate_variants_from_text(
    text: str,
    num_variants: int = Query(3, ge=1, le=10),
    preserve_structure: bool = Query(True),
    preserve_length: bool = Query(True),
    target_tone: Optional[str] = Query(None),
):
    """Generate variants from raw text (no previous transcription needed)"""
    start_time = time.time()
    
    try:
        generator = get_variant_generator()
        analyzer = get_ai_analyzer()
        
        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short")
        
        analysis = await analyzer.analyze_content(text)
        
        variants = await generator.generate_variants(
            text=text,
            num_variants=num_variants,
            preserve_structure=preserve_structure,
            preserve_length=preserve_length,
            target_tone=target_tone,
            analysis=analysis,
        )
        
        return VariantResponse(
            original_text=text,
            variants=variants,
            analysis=analysis,
            processing_time=time.time() - start_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text variant generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============== Batch Processing Endpoints ==============

@router.post("/batch")
async def create_batch_job(
    urls: List[str],
    background_tasks: BackgroundTasks,
    include_timestamps: bool = Query(True),
    include_analysis: bool = Query(True),
    language: Optional[str] = Query(None),
    webhook_url: Optional[str] = Query(None),
    api_key: Optional[APIKey] = Depends(get_api_key),
):
    """
    Create a batch transcription job for multiple URLs
    
    - Process multiple videos concurrently
    - Get notified via webhook when complete
    """
    try:
        # Check batch size limit
        max_batch_size = 10
        if api_key:
            limits = TIER_LIMITS.get(api_key.tier)
            if limits:
                max_batch_size = limits.max_batch_size
        
        if len(urls) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_batch_size} URLs per batch for your tier"
            )
        
        batch_processor = get_batch_processor()
        
        batch = await batch_processor.create_batch(
            urls=urls,
            include_timestamps=include_timestamps,
            include_analysis=include_analysis,
            language=language,
            webhook_url=webhook_url,
        )
        
        # Process in background
        background_tasks.add_task(
            batch_processor.process_batch,
            batch.batch_id,
        )
        
        return batch.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/{batch_id}")
async def get_batch_status(batch_id: UUID):
    """Get status of a batch job"""
    batch_processor = get_batch_processor()
    batch = batch_processor.get_batch(batch_id)
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return batch.to_dict()


@router.get("/batch/{batch_id}/results")
async def get_batch_results(batch_id: UUID):
    """Get results of a completed batch job"""
    batch_processor = get_batch_processor()
    batch = batch_processor.get_batch(batch_id)
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.status == BatchStatus.PENDING or batch.status == BatchStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail=f"Batch not ready. Status: {batch.status.value}"
        )
    
    return {
        "batch_id": str(batch.batch_id),
        "status": batch.status.value,
        "results": {
            url: {
                "job_id": str(result.job_id),
                "status": result.status.value,
                "video_title": result.video_title,
                "full_text": result.full_text,
            }
            for url, result in batch.results.items()
        },
        "errors": batch.errors,
    }


# ============== Utility Endpoints ==============

@router.get("/video/info")
async def get_video_info(url: str = Query(..., description="Video URL")):
    """Get video information without downloading"""
    try:
        downloader = get_video_downloader()
        info = await downloader.get_video_info(url)
        return info
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get video info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platforms")
async def list_supported_platforms():
    """List supported video platforms"""
    return {
        "platforms": [
            {"name": "YouTube", "id": "youtube", "supported_urls": ["youtube.com", "youtu.be"]},
            {"name": "TikTok", "id": "tiktok", "supported_urls": ["tiktok.com", "vm.tiktok.com"]},
            {"name": "Instagram", "id": "instagram", "supported_urls": ["instagram.com/reel", "instagram.com/p"]},
        ]
    }


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
):
    """List all transcription jobs"""
    jobs = list(transcription_jobs.values())
    
    if status:
        try:
            status_enum = TranscriptionStatus(status)
            jobs = [j for j in jobs if j.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return {
        "jobs": [
            {
                "job_id": str(j.job_id),
                "status": j.status.value,
                "video_title": j.video_title,
                "platform": j.platform_detected.value if j.platform_detected else None,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs[:limit]
        ],
        "total": len(jobs),
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: UUID):
    """Delete a transcription job"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del transcription_jobs[job_id]
    return {"message": "Job deleted successfully"}


# ============== Cache Endpoints ==============

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    cache = get_cache_service()
    return cache.get_stats()


@router.post("/cache/clear")
async def clear_expired_cache():
    """Clear expired cache entries"""
    cache = get_cache_service()
    cleared = await cache.clear_expired()
    return {"cleared_entries": cleared}


# ============== Auth Endpoints ==============

@router.get("/auth/tiers")
async def list_tiers():
    """List available subscription tiers"""
    return {
        "tiers": [
            {
                "name": tier.value,
                "limits": {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "requests_per_day": limits.requests_per_day,
                    "max_video_duration_seconds": limits.max_video_duration,
                    "max_batch_size": limits.max_batch_size,
                },
            }
            for tier, limits in TIER_LIMITS.items()
        ]
    }


@router.get("/auth/usage")
async def get_usage(api_key: APIKey = Depends(require_api_key)):
    """Get current API key usage statistics"""
    auth_service = get_auth_service()
    stats = auth_service.get_usage_stats(api_key.key_id)
    return stats


# ============== Health Endpoints ==============

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Social Video Transcriber AI",
        "version": "2.0.0",
        "active_jobs": len(transcription_jobs),
        "features": [
            "transcription",
            "timestamps",
            "framework_analysis",
            "variants",
            "batch_processing",
            "caching",
            "keywords",
            "summary",
            "sentiment",
            "speaker_detection",
        ],
    }


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with service status"""
    from ..services.retry_handler import get_circuit_breaker
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "transcription": "operational",
            "ai_analysis": "operational",
            "cache": get_cache_service().get_stats(),
        },
        "circuit_breakers": {
            name: cb.get_status()
            for name, cb in [
                ("openrouter", get_circuit_breaker("openrouter")),
                ("video_download", get_circuit_breaker("video_download")),
            ]
        },
    }
