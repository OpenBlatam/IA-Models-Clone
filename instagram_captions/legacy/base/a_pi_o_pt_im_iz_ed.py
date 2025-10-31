from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import List, Dict, Any, Optional
import logging
from fastapi import APIRouter, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from .schemas import (
from .dependencies import (
from .utils import (
from .speed_optimizations import (
from .core import InstagramCaptionsEngine
from .gmt_system import SimplifiedGMTSystem
from typing import Any, List, Dict, Optional
"""
Optimized Instagram Captions API v2.0

Modern FastAPI implementation with RORO pattern, dependency injection,
caching, error handling, and performance optimization.
"""



    CaptionGenerationRequest,
    CaptionGenerationResponse,
    QualityAnalysisRequest,
    QualityMetricsResponse,
    CaptionOptimizationRequest,
    CaptionOptimizationResponse,
    BatchOptimizationRequest,
    BatchOptimizationResponse,
    HashtagGenerationRequest,
    HashtagGenerationResponse,
    HealthCheckResponse,
    QualityGuidelinesResponse,
    TimezoneInsightsResponse
)
    get_captions_engine,
    get_gmt_system,
    get_cache_manager,
    check_rate_limit,
    get_health_checker,
    validate_request_size
)
    handle_api_errors,
    measure_execution_time,
    log_performance_metrics,
    validate_non_empty_string,
    calculate_improvement_percentage,
    serialize_for_cache,
    deserialize_from_cache,
    generate_cache_key
)
    ultra_fast_cache,
    parallel_process,
    batch_optimize_ultra_fast,
    get_performance_stats,
    format_response_fast
)

logger = logging.getLogger(__name__)

# Initialize optimized router
router = APIRouter(
    prefix="/api/v2/instagram-captions",
    tags=["Instagram Captions v2"],
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        429: {"description": "Rate Limit Exceeded"},
        500: {"description": "Internal Server Error"}
    }
)


@router.get("/", response_model=Dict[str, Any])
@handle_api_errors
async async def get_api_info() -> Dict[str, Any]:
    """Get API information and capabilities."""
    return {
        "name": "Instagram Captions API",
        "version": "2.0.0",
        "description": "Advanced Instagram caption generation with quality optimization",
        "features": [
            "High-quality caption generation",
            "Content quality analysis",
            "Caption optimization",
            "Batch processing",
            "Hashtag generation", 
            "Timezone adaptation",
            "Performance monitoring",
            "Caching support"
        ],
        "endpoints": {
            "generation": "/generate",
            "analysis": "/analyze-quality",
            "optimization": "/optimize",
            "batch": "/batch-optimize",
            "hashtags": "/hashtags/generate",
            "health": "/health",
            "guidelines": "/quality-guidelines"
        }
    }


@router.post("/generate", response_model=CaptionGenerationResponse)
@handle_api_errors
@ultra_fast_cache(ttl=1800)
@log_performance_metrics("caption_generation")
async def generate_caption(
    request: CaptionGenerationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    gmt_system: SimplifiedGMTSystem = Depends(get_gmt_system),
    cache_manager = Depends(get_cache_manager),
    rate_limit: Dict[str, Any] = Depends(check_rate_limit),
    _: None = Depends(validate_request_size)
) -> CaptionGenerationResponse:
    """Generate high-quality Instagram captions with timezone optimization."""
    
    # Generate cache key
    cache_key = generate_cache_key(
        "caption_generation",
        content=request.content_description,
        style=request.style.value,
        audience=request.audience.value,
        timezone=request.timezone
    )
    
    # Try to get from cache
    cached_response = await cache_manager.get(cache_key)
    if cached_response:
        logger.debug("Returning cached caption generation result")
        return deserialize_from_cache(cached_response, CaptionGenerationResponse)
    
    # Validate required fields with early returns
    validate_non_empty_string(request.content_description, "content_description")
    
    # Generate captions
    captions_result = await engine.generate_captions_async(
        content_description=request.content_description,
        style=request.style,
        audience=request.audience,
        include_hashtags=request.include_hashtags,
        hashtag_strategy=request.hashtag_strategy,
        hashtag_count=request.hashtag_count,
        brand_context=request.brand_context
    )
    
    # Get timezone insights if specified
    timezone_insights = None
    if request.timezone and request.timezone != "UTC":
        timezone_insights = gmt_system.get_timezone_insights(request.timezone)
    
    # Build response
    response = CaptionGenerationResponse(
        status="success",
        variations=captions_result.variations,
        best_variation_id=captions_result.best_variation_id,
        timezone_insights=timezone_insights,
        engagement_recommendations=captions_result.engagement_recommendations,
        generation_metadata=captions_result.metadata
    )
    
    # Cache the response
    await cache_manager.set(
        cache_key,
        serialize_for_cache(response),
        ttl=1800  # 30 minutes
    )
    
    return response


@router.post("/analyze-quality", response_model=QualityMetricsResponse)
@handle_api_errors
@ultra_fast_cache(ttl=3600)
@measure_execution_time
async def analyze_caption_quality(
    request: QualityAnalysisRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    cache_manager = Depends(get_cache_manager),
    rate_limit: Dict[str, Any] = Depends(check_rate_limit)
) -> QualityMetricsResponse:
    """Analyze the quality of an Instagram caption."""
    
    # Validate input
    validate_non_empty_string(request.caption, "caption")
    
    # Generate cache key
    cache_key = generate_cache_key(
        "quality_analysis",
        caption=request.caption,
        style=request.style.value,
        audience=request.audience.value
    )
    
    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        return deserialize_from_cache(cached_result, QualityMetricsResponse)
    
    # Analyze quality
    quality_metrics = engine.analyze_quality(
        caption=request.caption,
        style=request.style,
        audience=request.audience
    )
    
    response = QualityMetricsResponse(**quality_metrics.model_dump())
    
    # Cache result
    await cache_manager.set(
        cache_key,
        serialize_for_cache(response),
        ttl=3600  # 1 hour
    )
    
    return response


@router.post("/optimize", response_model=CaptionOptimizationResponse)
@handle_api_errors
@ultra_fast_cache(ttl=1800)
@log_performance_metrics("caption_optimization")
async def optimize_caption(
    request: CaptionOptimizationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    cache_manager = Depends(get_cache_manager),
    rate_limit: Dict[str, Any] = Depends(check_rate_limit)
) -> CaptionOptimizationResponse:
    """Optimize an existing Instagram caption for better engagement."""
    
    # Validate input
    validate_non_empty_string(request.caption, "caption")
    
    # Generate cache key
    cache_key = generate_cache_key(
        "caption_optimization",
        caption=request.caption,
        style=request.style.value,
        audience=request.audience.value,
        enhancement_level=request.enhancement_level
    )
    
    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        return deserialize_from_cache(cached_result, CaptionOptimizationResponse)
    
    # Analyze original quality
    original_quality = engine.analyze_quality(
        caption=request.caption,
        style=request.style,
        audience=request.audience
    )
    
    # Optimize caption
    optimized_caption, optimized_quality = await engine.optimize_content(
        caption=request.caption,
        style=request.style,
        audience=request.audience,
        enhancement_level=request.enhancement_level,
        preserve_meaning=request.preserve_meaning
    )
    
    # Calculate improvement
    improvement_score = calculate_improvement_percentage(
        original_quality.overall_score,
        optimized_quality.overall_score
    )
    
    response = CaptionOptimizationResponse(
        status="success",
        original_caption=request.caption,
        optimized_caption=optimized_caption,
        original_quality=QualityMetricsResponse(**original_quality.model_dump()),
        optimized_quality=QualityMetricsResponse(**optimized_quality.model_dump()),
        improvements_applied=optimized_quality.suggestions,
        improvement_score=improvement_score
    )
    
    # Cache result
    await cache_manager.set(
        cache_key,
        serialize_for_cache(response),
        ttl=1800  # 30 minutes
    )
    
    return response


@router.post("/batch-optimize", response_model=BatchOptimizationResponse)
@handle_api_errors
@log_performance_metrics("batch_optimization")
async def batch_optimize_captions(
    request: BatchOptimizationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    background_tasks: BackgroundTasks,
    rate_limit: Dict[str, Any] = Depends(check_rate_limit)
) -> BatchOptimizationResponse:
    """Optimize multiple captions in batch with controlled concurrency."""
    
    # Validate input
    if not request.captions:
        raise ValueError("At least one caption is required")
    
    async def optimize_single_caption(caption: str) -> CaptionOptimizationResponse:
        """Optimize a single caption."""
        try:
            # Analyze original
            original_quality = engine.analyze_quality(
                caption=caption,
                style=request.style,
                audience=request.audience
            )
            
            # Optimize
            optimized_caption, optimized_quality = await engine.optimize_content(
                caption=caption,
                style=request.style,
                audience=request.audience,
                enhancement_level=request.enhancement_level
            )
            
            # Calculate improvement
            improvement_score = calculate_improvement_percentage(
                original_quality.overall_score,
                optimized_quality.overall_score
            )
            
            return CaptionOptimizationResponse(
                status="success",
                original_caption=caption,
                optimized_caption=optimized_caption,
                original_quality=QualityMetricsResponse(**original_quality.model_dump()),
                optimized_quality=QualityMetricsResponse(**optimized_quality.model_dump()),
                improvements_applied=optimized_quality.suggestions,
                improvement_score=improvement_score
            )
            
        except Exception as e:
            logger.error(f"Failed to optimize caption: {e}")
            return CaptionOptimizationResponse(
                status="failed",
                original_caption=caption,
                optimized_caption=caption,
                original_quality=QualityMetricsResponse(
                    overall_score=0,
                    grade="F",
                    hook_strength=0,
                    engagement_potential=0,
                    readability=0,
                    cta_effectiveness=0,
                    specificity=0,
                    issues=["Processing failed"],
                    suggestions=[],
                    performance_expectation="Unable to analyze"
                ),
                optimized_quality=QualityMetricsResponse(
                    overall_score=0,
                    grade="F",
                    hook_strength=0,
                    engagement_potential=0,
                    readability=0,
                    cta_effectiveness=0,
                    specificity=0,
                    issues=["Processing failed"],
                    suggestions=[],
                    performance_expectation="Unable to analyze"
                ),
                improvements_applied=[],
                improvement_score=0.0
            )
    
    # Ultra-fast batch processing with high concurrency
    results = await batch_optimize_ultra_fast(
        captions=request.captions,
        engine=engine,
        style=request.style,
        audience=request.audience,
        enhancement_level=request.enhancement_level
    )
    
    # Calculate batch statistics
    successful_results = [r for r in results if isinstance(r, CaptionOptimizationResponse) and r.status == "success"]
    failed_count = len(results) - len(successful_results)
    
    avg_improvement = 0.0
    if successful_results:
        avg_improvement = sum(r.improvement_score for r in successful_results) / len(successful_results)
    
    batch_statistics = {
        "total_processed": len(request.captions),
        "successful": len(successful_results),
        "failed": failed_count,
        "success_rate": len(successful_results) / len(request.captions) * 100,
        "average_improvement": round(avg_improvement, 2)
    }
    
    return BatchOptimizationResponse(
        status="success" if failed_count == 0 else "partial",
        results=results,
        batch_statistics=batch_statistics
    )


@router.post("/hashtags/generate", response_model=HashtagGenerationResponse)
@handle_api_errors
@measure_execution_time
async def generate_hashtags(
    request: HashtagGenerationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    cache_manager = Depends(get_cache_manager),
    rate_limit: Dict[str, Any] = Depends(check_rate_limit)
) -> HashtagGenerationResponse:
    """Generate optimized hashtags for Instagram content."""
    
    # Validate input
    if not request.content_keywords:
        raise ValueError("At least one content keyword is required")
    
    # Generate cache key
    cache_key = generate_cache_key(
        "hashtag_generation",
        keywords=request.content_keywords,
        audience=request.audience.value,
        strategy=request.strategy.value,
        count=request.count
    )
    
    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        return deserialize_from_cache(cached_result, HashtagGenerationResponse)
    
    # Generate hashtags
    hashtags_result = engine.generate_hashtags(
        keywords=request.content_keywords,
        audience=request.audience,
        strategy=request.strategy,
        count=request.count
    )
    
    response = HashtagGenerationResponse(
        status="success",
        hashtags=hashtags_result.hashtags,
        strategy_used=request.strategy,
        performance_estimate=hashtags_result.performance_estimate,
        category_distribution=hashtags_result.category_distribution
    )
    
    # Cache result
    await cache_manager.set(
        cache_key,
        serialize_for_cache(response),
        ttl=3600  # 1 hour
    )
    
    return response


@router.get("/health", response_model=HealthCheckResponse)
@handle_api_errors
async def health_check(
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    gmt_system: SimplifiedGMTSystem = Depends(get_gmt_system),
    health_checker = Depends(get_health_checker),
    cache_manager = Depends(get_cache_manager)
) -> HealthCheckResponse:
    """Comprehensive health check for all system components."""
    
    # Run health checks in parallel
    health_tasks = [
        health_checker.check_engine_health(engine),
        health_checker.check_gmt_health(gmt_system),
        health_checker.check_redis_health(cache_manager.redis_client)
    ]
    
    engine_health, gmt_health, redis_health = await asyncio.gather(
        *health_tasks, return_exceptions=True
    )
    
    # Determine overall status
    all_healthy = all(
        isinstance(h, dict) and h.get("status") == "healthy" 
        for h in [engine_health, gmt_health]
    )
    
    overall_status = "healthy" if all_healthy else "degraded"
    
    components = {
        "captions_engine": engine_health,
        "gmt_system": gmt_health,
        "redis_cache": redis_health
    }
    
    return HealthCheckResponse(
        status=overall_status,
        components=components,
        performance_metrics={
            "cache_available": redis_health.get("status") == "healthy",
            "all_services_operational": all_healthy
        }
    )


@router.get("/quality-guidelines", response_model=QualityGuidelinesResponse)
@handle_api_errors
async def get_quality_guidelines() -> QualityGuidelinesResponse:
    """Get comprehensive quality guidelines for Instagram captions."""
    
    guidelines = {
        "hook_strategies": {
            "question_hooks": [
                "Start with an intriguing question",
                "Use 'What if...' scenarios",
                "Ask about experiences or opinions"
            ],
            "statement_hooks": [
                "Make bold, surprising statements",
                "Share counterintuitive facts",
                "Use emotional triggers"
            ],
            "story_hooks": [
                "Begin with 'I used to...'",
                "Share a moment of transformation",
                "Use sensory details"
            ]
        },
        "engagement_tactics": {
            "call_to_action": [
                "Ask followers to share their thoughts",
                "Request specific actions (save, share, comment)",
                "Create urgency with time-sensitive offers"
            ],
            "community_building": [
                "Use 'we' language to create inclusion",
                "Reference shared experiences",
                "Ask followers to tag friends"
            ]
        },
        "content_structure": {
            "opening": "Strong hook in first 1-2 lines",
            "body": "Value-driven content with specific examples",
            "closing": "Clear call-to-action",
            "hashtags": "Mix of trending and niche hashtags"
        }
    }
    
    examples = {
        "good_hooks": [
            "The mistake that cost me $10,000 in sales...",
            "What if I told you there's a 5-minute morning routine that changed everything?",
            "Yesterday I was scrolling through old photos and realized something huge..."
        ],
        "weak_hooks": [
            "Hey everyone, how's your day going?",
            "Here's another post about my business...",
            "Hope you're having a great week!"
        ]
    }
    
    metrics_explanation = {
        "hook_strength": "Measures how compelling and attention-grabbing your opening is",
        "engagement_potential": "Predicts likelihood of likes, comments, and shares",
        "readability": "Assesses how easy your caption is to read and understand",
        "cta_effectiveness": "Evaluates clarity and persuasiveness of your call-to-action",
        "specificity": "Measures how detailed and concrete your content is"
    }
    
    return QualityGuidelinesResponse(
        guidelines=guidelines,
        examples=examples,
        metrics_explanation=metrics_explanation
    )


@router.get("/timezone/{timezone}/insights", response_model=TimezoneInsightsResponse)
@handle_api_errors
async def get_timezone_insights(
    timezone: str,
    gmt_system: SimplifiedGMTSystem = Depends(get_gmt_system),
    cache_manager = Depends(get_cache_manager)
) -> TimezoneInsightsResponse:
    """Get comprehensive insights for a specific timezone."""
    
    # Validate timezone
    validate_non_empty_string(timezone, "timezone")
    
    # Generate cache key
    cache_key = generate_cache_key("timezone_insights", timezone=timezone)
    
    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        return deserialize_from_cache(cached_result, TimezoneInsightsResponse)
    
    # Get timezone insights
    insights = gmt_system.get_timezone_insights(timezone)
    
    response = TimezoneInsightsResponse(
        timezone=timezone,
        optimal_posting_times=insights.optimal_posting_times,
        peak_engagement_windows=insights.peak_engagement_windows,
        cultural_context=insights.cultural_context,
        engagement_predictions=insights.engagement_predictions,
        recommendations=insights.recommendations,
        confidence_score=insights.confidence_score
    )
    
    # Cache result (shorter TTL for timezone data)
    await cache_manager.set(
        cache_key,
        serialize_for_cache(response),
        ttl=1800  # 30 minutes
    )
    
    return response


@router.get("/metrics")
@handle_api_errors
@ultra_fast_cache(ttl=60)
async async def get_api_metrics(request: Request) -> Dict[str, Any]:
    """Get API performance metrics with ultra-fast caching."""
    
    # Get performance middleware metrics if available
    performance_metrics = {}
    
    for middleware in request.app.middleware_stack:
        if hasattr(middleware, 'middleware') and hasattr(middleware.middleware, 'get_metrics'):
            performance_metrics = middleware.middleware.get_metrics()
            break
    
    # Get speed optimization stats
    speed_stats = get_performance_stats()
    
    return format_response_fast({
        "api_version": "2.0.0",
        "performance_metrics": performance_metrics,
        "speed_optimizations": speed_stats,
        "status": "operational",
        "optimization_level": "ultra_fast"
    }) 