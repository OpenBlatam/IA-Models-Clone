from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import get_settings, get_cors_config, get_logging_config, validate_environment_variables
from .middleware import (
from .api_optimized import router as optimized_router
from .dependencies import dependency_lifespan, cleanup_dependencies
import logging.config
    from .api_ultra_fast import ultra_router
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi import Field
from .models import (
from .gmt_instagram_agent import GMTInstagramAgent
from .service import InstagramCaptionsService
        from .core import InstagramCaptionsEngine
        from .core import InstagramCaptionsEngine
        from .core import InstagramCaptionsEngine
from typing import Any, List, Dict, Optional
import asyncio
"""
Instagram Captions API v2.0.

Modern FastAPI implementation with optimization, caching, and comprehensive error handling.
"""



    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware,
    SecurityMiddleware,
    ErrorHandlingMiddleware,
    CacheHeaderMiddleware
)

# Configure logging
settings = get_settings()
logging.config.dictConfig(get_logging_config(settings))

logger = logging.getLogger(__name__)

# Validate environment on startup
env_errors = validate_environment_variables()
if env_errors:
    logger.error("Environment validation errors:")
    for error in env_errors:
        logger.error(f"  - {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting Instagram Captions API v{settings.api_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")
    
    async with dependency_lifespan():
        yield
    
    logger.info("Shutting down Instagram Captions API")


def create_optimized_app() -> FastAPI:
    """Create optimized FastAPI application with all middleware and configurations."""
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # Add middleware in reverse order (last added = first executed)
    
    # CORS middleware
    cors_config = get_cors_config(settings)
    app.add_middleware(
        CORSMiddleware,
        **cors_config
    )
    
    # Cache headers middleware
    app.add_middleware(CacheHeaderMiddleware)
    
    # Security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Performance monitoring middleware
    if settings.performance.enable_metrics:
        app.add_middleware(
            PerformanceMonitoringMiddleware,
            slow_request_threshold=settings.performance.slow_request_threshold
        )
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Error handling middleware (should be first/outer)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Include optimized routers
    app.include_router(optimized_router)
    
    # Include ultra-fast router
    app.include_router(ultra_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """API root endpoint with version information."""
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "description": settings.api_description,
            "environment": settings.environment.value,
            "status": "operational",
            "endpoints": {
                "api_v2": "/api/v2/instagram-captions",
                "api_v2_1_ultra_fast": "/api/v2.1/instagram-captions",
                "health": "/api/v2/instagram-captions/health",
                "ultra_health": "/api/v2.1/instagram-captions/health",
                "docs": "/docs",
                "openapi": "/openapi.json"
            }
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    return app


# Create the optimized app instance
app = create_optimized_app()

# Legacy compatibility: maintain old router structure


    InstagramCaptionRequest,
    InstagramCaptionResponse,
    InstagramCaptionError,
    GlobalCampaignRequest,
    CampaignStatus,
    TimeZone,
    TimeZoneInfo,
    CaptionStyle,
    InstagramTarget,
    HashtagStrategy
)

# Legacy router for backward compatibility
instagram_captions_router = APIRouter(
    prefix="/instagram-captions",
    tags=["Instagram Captions (Legacy)"],
    responses={404: {"description": "Not found"}},
    deprecated=True
)

# Global agent instance
_gmt_agent: Optional[GMTInstagramAgent] = None

async def get_gmt_agent() -> GMTInstagramAgent:
    """Get or create GMT Instagram agent instance."""
    global _gmt_agent
    if _gmt_agent is None:
        _gmt_agent = GMTInstagramAgent()
    return _gmt_agent

@instagram_captions_router.get("/")
async def root():
    """Root endpoint with feature information."""
    return {
        "message": "Instagram Captions GMT Generator",
        "version": "1.0.0",
        "features": [
            "Multi-timezone caption generation",
            "LangChain integration", 
            "OpenRouter support",
            "OpenAI integration",
            "Regional content adaptation",
            "Global campaign scheduling",
            "Hashtag optimization",
            "Engagement prediction"
        ],
        "supported_timezones": [tz.value for tz in TimeZone],
        "caption_styles": [style.value for style in CaptionStyle],
        "target_audiences": [target.value for target in InstagramTarget],
        "hashtag_strategies": [strategy.value for strategy in HashtagStrategy]
    }

@instagram_captions_router.post("/generate", response_model=InstagramCaptionResponse)
async def generate_instagram_captions(
    request: InstagramCaptionRequest,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Generate Instagram captions with GMT awareness."""
    try:
        logger.info(f"Generating Instagram captions for timezone {request.target_timezone}")
        
        response = await agent.generate_gmt_captions(request)
        
        logger.info(f"Successfully generated {len(response.variations)} caption variations")
        return response
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="GENERATION_FAILED",
                error_message=str(e),
                details={"timezone": request.target_timezone.value}
            ).dict()
        )

@instagram_captions_router.post("/campaign/global")
async def create_global_campaign(
    request: GlobalCampaignRequest,
    background_tasks: BackgroundTasks,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Create a global Instagram campaign across multiple timezones."""
    try:
        logger.info(f"Creating global campaign for {len(request.target_timezones)} timezones")
        
        campaign_id = await agent.create_global_campaign(request)
        
        return {
            "campaign_id": campaign_id,
            "status": "scheduled",
            "target_timezones": [tz.value for tz in request.target_timezones],
            "total_posts": len(request.target_timezones),
            "created_at": datetime.utcnow().isoformat(),
            "message": "Global campaign scheduled successfully"
        }
        
    except Exception as e:
        logger.error(f"Global campaign creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="CAMPAIGN_CREATION_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/campaign/{campaign_id}", response_model=CampaignStatus)
async def get_campaign_status(
    campaign_id: str,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get status of a global campaign."""
    try:
        status = agent.get_campaign_status(campaign_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=InstagramCaptionError(
                    error_code="CAMPAIGN_NOT_FOUND",
                    error_message=f"Campaign {campaign_id} not found"
                ).dict()
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign status: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="STATUS_RETRIEVAL_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/timezone/{timezone}", response_model=TimeZoneInfo)
async def get_timezone_status(
    timezone: TimeZone,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get current status and optimal posting times for a timezone."""
    try:
        tz_info = agent.get_timezone_status(timezone)
        return tz_info
        
    except Exception as e:
        logger.error(f"Failed to get timezone status: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="TIMEZONE_STATUS_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/timezone/{timezone}/optimal-times")
async def get_optimal_posting_times(
    timezone: TimeZone,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get optimal posting times for a specific timezone."""
    try:
        optimal_hours = agent.get_optimal_posting_times(timezone)
        tz_info = agent.get_timezone_status(timezone)
        
        return {
            "timezone": timezone.value,
            "current_hour": tz_info.local_hour,
            "optimal_hours": optimal_hours,
            "is_optimal_now": tz_info.optimal_posting_window,
            "is_peak_time": tz_info.peak_engagement_time,
            "next_optimal_in_hours": min([h for h in optimal_hours if h > tz_info.local_hour] + 
                                       [optimal_hours[0] + 24]) - tz_info.local_hour
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimal times: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="OPTIMAL_TIMES_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/timezones")
async def get_all_timezones_status(
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get status for all supported timezones."""
    try:
        timezones_status = {}
        
        for tz in TimeZone:
            tz_info = agent.get_timezone_status(tz)
            timezones_status[tz.value] = {
                "current_time": tz_info.current_time.isoformat(),
                "local_hour": tz_info.local_hour,
                "optimal_posting_window": tz_info.optimal_posting_window,
                "peak_engagement_time": tz_info.peak_engagement_time,
                "utc_offset": tz_info.utc_offset
            }
        
        return {
            "timezones": timezones_status,
            "total_supported": len(TimeZone),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get all timezones status: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="ALL_TIMEZONES_STATUS_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/stats")
async def get_agent_stats(
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get Instagram captions agent statistics."""
    try:
        stats = await agent.get_enhanced_agent_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="STATS_RETRIEVAL_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/timezone/{timezone}/insights")
async def get_timezone_insights(
    timezone: TimeZone,
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get comprehensive timezone insights with engagement predictions."""
    try:
        insights = await agent.get_timezone_insights(timezone)
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get timezone insights: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="TIMEZONE_INSIGHTS_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.get("/gmt/system-health")
async def get_gmt_system_health(
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Get GMT system health and performance metrics."""
    try:
        health = agent.gmt_system.get_system_health()
        return {
            "gmt_system": health,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational" if health["status"] == "healthy" else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Failed to get GMT system health: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="GMT_HEALTH_CHECK_FAILED",
                error_message=str(e)
            ).dict()
        )

@instagram_captions_router.post("/analyze/engagement-score")
async def analyze_engagement_score(
    timezone: TimeZone,
    hour: int = Field(..., ge=0, le=23, description="Hour to analyze (0-23)"),
    audience_type: str = Field("general", description="Audience type for analysis"),
    agent: GMTInstagramAgent = Depends(get_gmt_agent)
):
    """Analyze engagement score for specific timezone and hour."""
    try:
        score = agent.gmt_system.calculate_engagement_score(timezone, hour, audience_type)
        
        # Get tier information
        tier = "low"
        if score >= 2.0:
            tier = "ultra_peak"
        elif score >= 1.5:
            tier = "peak"  
        elif score >= 1.2:
            tier = "high"
        elif score >= 0.8:
            tier = "standard"
        
        return {
            "timezone": timezone.value,
            "hour": hour,
            "audience_type": audience_type,
            "engagement_score": round(score, 2),
            "engagement_tier": tier,
            "recommendation": _get_score_recommendation(score),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze engagement score: {e}")
        raise HTTPException(
            status_code=500,
            detail=InstagramCaptionError(
                error_code="ENGAGEMENT_ANALYSIS_FAILED",
                error_message=str(e)
            ).dict()
        )

def _get_score_recommendation(score: float) -> str:
    """Get recommendation based on engagement score."""
    if score >= 2.0:
        return "🚀 Ultra-peak time - Perfect for viral content and major announcements!"
    elif score >= 1.5:
        return "📈 Peak engagement - Excellent time for important posts and campaigns!"
    elif score >= 1.2:
        return "✅ High engagement - Good time for regular content posting!"
    elif score >= 0.8:
        return "⏰ Standard engagement - Acceptable posting time!"
    else:
        return "⏳ Low engagement - Consider waiting for better timing or targeting different audience!"

@instagram_captions_router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        agent = await get_gmt_agent()
        
        # Check AI providers availability
        providers_status = {}
        for name, provider in agent.captions_service.providers.items():
            providers_status[name] = provider.is_available()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "providers": providers_status,
            "features": {
                "gmt_support": True,
                "langchain_integration": "langchain" in providers_status,
                "openai_integration": "openai" in providers_status,
                "openrouter_integration": "openrouter" in providers_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Configuration endpoints
@instagram_captions_router.get("/config/styles")
async def get_caption_styles():
    """Get available caption styles."""
    return {
        "styles": [
            {
                "value": style.value,
                "name": style.value.replace("_", " ").title(),
                "description": f"Generate {style.value} style captions"
            }
            for style in CaptionStyle
        ]
    }

@instagram_captions_router.get("/config/audiences")
async def get_target_audiences():
    """Get available target audiences."""
    return {
        "audiences": [
            {
                "value": audience.value,
                "name": audience.value.replace("_", " ").title(),
                "description": f"Target {audience.value} audience"
            }
            for audience in InstagramTarget
        ]
    }

@instagram_captions_router.get("/config/hashtag-strategies")
async def get_hashtag_strategies():
    """Get available hashtag strategies."""
    return {
        "strategies": [
            {
                "value": strategy.value,
                "name": strategy.value.replace("_", " ").title(),
                "description": f"Use {strategy.value} hashtag strategy"
            }
            for strategy in HashtagStrategy
        ]
    }

@instagram_captions_router.post("/analyze-quality")
async def analyze_caption_quality(
    caption: str = Field(..., description="Caption to analyze"),
    style: CaptionStyle = Field(CaptionStyle.CASUAL, description="Caption style"),
    audience: InstagramTarget = Field(InstagramTarget.GENERAL, description="Target audience")
):
    """Analyze the quality of an existing caption."""
    try:
        
        engine = InstagramCaptionsEngine()
        quality_metrics = engine.analyze_quality(caption)
        
        quality_report = engine.get_quality_report(quality_metrics)
        
        return {
            "status": "success",
            "quality_analysis": quality_report,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing caption quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")

@instagram_captions_router.post("/optimize-caption")
async def optimize_existing_caption(
    caption: str = Field(..., description="Caption to optimize"),
    style: CaptionStyle = Field(CaptionStyle.CASUAL, description="Caption style"),
    audience: InstagramTarget = Field(InstagramTarget.GENERAL, description="Target audience")
):
    """Optimize an existing caption for better quality and engagement."""
    try:
        
        engine = InstagramCaptionsEngine()
        optimized_caption, quality_metrics = await engine.optimize_content(caption, style, audience)
        
        quality_report = engine.get_quality_report(quality_metrics)
        
        return {
            "status": "success",
            "original_caption": caption,
            "optimized_caption": optimized_caption,
            "quality_metrics": quality_report,
            "improvements": {
                "issues_fixed": quality_metrics.issues,
                "suggestions_applied": quality_metrics.suggestions
            },
            "optimized_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Caption optimization failed: {str(e)}")

@instagram_captions_router.get("/quality-guidelines")
async def get_quality_guidelines():
    """Get quality guidelines for creating high-engagement captions."""
    
    guidelines = {
        "hook_guidelines": {
            "strong_hooks": [
                "Start with a question that makes people think",
                "Use numbers (3 ways to..., 5 secrets of...)",
                "Share a controversial or bold statement",
                "Tell a mini-story in the first line",
                "Use 'What if...' or 'Did you know...' starters"
            ],
            "avoid": [
                "Generic greetings like 'Hey everyone'",
                "Boring statements without intrigue",
                "Starting with your brand name"
            ]
        },
        "engagement_optimization": {
            "best_practices": [
                "Use 'you' and 'your' to create connection",
                "Ask specific questions, not just 'thoughts?'",
                "Include clear call-to-actions",
                "Add line breaks for mobile readability",
                "Use 1-2 emojis per section, not excessive"
            ],
            "proven_ctas": [
                "What's your experience with this? Share below! 👇",
                "Tag someone who needs to see this! 🏷️",
                "Save this for later reference! 📌",
                "Drop your best tip in the comments! 💬"
            ]
        },
        "content_structure": {
            "framework": [
                "Hook (attention-grabbing first line)",
                "Value/Story (main content with specifics)",
                "Connection (personal/relatable element)",
                "Action (clear call-to-action)"
            ],
            "optimal_length": {
                "posts": "150-300 characters for optimal engagement",
                "stories": "50-150 characters",
                "reels": "80-200 characters",
                "carousel": "200-400 characters"
            }
        },
        "quality_metrics": {
            "excellent": "90-100% - Outstanding engagement potential",
            "very_good": "80-89% - Strong performance expected", 
            "good": "70-79% - Good engagement likely",
            "average": "60-69% - Room for improvement",
            "poor": "Below 60% - Needs significant enhancement"
        }
    }
    
    return {
        "status": "success",
        "guidelines": guidelines,
        "generated_at": datetime.utcnow().isoformat()
    }

@instagram_captions_router.post("/batch-optimize")
async def batch_optimize_captions(
    captions: List[str] = Field(..., description="List of captions to optimize"),
    style: CaptionStyle = Field(CaptionStyle.CASUAL, description="Caption style"),
    audience: InstagramTarget = Field(InstagramTarget.GENERAL, description="Target audience")
):
    """Optimize multiple captions in batch for better efficiency."""
    try:
        
        if len(captions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 captions per batch")
        
        engine = InstagramCaptionsEngine()
        results = []
        
        for i, caption in enumerate(captions):
            try:
                optimized_caption, quality_metrics = await engine.optimize_content(caption, style, audience)
                quality_report = engine.get_quality_report(quality_metrics)
                
                results.append({
                    "index": i,
                    "original_caption": caption,
                    "optimized_caption": optimized_caption,
                    "quality_score": quality_report["overall_score"],
                    "grade": quality_report["grade"],
                    "issues_fixed": len(quality_metrics.issues),
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "original_caption": caption,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Calculate batch statistics
        successful_results = [r for r in results if r["status"] == "success"]
        avg_quality_score = sum(r["quality_score"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            "status": "success",
            "results": results,
            "batch_stats": {
                "total_captions": len(captions),
                "successful_optimizations": len(successful_results),
                "failed_optimizations": len(captions) - len(successful_results),
                "average_quality_score": round(avg_quality_score, 1)
            },
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}") 