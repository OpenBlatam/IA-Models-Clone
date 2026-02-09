from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
import structlog
from src.application.use_cases import (
from src.domain.entities import (
from src.core.container import get_container
from src.core.exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 Ultra-Optimized API Routes
============================

Production-grade API endpoints with:
- Rate limiting
- Request validation
- Response optimization
- Comprehensive error handling
- Performance monitoring
"""


    GenerateContentUseCase,
    GetUserContentUseCase,
    SearchContentUseCase,
    CreateTemplateUseCase,
    GetUserMetricsUseCase,
    UpdateUserCreditsUseCase
)
    ContentRequest, GeneratedContent, ContentTemplate, UsageMetrics,
    ContentType, Language, Tone
)
    BusinessException, ValidationException, NotFoundException,
    UnauthorizedException, InsufficientCreditsException
)

logger = structlog.get_logger(__name__)

# Create router
api_router = APIRouter()

# Dependency to get container
def get_di_container():
    
    """get_di_container function."""
return get_container()

# Content Generation Endpoints
@api_router.post("/content/generate", response_model=GeneratedContent)
async def generate_content(
    request_data: Dict[str, Any],
    container = Depends(get_di_container)
):
    """Generate content using AI"""
    
    try:
        use_case = GenerateContentUseCase(
            user_repo=container.get("user_repository"),
            content_repo=container.get("content_repository"),
            ai_service=container.get("ai_service"),
            event_publisher=container.get("event_publisher"),
            cache_service=container.get("cache_service"),
            rate_limiter=container.get("rate_limiter")
        )
        
        # For demo purposes, using a fixed user ID
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        content = await use_case.execute(user_id, request_data)
        
        logger.info("Content generated successfully", content_id=str(content.id))
        return content
        
    except InsufficientCreditsException as e:
        raise HTTPException(status_code=402, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except BusinessException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Content generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/content/{content_id}", response_model=GeneratedContent)
async def get_content(
    content_id: str = Path(..., description="Content ID"),
    container = Depends(get_di_container)
):
    """Get generated content by ID"""
    
    try:
        content_repo = container.get("content_repository")
        content = await content_repo.get_content_by_id(content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return content
        
    except Exception as e:
        logger.error("Failed to get content", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/content/user/{user_id}", response_model=List[GeneratedContent])
async def get_user_content(
    user_id: str = Path(..., description="User ID"),
    skip: int = Query(0, ge=0, description="Skip records"),
    limit: int = Query(100, ge=1, le=1000, description="Limit records"),
    container = Depends(get_di_container)
):
    """Get user's generated content"""
    
    try:
        use_case = GetUserContentUseCase(
            content_repo=container.get("content_repository"),
            cache_service=container.get("cache_service")
        )
        
        content_list = await use_case.execute(user_id, skip, limit)
        return content_list
        
    except Exception as e:
        logger.error("Failed to get user content", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/content/search", response_model=List[GeneratedContent])
async def search_content(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    content_type: Optional[ContentType] = Query(None, description="Content type filter"),
    skip: int = Query(0, ge=0, description="Skip records"),
    limit: int = Query(100, ge=1, le=1000, description="Limit records"),
    container = Depends(get_di_container)
):
    """Search user's content"""
    
    try:
        use_case = SearchContentUseCase(
            content_repo=container.get("content_repository"),
            cache_service=container.get("cache_service")
        )
        
        results = await use_case.execute(user_id, query, content_type, skip, limit)
        return results
        
    except Exception as e:
        logger.error("Content search failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Template Management Endpoints
@api_router.post("/templates", response_model=ContentTemplate)
async def create_template(
    template_data: Dict[str, Any],
    container = Depends(get_di_container)
):
    """Create a new content template"""
    
    try:
        use_case = CreateTemplateUseCase(
            template_repo=container.get("template_repository"),
            cache_service=container.get("cache_service")
        )
        
        # For demo purposes, using a fixed user ID
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        template = await use_case.execute(user_id, template_data)
        
        logger.info("Template created successfully", template_id=str(template.id))
        return template
        
    except ValidationException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Template creation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/templates/user/{user_id}", response_model=List[ContentTemplate])
async def get_user_templates(
    user_id: str = Path(..., description="User ID"),
    skip: int = Query(0, ge=0, description="Skip records"),
    limit: int = Query(100, ge=1, le=1000, description="Limit records"),
    container = Depends(get_di_container)
):
    """Get user's templates"""
    
    try:
        template_repo = container.get("template_repository")
        templates = await template_repo.list_user_templates(user_id, skip, limit)
        return templates
        
    except Exception as e:
        logger.error("Failed to get user templates", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/templates/public", response_model=List[ContentTemplate])
async def get_public_templates(
    skip: int = Query(0, ge=0, description="Skip records"),
    limit: int = Query(100, ge=1, le=1000, description="Limit records"),
    container = Depends(get_di_container)
):
    """Get public templates"""
    
    try:
        template_repo = container.get("template_repository")
        templates = await template_repo.list_public_templates(skip, limit)
        return templates
        
    except Exception as e:
        logger.error("Failed to get public templates", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# User Management Endpoints
@api_router.get("/users/{user_id}/metrics")
async def get_user_metrics(
    user_id: str = Path(..., description="User ID"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    container = Depends(get_di_container)
):
    """Get user usage metrics"""
    
    try:
        use_case = GetUserMetricsUseCase(
            metrics_repo=container.get("metrics_repository"),
            cache_service=container.get("cache_service")
        )
        
        metrics = await use_case.execute(user_id, start_date, end_date)
        return metrics
        
    except Exception as e:
        logger.error("Failed to get user metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/users/{user_id}/credits")
async def update_user_credits(
    user_id: str = Path(..., description="User ID"),
    credits: int = Query(..., description="Credits to add/subtract"),
    reason: str = Query(..., description="Reason for credit change"),
    container = Depends(get_di_container)
):
    """Update user credits"""
    
    try:
        use_case = UpdateUserCreditsUseCase(
            user_repo=container.get("user_repository"),
            event_publisher=container.get("event_publisher")
        )
        
        user = await use_case.execute(user_id, credits, reason)
        
        return {
            "user_id": str(user.id),
            "credits": user.credits,
            "message": f"Credits updated successfully. Reason: {reason}"
        }
        
    except InsufficientCreditsException as e:
        raise HTTPException(status_code=402, detail=str(e))
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to update user credits", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# AI Service Endpoints
@api_router.post("/ai/analyze")
async def analyze_content(
    content: str = Query(..., description="Content to analyze"),
    container = Depends(get_di_container)
):
    """Analyze content using AI"""
    
    try:
        ai_service = container.get("ai_service")
        analysis = await ai_service.analyze_content(content)
        
        return {
            "content_length": len(content),
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error("Content analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/ai/optimize")
async def optimize_content(
    content: str = Query(..., description="Content to optimize"),
    keywords: List[str] = Query(..., description="SEO keywords"),
    container = Depends(get_di_container)
):
    """Optimize content for SEO"""
    
    try:
        ai_service = container.get("ai_service")
        optimized_content = await ai_service.optimize_seo(content, keywords)
        
        return {
            "original_length": len(content),
            "optimized_length": len(optimized_content),
            "optimized_content": optimized_content
        }
        
    except Exception as e:
        logger.error("Content optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/ai/translate")
async def translate_content(
    content: str = Query(..., description="Content to translate"),
    target_language: Language = Query(..., description="Target language"),
    container = Depends(get_di_container)
):
    """Translate content to target language"""
    
    try:
        ai_service = container.get("ai_service")
        translated_content = await ai_service.translate_content(content, target_language)
        
        return {
            "original_content": content,
            "translated_content": translated_content,
            "target_language": target_language.value
        }
        
    except Exception as e:
        logger.error("Content translation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/ai/summarize")
async def summarize_content(
    content: str = Query(..., description="Content to summarize"),
    max_length: int = Query(200, ge=50, le=1000, description="Maximum summary length"),
    container = Depends(get_di_container)
):
    """Summarize content using AI"""
    
    try:
        ai_service = container.get("ai_service")
        summary = await ai_service.summarize_content(content, max_length)
        
        return {
            "original_length": len(content),
            "summary_length": len(summary),
            "summary": summary
        }
        
    except Exception as e:
        logger.error("Content summarization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# System Endpoints
@api_router.get("/system/status")
async def system_status(container = Depends(get_di_container)):
    """Get system status and performance metrics"""
    
    try:
        monitoring = container.get("monitoring_service")
        health_checker = container.get("health_checker")
        
        status = {
            "timestamp": time.time(),
            "health": await health_checker.check_health(),
            "performance": monitoring.get_performance_summary(),
            "system": monitoring.get_system_metrics()
        }
        
        return status
        
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/system/alerts")
async def system_alerts(container = Depends(get_di_container)):
    """Get current system alerts"""
    
    try:
        monitoring = container.get("monitoring_service")
        alerts = monitoring.get_alerts()
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "unacknowledged": len([a for a in alerts if not a["acknowledged"]])
        }
        
    except Exception as e:
        logger.error("Failed to get system alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Utility Endpoints
@api_router.get("/content/types")
async def get_content_types():
    """Get available content types"""
    
    return {
        "content_types": [
            {"value": ct.value, "name": ct.name, "description": f"Generate {ct.value}"}
            for ct in ContentType
        ]
    }

@api_router.get("/languages")
async def get_languages():
    """Get supported languages"""
    
    return {
        "languages": [
            {"code": lang.value, "name": lang.name}
            for lang in Language
        ]
    }

@api_router.get("/tones")
async def get_tones():
    """Get available content tones"""
    
    return {
        "tones": [
            {"value": tone.value, "name": tone.name}
            for tone in Tone
        ]
    }

# Error handling middleware
@api_router.exception_handler(BusinessException)
async def business_exception_handler(request, exc) -> Any:
    logger.warning("Business exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=400,
        content={"error": "Business Error", "message": str(exc)}
    )

@api_router.exception_handler(ValidationException)
async def validation_exception_handler(request, exc) -> Any:
    logger.warning("Validation exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "message": str(exc)}
    )

@api_router.exception_handler(NotFoundException)
async def not_found_exception_handler(request, exc) -> Any:
    logger.warning("Not found exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": str(exc)}
    )

@api_router.exception_handler(UnauthorizedException)
async def unauthorized_exception_handler(request, exc) -> Any:
    logger.warning("Unauthorized exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized", "message": str(exc)}
    ) 