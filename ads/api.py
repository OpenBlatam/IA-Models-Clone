from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import base64
import io
from PIL import Image
import numpy as np
import cv2
import pyvips
import logging
from datetime import datetime
from onyx.server.auth_check import check_router_auth
from onyx.server.utils import BasicAuthenticationError
from onyx.utils.logger import setup_logger
from onyx.core.auth import get_current_user
from onyx.core.functions import format_response, handle_error
from onyx.server.features.ads.db_service import AdsDBService
from onyx.server.features.ads.service import AdsService
    from rembg import remove, new_session
        from onyx.llm.interface import generate_ads_lcel_streaming_parallel
from typing import Any, List, Dict, Optional
import asyncio
"""
Deprecated module. Use unified routers under `agents.backend.onyx.server.features.ads.api` package.

This file re-exports the unified core router to preserve backward compatibility for imports that still
reference `...features.ads.api` as a module.
"""


logger = setup_logger()

try:
    from .api import core as _core
    router = _core.router  # Back-compat: expose a router from unified API
except Exception:
    # Provide a minimal router to avoid import errors
    from fastapi import APIRouter
    router = APIRouter(prefix="/ads/legacy", tags=["ads-legacy"])

# Models
class AdsRequest(BaseModel):
    """Request model for ads generation."""
    url: str
    type: str  # "ads", "brand-kit", "custom"
    prompt: Optional[str] = None
    target_audience: Optional[str] = None
    context: Optional[str] = None
    keywords: Optional[List[str]] = None
    max_length: Optional[int] = 10000

class ImageRequest(BaseModel):
    """Request model for image processing."""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class AdsResponse(BaseModel):
    """Response model for ads generation."""
    type: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None

class BrandVoice(BaseModel):
    """Brand voice settings."""
    tone: str = "professional"
    style: str = "conversational"
    personality_traits: List[str] = []
    industry_specific_terms: List[str] = []
    brand_guidelines: Optional[Dict[str, Any]] = None

class AudienceProfile(BaseModel):
    """Audience profile settings."""
    demographics: Dict[str, Any] = {
        "age_range": None,
        "gender": None,
        "location": None,
        "occupation": None,
        "income_level": None
    }
    interests: List[str] = []
    pain_points: List[str] = []
    goals: List[str] = []
    buying_behavior: Optional[Dict[str, Any]] = None
    customer_stage: str = "awareness"

class ContentSource(BaseModel):
    """Content source settings."""
    type: str
    content: str
    priority: int = 1
    relevance_score: Optional[float] = None

class ProjectContext(BaseModel):
    """Project context settings."""
    project_name: str
    project_description: str
    industry: str
    key_messages: List[str] = []
    brand_assets: List[str] = []
    content_sources: List[ContentSource] = []
    custom_variables: Dict[str, Any] = {}

class AdsGenerationRequest(BaseModel):
    """Request model for ads generation."""
    prompt: str
    type: str
    metadata: Optional[Dict[str, Any]] = None
    brand_voice: Optional[BrandVoice] = None
    audience_profile: Optional[AudienceProfile] = None
    project_context: Optional[ProjectContext] = None

class BackgroundRemovalRequest(BaseModel):
    """Request model for background removal."""
    image_url: str
    metadata: Optional[Dict[str, Any]] = None
    image_settings: Optional[Dict[str, Any]] = None
    content_sources: Optional[List[ContentSource]] = None

class EmailSequenceMetrics(BaseModel):
    """Email sequence metrics."""
    sequence_id: str
    total_sent: int
    opens: int
    clicks: int
    conversions: int
    bounces: int
    unsubscribes: int
    revenue: float
    last_updated: str
    engagement_score: float
    delivery_rate: float
    spam_complaints: int
    device_stats: Dict[str, int] = {
        "mobile": 0,
        "desktop": 0,
        "tablet": 0
    }
    location_stats: Dict[str, Any] = {}
    time_stats: Dict[str, Any] = {
        "best_time_to_send": None,
        "average_open_time": None
    }

class EmailSequenceSettings(BaseModel):
    """Email sequence settings."""
    sequence_id: str
    sender_name: str
    reply_to_email: str
    unsubscribe_link: bool = True
    double_opt_in: bool = False
    resend_failed_emails: bool = True
    max_retry_attempts: int = 3
    retry_delay_minutes: int = 30
    custom_tracking_domain: Optional[str] = None
    custom_branding: Optional[Dict[str, Any]] = None
    compliance_settings: Dict[str, Any] = {
        "gdpr_compliant": True,
        "can_spam_compliant": True,
        "privacy_policy_url": None
    }

class AdsAnalyticsRequest(BaseModel):
    """Request model for ads analytics."""
    ads_generation_id: int
    metrics: Dict[str, Any]
    email_metrics: Optional[EmailSequenceMetrics] = None
    email_settings: Optional[EmailSequenceSettings] = None

# Initialize router
router = APIRouter(prefix="/ads", tags=["ads"])
ads_service = AdsService()

# Initialize rembg session
try:
    rembg_session = new_session("u2netp")
    rembg_session_fast = new_session("u2netp")
    rembg_session_std = new_session("u2net")
except ImportError:
    remove = None
    logger.warning("rembg library not installed. Background removal will not be available.")

async def get_website_text(url: str) -> str:
    """Get text content from a website."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            return ""
    except Exception as e:
        logger.error(f"Error fetching website content: {e}")
        return ""

@router.post("/generate")
async def generate_ads(
    request: AdsGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate ads based on prompt."""
    try:
        result = await ads_service.generate_ads(
            prompt=request.prompt,
            type=request.type
        )
        
        # Store in database
        ads_record = await AdsDBService.create_ads_generation(
            user_id=current_user["id"],
            url=result["url"],
            type=request.type,
            content=result,
            prompt=request.prompt,
            metadata=request.metadata,
            brand_voice=request.brand_voice.dict() if request.brand_voice else None,
            audience_profile=request.audience_profile.dict() if request.audience_profile else None,
            project_context=request.project_context.dict() if request.project_context else None
        )
        
        return format_response({
            "id": ads_record.id,
            **result
        })
    except Exception as e:
        raise handle_error(e)

@router.post("/remove-background")
async def remove_background(
    request: BackgroundRemovalRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Remove background from image."""
    try:
        result = await ads_service.remove_background(
            image_url=request.image_url
        )
        
        # Store in database
        bg_removal = await AdsDBService.create_background_removal(
            user_id=current_user["id"],
            processed_image_url=result["processed_url"],
            original_image_url=request.image_url,
            metadata=request.metadata,
            image_settings=request.image_settings,
            content_sources=[cs.dict() for cs in request.content_sources] if request.content_sources else None
        )
        
        return format_response({
            "id": bg_removal.id,
            **result
        })
    except Exception as e:
        raise handle_error(e)

@router.post("/analytics")
async def track_analytics(
    request: AdsAnalyticsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Track analytics for an ads generation."""
    try:
        # Verify ads generation exists and belongs to user
        ads = await AdsDBService.get_ads_generation(
            user_id=current_user["id"],
            ads_id=request.ads_generation_id
        )
        if not ads:
            raise HTTPException(status_code=404, detail="Ads generation not found")
        
        # Store analytics
        analytics = await AdsDBService.create_ads_analytics(
            user_id=current_user["id"],
            ads_generation_id=request.ads_generation_id,
            metrics=request.metrics,
            email_metrics=request.email_metrics.dict() if request.email_metrics else None,
            email_settings=request.email_settings.dict() if request.email_settings else None
        )
        
        return format_response({
            "id": analytics.id,
            "ads_generation_id": analytics.ads_generation_id,
            "metrics": analytics.metrics,
            "email_metrics": analytics.email_metrics,
            "email_settings": analytics.email_settings
        })
    except Exception as e:
        raise handle_error(e)

@router.get("/list")
async def list_ads(
    type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List ads generations for current user."""
    try:
        ads_list = await AdsDBService.list_ads_generations(
            user_id=current_user["id"],
            type=type,
            limit=limit,
            offset=offset
        )
        
        return format_response({
            "items": [ads.to_dict() for ads in ads_list],
            "total": len(ads_list)
        })
    except Exception as e:
        raise handle_error(e)

@router.get("/{ads_id}")
async def get_ads(
    ads_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific ads generation."""
    try:
        ads = await AdsDBService.get_ads_generation(
            user_id=current_user["id"],
            ads_id=ads_id
        )
        if not ads:
            raise HTTPException(status_code=404, detail="Ads generation not found")
        
        return format_response(ads.to_dict())
    except Exception as e:
        raise handle_error(e)

@router.delete("/{ads_id}")
async def delete_ads(
    ads_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Soft delete an ads generation."""
    try:
        success = await AdsDBService.soft_delete_ads_generation(
            user_id=current_user["id"],
            ads_id=ads_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Ads generation not found")
        
        return format_response({"message": "Ads generation deleted successfully"})
    except Exception as e:
        raise handle_error(e)

@router.get("/background-removals")
async def list_background_removals(
    limit: int = 100,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List background removals for current user."""
    try:
        removals = await AdsDBService.list_background_removals(
            user_id=current_user["id"],
            limit=limit,
            offset=offset
        )
        
        return format_response({
            "items": [removal.to_dict() for removal in removals],
            "total": len(removals)
        })
    except Exception as e:
        raise handle_error(e)

@router.get("/background-removals/{removal_id}")
async def get_background_removal(
    removal_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific background removal."""
    try:
        removal = await AdsDBService.get_background_removal(
            user_id=current_user["id"],
            removal_id=removal_id
        )
        if not removal:
            raise HTTPException(status_code=404, detail="Background removal not found")
        
        return format_response(removal.to_dict())
    except Exception as e:
        raise handle_error(e)

@router.delete("/background-removals/{removal_id}")
async def delete_background_removal(
    removal_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Soft delete a background removal."""
    try:
        success = await AdsDBService.soft_delete_background_removal(
            user_id=current_user["id"],
            removal_id=removal_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Background removal not found")
        
        return format_response({"message": "Background removal deleted successfully"})
    except Exception as e:
        raise handle_error(e)

@router.get("/analytics")
async def list_analytics(
    ads_generation_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List analytics for current user."""
    try:
        analytics = await AdsDBService.list_ads_analytics(
            user_id=current_user["id"],
            ads_generation_id=ads_generation_id,
            limit=limit,
            offset=offset
        )
        
        return format_response({
            "items": [analytic.to_dict() for analytic in analytics],
            "total": len(analytics)
        })
    except Exception as e:
        raise handle_error(e)

@router.get("/analytics/{analytics_id}")
async def get_analytics(
    analytics_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific analytics record."""
    try:
        analytics = await AdsDBService.get_ads_analytics(
            user_id=current_user["id"],
            analytics_id=analytics_id
        )
        if not analytics:
            raise HTTPException(status_code=404, detail="Analytics record not found")
        
        return format_response(analytics.to_dict())
    except Exception as e:
        raise handle_error(e)

@router.post("/stream")
async def stream_ads(
    request: AdsRequest,
    http_request: httpx.Request = Depends(check_router_auth)
):
    """
    Stream ads generation in real-time.
    """
    try:
        website_text = await get_website_text(request.url)
        if website_text:
            website_text = website_text[:800]  # Limit to 800 characters for speed

        # Import streaming function here to avoid circular imports

        async def content_generator():
            
    """content_generator function."""
async for content in generate_ads_lcel_streaming_parallel(website_text or "", n_ads=1):
                yield content

        return StreamingResponse(
            content_generator(),
            media_type="text/event-stream"
        )

    except BasicAuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.exception("Error in streaming ads")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) 