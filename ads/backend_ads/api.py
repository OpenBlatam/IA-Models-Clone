from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from onyx.utils.logger import setup_logger
from onyx.core.functions import format_response
from onyx.server.features.ads.backend_ads.service import BackendAdsService
from onyx.server.features.ads.backend_ads.models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Backend Ads API - Enhanced Onyx Integration
Complete API with advanced Onyx capabilities and model adaptation.
"""
    AdsGenerationRequest,
    BrandKitRequest,
    EmailSequenceRequest,
    BackgroundRemovalRequest,
    ModelConfig
)

logger = setup_logger()
backend_ads_router = APIRouter(prefix="/backend-ads", tags=["Backend Ads"])
service = BackendAdsService()

@backend_ads_router.post("/generate")
async def generate_ads(
    request: AdsGenerationRequest,
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Generate ads with enhanced capabilities."""
    try:
        result = await service.generate_ads(
            url=request.url,
            prompt=request.prompt,
            type=request.type,
            user_id=request.user_id,
            brand_voice=request.brand_voice,
            audience_profile=request.audience_profile,
            project_context=request.project_context,
            advanced=request.advanced,
            use_langchain=request.use_langchain,
            model_config=request.model_config
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error generating ads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.post("/brand-kit")
async def generate_brand_kit(
    request: BrandKitRequest,
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Generate brand kit with enhanced capabilities."""
    try:
        result = await service.generate_brand_kit(
            url=request.url,
            prompt=request.prompt,
            user_id=request.user_id,
            brand_voice=request.brand_voice,
            audience_profile=request.audience_profile,
            project_context=request.project_context,
            model_config=request.model_config
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error generating brand kit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.post("/email-sequence")
async def generate_email_sequence(
    request: EmailSequenceRequest,
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Generate email sequence with enhanced capabilities."""
    try:
        result = await service.generate_email_sequence(
            url=request.url,
            prompt=request.prompt,
            user_id=request.user_id,
            brand_voice=request.brand_voice,
            audience_profile=request.audience_profile,
            project_context=request.project_context,
            model_config=request.model_config
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error generating email sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.post("/remove-background")
async def remove_background(
    request: BackgroundRemovalRequest,
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Remove background with enhanced capabilities."""
    try:
        result = await service.remove_background(
            image_url=request.image_url,
            user_id=request.user_id,
            metadata=request.metadata,
            model_config=request.model_config
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error removing background: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.get("/history")
async def get_ads_history(
    user_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Get ads history with enhanced capabilities."""
    try:
        result = await service.get_ads_history(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error getting ads history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.get("/brand-kit/history")
async def get_brand_kit_history(
    user_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: BackendAdsService = Depends()
) -> Dict[str, Any]:
    """Get brand kit history with enhanced capabilities."""
    try:
        result = await service.get_brand_kit_history(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return format_response(result)
    except Exception as e:
        logger.error(f"Error getting brand kit history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@backend_ads_router.post("/stream")
async def stream_ads(
    request: AdsGenerationRequest,
    service: BackendAdsService = Depends()
) -> AsyncGenerator[str, None]:
    """Stream ads generation with enhanced capabilities."""
    try:
        async for chunk in service.stream_ads(
            url=request.url,
            prompt=request.prompt,
            type=request.type,
            advanced=request.advanced,
            use_langchain=request.use_langchain,
            model_config=request.model_config
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming ads: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 