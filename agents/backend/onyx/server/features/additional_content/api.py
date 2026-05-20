from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import zlib
import threading
import mmh3
from typing import Dict, Any

from .models import (
    AdditionalContentRequest,
    AdditionalContentResponse,
    ErrorResponse
)
from .services import AdditionalContentService

router = APIRouter(prefix="/additional-content", tags=["additional-content"])
service = AdditionalContentService()

# Response compression cache
response_cache: Dict[str, bytes] = {}
response_cache_lock = threading.Lock()

@router.post("/generate", response_model=AdditionalContentResponse)
async def generate_additional_content(request: AdditionalContentRequest):
    """Generate additional content like hashtags, CTAs, and links."""
    try:
        # Generate response
        response = await service.generate_additional_content(request)
        
        # Convert to JSON
        response_json = response.model_dump_json()
        
        # Compress response
        compressed = zlib.compress(response_json.encode())
        
        # Cache the compressed response
        cache_key = str(mmh3.hash(response_json))
        with response_cache_lock:
            response_cache[cache_key] = compressed
        
        # Return streaming response
        return StreamingResponse(
            iter([compressed]),
            media_type="application/json",
            headers={
                "Content-Encoding": "gzip",
                "ETag": cache_key,
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/platforms")
async def get_platforms():
    """Get list of supported platforms."""
    return {
        "platforms": [
            "instagram",
            "twitter",
            "linkedin",
            "facebook",
            "tiktok"
        ]
    }

@router.get("/content-types")
async def get_content_types():
    """Get list of supported content types."""
    return {
        "content_types": [
            "post",
            "article",
            "tweet",
            "story",
            "reel"
        ]
    }

@router.get("/cta-types")
async def get_cta_types():
    """Get list of supported CTA types."""
    return {
        "cta_types": [
            "engagement",
            "click",
            "share",
            "follow",
            "comment"
        ]
    }

@router.delete("/cache")
async def clear_cache():
    """Clear all caches."""
    service.clear_cache()
    with response_cache_lock:
        response_cache.clear()
    return {"message": "Cache cleared successfully"} 