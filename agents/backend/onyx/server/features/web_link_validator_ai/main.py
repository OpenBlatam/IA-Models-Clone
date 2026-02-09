"""Web Link Validator AI - FastAPI Application"""
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import uvicorn

from config import settings
from services.link_validator import LinkValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL protocols
PROTOCOL_HTTP = "http://"
PROTOCOL_HTTPS = "https://"

# API endpoints
ENDPOINT_VALIDATE = "/validate"
ENDPOINT_VALIDATE_BATCH = "/validate/batch"
ENDPOINT_CHECK = "/check/{url}"
ENDPOINT_HEALTH = "/health"
ENDPOINT_CACHE_CLEAR = "/cache/clear"
ENDPOINT_CACHE_STATS = "/cache/stats"
ENDPOINT_ROOT = "/"


def normalize_query(query: Optional[str]) -> Optional[str]:
    """Normalize query string"""
    if not query:
        return None
    normalized = query.strip()
    return normalized if normalized else None

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered web link validator that checks if links are real and relevant",
    version=settings.app_version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=[CORS_ALLOW_ALL_METHODS],
    allow_headers=[CORS_ALLOW_ALL_HEADERS],
)

# Initialize validator
validator = LinkValidator()


class LinkValidationRequest(BaseModel):
    url: HttpUrl
    query: Optional[str] = None


# Batch validation limits
BATCH_MIN_URLS = 1
BATCH_MAX_URLS = 10

# CORS settings
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_METHODS = "*"
CORS_ALLOW_ALL_HEADERS = "*"

# Response messages
RESPONSE_STATUS_HEALTHY = "healthy"
RESPONSE_STATUS_CACHE_CLEARED = "cache cleared"

class MultipleLinksValidationRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_items=BATCH_MIN_URLS, max_items=BATCH_MAX_URLS)
    query: Optional[str] = None


class LinkValidationResponse(BaseModel):
    url: str
    valid: bool
    exists: bool
    relevant: Optional[bool] = None
    relevance_score: Optional[float] = None
    is_legitimate: Optional[bool] = None
    reason: str
    suggestions: List[str] = []
    timestamp: str


@app.get(ENDPOINT_HEALTH)
async def health_check():
    return {
        "status": RESPONSE_STATUS_HEALTHY,
        "service": settings.app_name,
        "version": settings.app_version
    }


@app.post(ENDPOINT_VALIDATE, response_model=LinkValidationResponse)
async def validate_link(request: LinkValidationRequest):
    """Validate a single web link with optional relevance checking"""
    result = await validator.validate_link(str(request.url), normalize_query(request.query))
    return LinkValidationResponse(**result)


@app.post(ENDPOINT_VALIDATE_BATCH, response_model=List[LinkValidationResponse])
async def validate_multiple_links(request: MultipleLinksValidationRequest):
    """Validate multiple links in parallel (max {max_urls})""".format(max_urls=BATCH_MAX_URLS)
    urls = [str(url) for url in request.urls]
    results = await validator.validate_multiple_links(urls, normalize_query(request.query))
    return [LinkValidationResponse(**r) for r in results]


@app.get(ENDPOINT_CHECK)
async def quick_check(url: str):
    """Quick existence check (no AI analysis)"""
    normalized_url = url if url.startswith((PROTOCOL_HTTP, PROTOCOL_HTTPS)) else f"{PROTOCOL_HTTPS}{url}"
    exists, error = await validator.validate_link_exists(normalized_url)
    return {
        "url": normalized_url,
        "exists": exists,
        "error": error,
        "timestamp": LinkValidator._get_timestamp()
    }


@app.post(ENDPOINT_CACHE_CLEAR)
async def clear_cache():
    """Clear validation cache"""
    validator.clear_cache()
    return {"status": RESPONSE_STATUS_CACHE_CLEARED}


@app.get(ENDPOINT_CACHE_STATS)
async def cache_stats():
    """Get cache statistics"""
    return validator.get_cache_stats()


@app.get(ENDPOINT_ROOT)
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "endpoints": [
            ENDPOINT_VALIDATE,
            ENDPOINT_VALIDATE_BATCH,
            ENDPOINT_CHECK,
            ENDPOINT_HEALTH,
            ENDPOINT_CACHE_CLEAR,
            ENDPOINT_CACHE_STATS
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level="info"
    )

