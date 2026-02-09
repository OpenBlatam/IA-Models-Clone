from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..dependencies import (
from ..operations import AsyncExternalAPIOperations
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
External API routes for Ultra-Optimized SEO Service v15.

This module contains external API operation endpoints including:
- Fetch page content
- Check URL accessibility
- Batch URL checking
- Fetch robots.txt and sitemaps
- Extract webpage metadata
"""


    get_async_api_operations,
    get_logger
)

# Create router with prefix and tags
router = APIRouter(
    prefix="/api",
    tags=["External API Operations"],
    responses={
        400: {"description": "Bad Request"},
        429: {"description": "Rate Limited"},
        500: {"description": "Internal Server Error"}
    }
)

@router.post("/fetch-content")
async def fetch_page_content_endpoint(
    url: str,
    timeout: int = 30,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Fetch page content using dedicated async external API operations.
    
    Fetches and parses webpage content including HTML, metadata, and structure.
    Supports various content types and implements intelligent retry logic.
    """
    try:
        content_data = await async_api_ops.fetch_page_content(url, timeout)
        return content_data
    except Exception as e:
        logger.error("Failed to fetch page content", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.post("/check-accessibility")
async def check_url_accessibility_endpoint(
    url: str,
    timeout: int = 10,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Check URL accessibility using dedicated async external API operations.
    
    Performs comprehensive accessibility checks including:
    - HTTP status code verification
    - Response time measurement
    - Content availability check
    - Redirect handling
    """
    try:
        accessibility_data = await async_api_ops.check_url_accessibility(url, timeout)
        return accessibility_data
    except Exception as e:
        logger.error("Failed to check URL accessibility", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.post("/batch-check-urls")
async def batch_check_urls_endpoint(
    urls: List[str],
    timeout: int = 10,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Batch check URLs using dedicated async external API operations.
    
    Performs parallel accessibility checks for multiple URLs.
    Implements rate limiting and error handling for large batches.
    """
    try:
        results = await async_api_ops.batch_check_urls(urls, timeout)
        return {"results": results, "total_urls": len(urls)}
    except Exception as e:
        logger.error("Failed to batch check URLs", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.get("/robots-txt/{base_url:path}")
async def fetch_robots_txt_endpoint(
    base_url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Fetch robots.txt using dedicated async external API operations.
    
    Fetches and parses robots.txt file from the specified base URL.
    Returns structured robots.txt data with parsing results.
    """
    try:
        robots_data = await async_api_ops.fetch_robots_txt(base_url)
        return robots_data
    except Exception as e:
        logger.error("Failed to fetch robots.txt", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.get("/sitemap/{sitemap_url:path}")
async def fetch_sitemap_endpoint(
    sitemap_url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Fetch sitemap using dedicated async external API operations.
    
    Fetches and parses XML sitemaps including sitemap index files.
    Supports various sitemap formats and compression.
    """
    try:
        sitemap_data = await async_api_ops.fetch_sitemap(sitemap_url)
        return sitemap_data
    except Exception as e:
        logger.error("Failed to fetch sitemap", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.get("/metadata/{url:path}")
async def fetch_webpage_metadata_endpoint(
    url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Fetch webpage metadata using dedicated async external API operations.
    
    Extracts comprehensive webpage metadata including:
    - Open Graph tags
    - Twitter Card tags
    - Meta description and keywords
    - Structured data (JSON-LD)
    - Canonical URLs
    """
    try:
        metadata = await async_api_ops.fetch_webpage_metadata(url)
        return metadata
    except Exception as e:
        logger.error("Failed to fetch webpage metadata", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@router.post("/social-media-check")
async def check_social_media_apis_endpoint(
    url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger = Depends(get_logger)
):
    """
    Check social media APIs for URL sharing data.
    
    Checks various social media platforms for:
    - Share counts
    - Engagement metrics
    - Social media metadata
    - Platform-specific data
    """
    try:
        social_data = await async_api_ops.check_social_media_apis(url)
        return social_data
    except Exception as e:
        logger.error("Failed to check social media APIs", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed") 