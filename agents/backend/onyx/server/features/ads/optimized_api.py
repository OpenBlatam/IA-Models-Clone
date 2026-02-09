from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import List, Optional, Dict, Any, BackgroundTasks
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import asyncio
import time
import json
from datetime import datetime, timedelta
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from contextlib import asynccontextmanager
from onyx.server.auth_check import check_router_auth
from onyx.server.utils import BasicAuthenticationError
from onyx.utils.logger import setup_logger
from onyx.core.auth import get_current_user
from onyx.core.functions import format_response, handle_error
from onyx.server.features.ads.optimized_db_service import OptimizedAdsDBService
from onyx.server.features.ads.optimized_service import OptimizedAdsService
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.tokenization_api import router as tokenization_router
from onyx.server.features.ads.diffusion_api import router as diffusion_router
from typing import Any, List, Dict, Optional
import logging
"""
Optimized API for ads generation and management with production-ready features.
"""


logger = setup_logger()

# Request Models
class OptimizedAdsRequest(BaseModel):
    """Optimized request model for ads generation."""
    prompt: str = Field(..., min_length=10, max_length=2000)
    type: str = Field(..., regex="^(ads|brand-kit|custom)$")
    target_audience: Optional[str] = Field(None, max_length=500)
    context: Optional[str] = Field(None, max_length=1000)
    keywords: Optional[List[str]] = Field(None, max_items=20)
    max_length: Optional[int] = Field(10000, ge=100, le=50000)
    use_cache: bool = Field(True, description="Whether to use caching")
    priority: str = Field("normal", regex="^(low|normal|high)$")

class OptimizedImageRequest(BaseModel):
    """Optimized request model for image processing."""
    image_url: Optional[str] = Field(None, max_length=2048)
    image_base64: Optional[str] = Field(None, max_length=10485760)  # 10MB
    max_size: Optional[int] = Field(1024, ge=256, le=4096)
    quality: Optional[int] = Field(85, ge=10, le=100)
    format: Optional[str] = Field("auto", regex="^(auto|png|jpeg)$")

class OptimizedAnalyticsRequest(BaseModel):
    """Optimized request model for analytics tracking."""
    ads_generation_id: int = Field(..., gt=0)
    metrics: Dict[str, Any] = Field(..., max_items=50)
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = Field(None, max_length=100)

# Response Models
class OptimizedAdsResponse(BaseModel):
    """Optimized response model for ads generation."""
    id: str
    type: str
    content: Any
    metadata: Dict[str, Any]
    generation_time: float
    cached: bool = False
    created_at: datetime

class OptimizedImageResponse(BaseModel):
    """Optimized response model for image processing."""
    id: str
    original_url: Optional[str]
    processed_url: str
    mime_type: str
    file_size: int
    processing_time: float
    cached: bool = False
    created_at: datetime

class OptimizedAnalyticsResponse(BaseModel):
    """Optimized response model for analytics."""
    id: str
    ads_generation_id: int
    metrics: Dict[str, Any]
    timestamp: datetime
    created_at: datetime

# Initialize services
router = APIRouter(prefix="/ads/v2", tags=["ads-optimized"])
ads_service = OptimizedAdsService()
db_service = OptimizedAdsDBService()

# Include tokenization router
router.include_router(tokenization_router)

# Include diffusion router
router.include_router(diffusion_router)

# Rate limiting middleware
class RateLimiter:
    def __init__(self) -> Any:
        self._redis_client = None
    
    @property
    async def redis_client(self) -> Any:
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    async def check_rate_limit(self, user_id: int, operation: str) -> bool:
        """Check if user has exceeded rate limit for operation."""
        redis = await self.redis_client
        key = f"rate_limit:{user_id}:{operation}"
        
        current = await redis.get(key)
        limit = settings.rate_limits.get(operation, 100)
        
        if current and int(current) >= limit:
            return False
        
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 3600)  # 1 hour window
        await pipe.execute()
        
        return True

rate_limiter = RateLimiter()

# Background task queue
class BackgroundTaskQueue:
    def __init__(self) -> Any:
        self._queue = asyncio.Queue(maxsize=settings.task_queue_size)
        self._workers = []
        self._running = False
    
    async def start_workers(self) -> Any:
        """Start background task workers."""
        self._running = True
        for _ in range(settings.background_task_workers):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)
    
    async def stop_workers(self) -> Any:
        """Stop background task workers."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
    
    async def _worker(self) -> Any:
        """Background task worker."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_task(task)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background task error: {e}")
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a background task."""
        task_type = task.get("type")
        
        if task_type == "analytics_batch":
            await self._process_analytics_batch(task["data"])
        elif task_type == "file_cleanup":
            await self._cleanup_temp_files()
        elif task_type == "cache_cleanup":
            await self._cleanup_cache()
    
    async def _process_analytics_batch(self, analytics_data: List[Dict[str, Any]]):
        """Process analytics data in batch."""
        try:
            for data in analytics_data:
                await db_service.create_ads_analytics(**data)
        except Exception as e:
            logger.error(f"Analytics batch processing error: {e}")
    
    async def _cleanup_temp_files(self) -> Any:
        """Clean up temporary files."""
        # Implementation for temp file cleanup
        pass
    
    async def _cleanup_cache(self) -> Any:
        """Clean up expired cache entries."""
        try:
            redis = await rate_limiter.redis_client
            # Implementation for cache cleanup
            pass
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def add_task(self, task: Dict[str, Any]):
        """Add task to background queue."""
        try:
            await self._queue.put(task)
        except asyncio.QueueFull:
            logger.warning("Background task queue is full")

task_queue = BackgroundTaskQueue()

# Startup and shutdown events
@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await task_queue.start_workers()
    logger.info("Optimized ads API started")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown."""
    await task_queue.stop_workers()
    await ads_service.cleanup()
    await db_service.close()
    logger.info("Optimized ads API stopped")

# API Endpoints
@router.post("/generate", response_model=OptimizedAdsResponse)
async def generate_ads_optimized(
    request: OptimizedAdsRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate ads with optimized performance and caching."""
    try:
        # Rate limiting
        if not await rate_limiter.check_rate_limit(current_user["id"], "ads_generation"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Generate ads
        start_time = time.time()
        result = await ads_service.generate_ads(
            prompt=request.prompt,
            type=request.type,
            user_id=current_user["id"],
            use_cache=request.use_cache
        )
        
        # Store in database
        ads_record = await db_service.create_ads_generation(
            user_id=current_user["id"],
            url=result.get("url"),
            type=request.type,
            content=result,
            prompt=request.prompt,
            metadata={
                "target_audience": request.target_audience,
                "context": request.context,
                "keywords": request.keywords,
                "max_length": request.max_length,
                "priority": request.priority,
                "generation_time": time.time() - start_time
            }
        )
        
        # Add analytics task to background queue
        background_tasks.add_task(
            task_queue.add_task({
                "type": "analytics_batch",
                "data": [{
                    "user_id": current_user["id"],
                    "ads_generation_id": ads_record.id,
                    "metrics": {
                        "generation_time": time.time() - start_time,
                        "prompt_length": len(request.prompt),
                        "type": request.type,
                        "priority": request.priority
                    }
                }]
            })
        )
        
        return OptimizedAdsResponse(
            id=str(ads_record.id),
            type=result["type"],
            content=result["content"],
            metadata=result.get("metadata", {}),
            generation_time=result.get("generation_time", 0),
            cached=result.get("cached", False),
            created_at=ads_record.created_at
        )
        
    except Exception as e:
        logger.exception("Error generating ads")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove-background", response_model=OptimizedImageResponse)
async def remove_background_optimized(
    request: OptimizedImageRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Remove background from image with optimized processing."""
    try:
        # Rate limiting
        if not await rate_limiter.check_rate_limit(current_user["id"], "background_removal"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Process image
        start_time = time.time()
        result = await ads_service.remove_background(
            image_url=request.image_url,
            user_id=current_user["id"],
            max_size=request.max_size
        )
        
        # Store in database
        bg_record = await db_service.create_background_removal(
            user_id=current_user["id"],
            processed_image_url=result["processed_url"],
            original_image_url=result["original_url"],
            metadata={
                "max_size": request.max_size,
                "quality": request.quality,
                "format": request.format,
                "processing_time": time.time() - start_time
            }
        )
        
        return OptimizedImageResponse(
            id=str(bg_record.id),
            original_url=result["original_url"],
            processed_url=result["processed_url"],
            mime_type=result["mime"],
            file_size=result.get("file_size", 0),
            processing_time=result.get("processing_time", 0),
            cached=result.get("cached", False),
            created_at=bg_record.created_at
        )
        
    except Exception as e:
        logger.exception("Error removing background")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics", response_model=OptimizedAnalyticsResponse)
async def track_analytics_optimized(
    request: OptimizedAnalyticsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Track analytics with optimized batching."""
    try:
        # Rate limiting
        if not await rate_limiter.check_rate_limit(current_user["id"], "analytics_tracking"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Create analytics record
        analytics_record = await db_service.create_ads_analytics(
            user_id=current_user["id"],
            ads_generation_id=request.ads_generation_id,
            metrics=request.metrics
        )
        
        return OptimizedAnalyticsResponse(
            id=str(analytics_record.id),
            ads_generation_id=analytics_record.ads_generation_id,
            metrics=analytics_record.metrics,
            timestamp=request.timestamp or datetime.utcnow(),
            created_at=analytics_record.created_at
        )
        
    except Exception as e:
        logger.exception("Error tracking analytics")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_ads_optimized(
    type: Optional[str] = None,
    limit: int = Field(100, ge=1, le=1000),
    offset: int = Field(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List ads with optimized pagination and caching."""
    try:
        ads_list = await db_service.list_ads_generations(
            user_id=current_user["id"],
            type=type,
            limit=limit,
            offset=offset
        )
        
        return {
            "ads": [
                {
                    "id": str(ad.id),
                    "type": ad.type,
                    "prompt": ad.prompt,
                    "created_at": ad.created_at,
                    "metadata": ad.metadata
                }
                for ad in ads_list
            ],
            "total": len(ads_list),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.exception("Error listing ads")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_user_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user statistics with caching."""
    try:
        stats = await db_service.get_user_stats(current_user["id"])
        return stats
        
    except Exception as e:
        logger.exception("Error getting user stats")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{ads_id}")
async def delete_ads_optimized(
    ads_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete ads with cache invalidation."""
    try:
        success = await db_service.soft_delete_ads_generation(
            user_id=current_user["id"],
            ads_id=ads_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Ads not found")
        
        return {"message": "Ads deleted successfully"}
        
    except Exception as e:
        logger.exception("Error deleting ads")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        await db_service.get_user_stats(1)
        
        # Check Redis connection
        redis = await rate_limiter.redis_client
        await redis.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "connected",
                "redis": "connected",
                "background_tasks": "running"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy") 