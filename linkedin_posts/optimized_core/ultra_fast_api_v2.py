from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, Query
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import orjson
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from loguru import logger
from .ultra_fast_engine_v2 import UltraFastEngineV2, get_ultra_fast_engine_v2, ultra_fast_cache_v2, profile_performance_v2
import aiofiles
from aiofiles import open as aio_open
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import asyncio_mqtt as mqtt
import aiokafka
from aio_pika import connect_robust, Message
import orjson as fast_json
                import uuid
                    import uuid
from typing import Any, List, Dict, Optional
import logging
"""
Ultra Fast API V2 - LinkedIn Posts
=================================

API ultra optimizada V2 con las mejores librerías para máxima performance.
"""


# Ultra fast imports - Latest versions

# Pydantic models - Latest version

# Monitoring and metrics - Advanced

# Import our ultra fast engine V2

# Advanced imports


# Enhanced Pydantic Models V2
class LinkedInPostCreateV2(BaseModel):
    """Ultra fast post creation model V2."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    content: str = Field(..., min_length=10, max_length=3000, description="Post content")
    post_type: str = Field(..., regex='^(announcement|educational|update|promotional|story|question)$')
    tone: str = Field(..., regex='^(professional|casual|friendly|formal|enthusiastic|thoughtful)$')
    target_audience: str = Field(..., min_length=3, max_length=100)
    industry: str = Field(..., min_length=3, max_length=50)
    tags: Optional[List[str]] = Field(default_factory=list, max_items=15)
    language: str = Field(default="en", regex='^[a-z]{2}$')
    priority: str = Field(default="normal", regex='^(low|normal|high|urgent)$')
    
    @validator('content')
    def validate_content(cls, v) -> bool:
        if len(v.strip()) < 10:
            raise ValueError('Content must be at least 10 characters')
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        if v:
            # Remove duplicates and normalize
            return list(set([tag.strip().lower() for tag in v if tag.strip()]))
        return v


class LinkedInPostUpdateV2(BaseModel):
    """Ultra fast post update model V2."""
    model_config = ConfigDict(extra='forbid')
    
    content: Optional[str] = Field(None, min_length=10, max_length=3000)
    post_type: Optional[str] = Field(None, regex='^(announcement|educational|update|promotional|story|question)$')
    tone: Optional[str] = Field(None, regex='^(professional|casual|friendly|formal|enthusiastic|thoughtful)$')
    target_audience: Optional[str] = Field(None, min_length=3, max_length=100)
    industry: Optional[str] = Field(None, min_length=3, max_length=50)
    tags: Optional[List[str]] = Field(None, max_items=15)
    language: Optional[str] = Field(None, regex='^[a-z]{2}$')
    priority: Optional[str] = Field(None, regex='^(low|normal|high|urgent)$')


class LinkedInPostResponseV2(BaseModel):
    """Ultra fast post response model V2."""
    id: str
    content: str
    post_type: str
    tone: str
    target_audience: str
    industry: str
    tags: List[str]
    language: str
    priority: str
    created_at: str
    updated_at: str
    nlp_analysis: Optional[Dict[str, Any]] = None
    engagement_metrics: Optional[Dict[str, Any]] = None
    optimization_score: Optional[float] = None
    ai_recommendations: Optional[List[str]] = None


class OptimizationRequestV2(BaseModel):
    """Ultra fast optimization request model V2."""
    post_id: str
    optimization_type: str = Field(..., regex='^(content|tone|keywords|readability|engagement|comprehensive)$')
    optimization_level: str = Field(default="standard", regex='^(basic|standard|advanced|expert)$')
    include_ai_suggestions: bool = True
    preserve_original: bool = True


class OptimizationResponseV2(BaseModel):
    """Ultra fast optimization response model V2."""
    post_id: str
    original_content: str
    optimized_content: str
    optimization_score: float
    improvement_percentage: float
    suggestions: List[str]
    processing_time: float
    confidence_score: float
    optimization_details: Dict[str, Any]


class BatchCreateRequestV2(BaseModel):
    """Ultra fast batch creation request model V2."""
    posts: List[LinkedInPostCreateV2] = Field(..., max_items=200)
    processing_strategy: str = Field(default="parallel", regex='^(sequential|parallel|hybrid)$')
    priority_level: str = Field(default="normal", regex='^(low|normal|high|urgent)$')


class BatchCreateResponseV2(BaseModel):
    """Ultra fast batch creation response model V2."""
    created_posts: List[LinkedInPostResponseV2]
    failed_posts: List[Dict[str, Any]]
    total_processing_time: float
    success_rate: float
    performance_metrics: Dict[str, Any]


class AnalyticsRequestV2(BaseModel):
    """Advanced analytics request model V2."""
    post_id: str
    include_competitor_analysis: bool = False
    include_audience_insights: bool = True
    include_virality_prediction: bool = True
    include_sentiment_analysis: bool = True
    include_topic_modeling: bool = False
    include_complexity_analysis: bool = True


class AnalyticsResponseV2(BaseModel):
    """Advanced analytics response model V2."""
    post_id: str
    analytics: Dict[str, Any]
    processing_time: float
    confidence_score: float
    recommendations: List[str]
    insights: Dict[str, Any]


# Enhanced Metrics V2
class UltraFastMetricsV2:
    """Ultra fast API metrics V2."""
    
    def __init__(self) -> Any:
        # Request metrics - Enhanced
        self.request_counter = Counter('api_v2_requests_total', 'Total API V2 requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('api_v2_request_duration_seconds', 'API V2 request duration', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
        self.active_requests = Gauge('api_v2_active_requests', 'Active API V2 requests')
        
        # Business metrics - Enhanced
        self.posts_created = Counter('posts_v2_created_total', 'Total posts created V2')
        self.posts_updated = Counter('posts_v2_updated_total', 'Total posts updated V2')
        self.posts_deleted = Counter('posts_v2_deleted_total', 'Total posts deleted V2')
        self.optimizations_performed = Counter('optimizations_v2_performed_total', 'Total optimizations performed V2')
        self.analytics_processed = Counter('analytics_v2_processed_total', 'Total analytics processed V2')
        
        # Error metrics - Enhanced
        self.error_counter = Counter('api_v2_errors_total', 'Total API V2 errors', ['endpoint', 'error_type', 'severity'])
        
        # Performance metrics - Enhanced
        self.cache_hits = Counter('cache_v2_hits_total', 'Total cache hits V2')
        self.cache_misses = Counter('cache_v2_misses_total', 'Total cache misses V2')
        self.database_queries = Counter('database_v2_queries_total', 'Total database queries V2')
        self.nlp_processing_time = Histogram('nlp_v2_processing_duration_seconds', 'NLP processing duration V2')
        
        # System metrics - Enhanced
        self.memory_usage = Gauge('memory_v2_usage_bytes', 'Memory usage in bytes V2')
        self.cpu_usage = Gauge('cpu_v2_usage_percent', 'CPU usage percentage V2')
        self.cache_hit_rate = Gauge('cache_v2_hit_rate', 'Cache hit rate V2')
        self.response_time_p95 = Histogram('response_time_v2_p95_seconds', '95th percentile response time V2')
        self.response_time_p99 = Histogram('response_time_v2_p99_seconds', '99th percentile response time V2')


# Enhanced Middleware V2
class UltraFastMiddlewareV2(BaseHTTPMiddleware):
    """Ultra fast middleware for request processing V2."""
    
    def __init__(self, app, metrics: UltraFastMetricsV2):
        
    """__init__ function."""
super().__init__(app)
        self.metrics = metrics
    
    async def dispatch(self, request: Request, call_next):
        """Process request with ultra fast middleware V2."""
        start_time = time.time()
        
        # Increment active requests
        self.metrics.active_requests.inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.request_counter.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            self.metrics.request_duration.observe(duration)
            
            # Record percentile metrics
            if duration > 0.1:
                self.metrics.response_time_p95.observe(duration)
            if duration > 0.5:
                self.metrics.response_time_p99.observe(duration)
            
            # Add enhanced performance headers
            response.headers["X-Processing-Time"] = str(duration)
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "")
            response.headers["X-API-Version"] = "2.0-ultra-fast"
            response.headers["X-Features"] = "analytics,ai-testing,optimization,streaming"
            response.headers["X-Cache-Status"] = request.headers.get("X-Cache-Status", "miss")
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.metrics.error_counter.labels(
                endpoint=request.url.path,
                error_type=type(e).__name__,
                severity="high" if "500" in str(e) else "medium"
            ).inc()
            raise
        finally:
            # Decrement active requests
            self.metrics.active_requests.dec()


# Ultra Fast API V2
class UltraFastAPIV2:
    """Ultra fast FastAPI application V2."""
    
    def __init__(self) -> Any:
        self.app = FastAPI(
            title="LinkedIn Posts Ultra Fast API V2",
            description="Ultra optimized LinkedIn Posts management system V2 with advanced features",
            version="2.0-ultra-fast",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        self.metrics = UltraFastMetricsV2()
        self.engine = None
        
        self._setup_enhanced_middleware()
        self._setup_enhanced_routes()
        self._setup_enhanced_events()
    
    def _setup_enhanced_middleware(self) -> Any:
        """Setup ultra fast middleware V2."""
        # Add enhanced middleware
        self.app.add_middleware(UltraFastMiddlewareV2, metrics=self.metrics)
        
        # CORS middleware with enhanced configuration
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Processing-Time", "X-Request-ID", "X-API-Version", "X-Features", "X-Cache-Status"]
        )
        
        # Enhanced Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=500)
        
        # Enhanced trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
    
    def _setup_enhanced_routes(self) -> Any:
        """Setup ultra fast API routes V2."""
        
        @self.app.get("/health/v2", response_class=ORJSONResponse)
        async def enhanced_health_check_v2():
            """Ultra fast health check V2."""
            engine = await get_ultra_fast_engine_v2()
            return await engine.health_check()
        
        @self.app.get("/metrics/v2")
        async def get_metrics_v2():
            """Get Prometheus metrics V2."""
            engine = await get_ultra_fast_engine_v2()
            return Response(
                content=await engine.get_metrics(),
                media_type="text/plain"
            )
        
        @self.app.post("/posts/v2", response_model=LinkedInPostResponseV2, response_class=ORJSONResponse)
        @profile_performance_v2
        async def create_post_v2(
            post: LinkedInPostCreateV2,
            background_tasks: BackgroundTasks,
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Create a new LinkedIn post with ultra fast processing V2."""
            try:
                # Generate unique ID
                post_id = str(uuid.uuid4())
                
                # Prepare post data
                post_data = {
                    "id": post_id,
                    "content": post.content,
                    "post_type": post.post_type,
                    "tone": post.tone,
                    "target_audience": post.target_audience,
                    "industry": post.industry,
                    "tags": post.tags,
                    "language": post.language,
                    "priority": post.priority
                }
                
                # Create post with ultra fast processing
                result = await engine.create_post_ultra_fast_v2(post_data)
                
                # Record metrics
                self.metrics.posts_created.inc()
                
                # Add background task for additional processing
                background_tasks.add_task(self._process_post_analytics_v2, post_id)
                
                return LinkedInPostResponseV2(
                    id=post_id,
                    content=post.content,
                    post_type=post.post_type,
                    tone=post.tone,
                    target_audience=post.target_audience,
                    industry=post.industry,
                    tags=post.tags,
                    language=post.language,
                    priority=post.priority,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    nlp_analysis=result.get('nlp_analysis')
                )
                
            except Exception as e:
                logger.error(f"Post creation error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/posts/v2/{post_id}", response_model=LinkedInPostResponseV2, response_class=ORJSONResponse)
        @ultra_fast_cache_v2(ttl=300)  # Cache for 5 minutes
        async def get_post_v2(
            post_id: str,
            include_analytics: bool = Query(False, description="Include AI analytics"),
            include_optimization: bool = Query(False, description="Include optimization suggestions"),
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Get a LinkedIn post with ultra fast caching V2."""
            try:
                post = await engine.get_post_ultra_fast_v2(post_id)
                if not post:
                    raise HTTPException(status_code=404, detail="Post not found")
                
                response = LinkedInPostResponseV2(**post)
                
                # Add analytics if requested
                if include_analytics:
                    analytics = await engine.get_post_analytics_v2(post_id)
                    response.nlp_analysis = analytics
                
                # Add optimization suggestions if requested
                if include_optimization:
                    optimization = await engine.get_optimization_suggestions_v2(post_id)
                    response.optimization_score = optimization.get('score', 0.0)
                    response.ai_recommendations = optimization.get('recommendations', [])
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Post retrieval error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/posts/v2", response_model=List[LinkedInPostResponseV2], response_class=ORJSONResponse)
        @ultra_fast_cache_v2(ttl=60)  # Cache for 1 minute
        async def list_posts_v2(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0),
            post_type: Optional[str] = Query(None, regex='^(announcement|educational|update|promotional|story|question)$'),
            industry: Optional[str] = Query(None),
            priority: Optional[str] = Query(None, regex='^(low|normal|high|urgent)$'),
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """List LinkedIn posts with ultra fast pagination V2."""
            try:
                # Build query with filters
                query = "SELECT * FROM linkedin_posts"
                params = {}
                conditions = []
                
                if post_type:
                    conditions.append("post_type = :post_type")
                    params['post_type'] = post_type
                
                if industry:
                    conditions.append("industry = :industry")
                    params['industry'] = industry
                
                if priority:
                    conditions.append("priority = :priority")
                    params['priority'] = priority
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                params['limit'] = limit
                params['offset'] = offset
                
                # Execute query
                posts = await engine.database.execute_query(query, params)
                
                # Record metrics
                self.metrics.database_queries.inc()
                
                return [LinkedInPostResponseV2(**post) for post in posts]
                
            except Exception as e:
                logger.error(f"Post listing error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/posts/v2/{post_id}", response_model=LinkedInPostResponseV2, response_class=ORJSONResponse)
        @profile_performance_v2
        async def update_post_v2(
            post_id: str,
            post_update: LinkedInPostUpdateV2,
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Update a LinkedIn post with ultra fast processing V2."""
            try:
                # Get existing post
                existing_post = await engine.get_post_ultra_fast_v2(post_id)
                if not existing_post:
                    raise HTTPException(status_code=404, detail="Post not found")
                
                # Update fields
                update_data = post_update.dict(exclude_unset=True)
                update_data['id'] = post_id
                update_data['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Build update query
                set_clauses = []
                params = {'post_id': post_id}
                
                for field, value in update_data.items():
                    if field != 'id':
                        set_clauses.append(f"{field} = :{field}")
                        params[field] = value
                
                if set_clauses:
                    query = f"UPDATE linkedin_posts SET {', '.join(set_clauses)} WHERE id = :post_id"
                    await engine.database.execute_query(query, params)
                    
                    # Clear cache
                    await engine.cache.delete(f"post:{post_id}")
                    
                    # Record metrics
                    self.metrics.posts_updated.inc()
                
                # Return updated post
                updated_post = await engine.get_post_ultra_fast_v2(post_id)
                return LinkedInPostResponseV2(**updated_post)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Post update error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/posts/v2/{post_id}", response_class=ORJSONResponse)
        async def delete_post_v2(
            post_id: str,
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Delete a LinkedIn post with ultra fast processing V2."""
            try:
                # Check if post exists
                existing_post = await engine.get_post_ultra_fast_v2(post_id)
                if not existing_post:
                    raise HTTPException(status_code=404, detail="Post not found")
                
                # Delete from database
                query = "DELETE FROM linkedin_posts WHERE id = :post_id"
                await engine.database.execute_query(query, {'post_id': post_id})
                
                # Clear cache
                await engine.cache.delete(f"post:{post_id}")
                
                # Record metrics
                self.metrics.posts_deleted.inc()
                
                return {"message": "Post deleted successfully", "post_id": post_id}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Post deletion error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/posts/v2/{post_id}/optimize", response_model=OptimizationResponseV2, response_class=ORJSONResponse)
        @profile_performance_v2
        async def optimize_post_v2(
            post_id: str,
            optimization_request: OptimizationRequestV2,
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Optimize a LinkedIn post with ultra fast NLP processing V2."""
            try:
                start_time = time.time()
                
                # Optimize post
                result = await engine.optimize_post_ultra_fast_v2(post_id, optimization_request)
                
                processing_time = time.time() - start_time
                
                # Record metrics
                self.metrics.optimizations_performed.inc()
                self.metrics.nlp_processing_time.observe(processing_time)
                
                return OptimizationResponseV2(
                    post_id=post_id,
                    original_content=result['original_content'],
                    optimized_content=result.get('optimized_content', result['original_content']),
                    optimization_score=result.get('optimization_score', 0.0),
                    improvement_percentage=result.get('improvement_percentage', 0.0),
                    suggestions=result.get('optimization_suggestions', []),
                    processing_time=processing_time,
                    confidence_score=result.get('confidence_score', 0.85),
                    optimization_details=result.get('details', {})
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Post optimization error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/posts/v2/batch", response_model=BatchCreateResponseV2, response_class=ORJSONResponse)
        @profile_performance_v2
        async def batch_create_posts_v2(
            batch_request: BatchCreateRequestV2,
            engine: UltraFastEngineV2 = Depends(get_ultra_fast_engine_v2)
        ):
            """Create multiple posts with ultra fast batch processing V2."""
            try:
                start_time = time.time()
                
                # Prepare posts data
                posts_data = []
                for post in batch_request.posts:
                    post_data = {
                        "id": str(uuid.uuid4()),
                        "content": post.content,
                        "post_type": post.post_type,
                        "tone": post.tone,
                        "target_audience": post.target_audience,
                        "industry": post.industry,
                        "tags": post.tags,
                        "language": post.language,
                        "priority": post.priority
                    }
                    posts_data.append(post_data)
                
                # Process posts based on strategy
                if batch_request.processing_strategy == "parallel":
                    results = await engine.batch_process_posts_v2(posts_data)
                elif batch_request.processing_strategy == "sequential":
                    results = []
                    for post_data in posts_data:
                        result = await engine.create_post_ultra_fast_v2(post_data)
                        results.append(result)
                else:  # hybrid
                    # Process in chunks
                    chunk_size = 10
                    results = []
                    for i in range(0, len(posts_data), chunk_size):
                        chunk = posts_data[i:i + chunk_size]
                        chunk_results = await engine.batch_process_posts_v2(chunk)
                        results.extend(chunk_results)
                
                # Separate successful and failed posts
                created_posts = []
                failed_posts = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_posts.append({
                            "index": i,
                            "error": str(result),
                            "post_data": posts_data[i]
                        })
                    else:
                        created_posts.append(LinkedInPostResponseV2(
                            id=posts_data[i]["id"],
                            content=posts_data[i]["content"],
                            post_type=posts_data[i]["post_type"],
                            tone=posts_data[i]["tone"],
                            target_audience=posts_data[i]["target_audience"],
                            industry=posts_data[i]["industry"],
                            tags=posts_data[i]["tags"],
                            language=posts_data[i]["language"],
                            priority=posts_data[i]["priority"],
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            nlp_analysis=result.get('nlp_analysis')
                        ))
                
                total_processing_time = time.time() - start_time
                success_rate = len(created_posts) / len(posts_data) if posts_data else 0
                
                return BatchCreateResponseV2(
                    created_posts=created_posts,
                    failed_posts=failed_posts,
                    total_processing_time=total_processing_time,
                    success_rate=success_rate,
                    performance_metrics={
                        "posts_per_second": len(created_posts) / total_processing_time if total_processing_time > 0 else 0,
                        "processing_strategy": batch_request.processing_strategy,
                        "priority_level": batch_request.priority_level
                    }
                )
                
            except Exception as e:
                logger.error(f"Batch post creation error V2: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_enhanced_events(self) -> Any:
        """Setup enhanced startup and shutdown events V2."""
        
        @self.app.on_event("startup")
        async def enhanced_startup_event_v2():
            """Initialize ultra fast engine V2 on startup."""
            logger.info("🚀 Starting Ultra Fast LinkedIn Posts API V2")
            self.engine = await get_ultra_fast_engine_v2()
            logger.info("✅ Ultra Fast Engine V2 initialized")
        
        @self.app.on_event("shutdown")
        async def enhanced_shutdown_event_v2():
            """Cleanup on shutdown V2."""
            logger.info("🛑 Shutting down Ultra Fast LinkedIn Posts API V2")
    
    async def _process_post_analytics_v2(self, post_id: str):
        """Background task for post analytics processing V2."""
        try:
            engine = await get_ultra_fast_engine_v2()
            # Process analytics in background
            await engine.process_post_analytics_v2(post_id)
        except Exception as e:
            logger.error(f"Background analytics error V2: {e}")


# Create FastAPI app instance V2
ultra_fast_api_v2 = UltraFastAPIV2()
app = ultra_fast_api_v2.app


# Run with ultra fast settings V2
if __name__ == "__main__":
    uvicorn.run(
        "ultra_fast_api_v2:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for now, can be scaled with load balancer
        loop="asyncio",
        http="httptools",
        ws="websockets",
        log_level="info",
        access_log=True
    ) 