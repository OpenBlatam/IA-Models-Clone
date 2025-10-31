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

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Annotated, TypeVar
from functools import wraps
from contextlib import asynccontextmanager
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from fastapi import (
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from pydantic import (
from pydantic.json_schema import JsonSchemaValue
import pydantic_core
import aiohttp
import aiofiles
import redis.asyncio as redis
from databases import Database
from sqlalchemy import text
import motor.motor_asyncio
from bson import ObjectId
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    import uvicorn
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices Application
Following official FastAPI documentation for:
- Data Models (Pydantic v2 best practices)
- Path Operations (route organization and validation)
- Middleware (custom middleware patterns)
- Dependency Injection (advanced patterns)
- Error Handling (comprehensive error management)
"""


# FastAPI and Pydantic imports
    FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response,
    status, Query, Path, Body, Header, Cookie, Form, File, UploadFile
)
    BaseModel, Field, validator, ConfigDict, field_validator, 
    model_validator, computed_field, BeforeValidator, PlainSerializer
)

# Async dependencies

# Performance monitoring

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC V2 DATA MODELS - Best Practices
# ============================================================================

class PipelineType(str, Enum):
    """Enumeration of available diffusion pipeline types."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"

class ModelType(str, Enum):
    """Enumeration of available model types."""
    STABLE_DIFFUSION_V1_5 = "stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"
    CONTROLNET_CANNY = "controlnet-canny"
    CONTROLNET_DEPTH = "controlnet-depth"

class DiffusionRequest(BaseModel):
    """
    Request model for single image generation.
    Following Pydantic v2 best practices with field validation and computed fields.
    """
    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        str_strip_whitespace=True,  # Strip whitespace from strings
        validate_assignment=True,  # Validate on assignment
        json_schema_extra={
            "example": {
                "prompt": "A beautiful sunset over mountains",
                "negative_prompt": "blurry, low quality",
                "pipeline_type": "text_to_image",
                "model_type": "stable-diffusion-v1-5",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 42,
                "batch_size": 1
            }
        }
    )
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Text prompt for image generation",
        examples=["A beautiful sunset over mountains", "A futuristic city skyline"]
    )
    negative_prompt: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Negative prompt to avoid certain elements",
        examples=["blurry, low quality, distorted"]
    )
    pipeline_type: PipelineType = Field(
        default=PipelineType.TEXT_TO_IMAGE,
        description="Type of diffusion pipeline to use"
    )
    model_type: ModelType = Field(
        default=ModelType.STABLE_DIFFUSION_V1_5,
        description="Specific model to use"
    )
    num_inference_steps: int = Field(
        default=50, 
        ge=1, 
        le=100, 
        description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=7.5, 
        ge=1.0, 
        le=20.0, 
        description="How closely to follow the prompt"
    )
    width: int = Field(
        default=512, 
        ge=64, 
        le=2048, 
        description="Image width"
    )
    height: int = Field(
        default=512, 
        ge=64, 
        le=2048, 
        description="Image height"
    )
    seed: Optional[int] = Field(
        None, 
        ge=0, 
        le=2**32-1, 
        description="Random seed for reproducible results"
    )
    batch_size: int = Field(
        default=1, 
        ge=1, 
        le=4, 
        description="Number of images to generate"
    )
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions are divisible by 8."""
        if v % 8 != 0:
            raise ValueError('Width and height must be divisible by 8')
        return v
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Sanitize and validate prompt."""
        sanitized = v.strip()
        if not sanitized:
            raise ValueError('Prompt cannot be empty')
        return sanitized[:1000]  # Ensure max length
    
    @model_validator(mode='after')
    def validate_model(self) -> 'DiffusionRequest':
        """Cross-field validation."""
        if self.width * self.height > 1024 * 1024:  # 1MP limit
            raise ValueError('Total image pixels cannot exceed 1,048,576')
        return self
    
    @computed_field
    @property
    def total_pixels(self) -> int:
        """Computed field for total pixels."""
        return self.width * self.height
    
    @computed_field
    @property
    def estimated_memory_mb(self) -> float:
        """Computed field for estimated memory usage."""
        return (self.width * self.height * 3 * 4) / (1024 * 1024)  # RGB float32

class BatchDiffusionRequest(BaseModel):
    """
    Request model for batch image generation.
    Using Pydantic v2 model validation.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "prompt": "A beautiful sunset",
                        "width": 512,
                        "height": 512
                    },
                    {
                        "prompt": "A futuristic city",
                        "width": 768,
                        "height": 512
                    }
                ]
            }
        }
    )
    
    requests: List[DiffusionRequest] = Field(
        ..., 
        min_length=1, 
        max_length=10, 
        description="List of diffusion requests"
    )
    
    @model_validator(mode='after')
    def validate_batch_size(self) -> 'BatchDiffusionRequest':
        """Validate batch size and total images."""
        total_images = sum(req.batch_size for req in self.requests)
        if total_images > 20:
            raise ValueError('Total images in batch cannot exceed 20')
        
        total_memory = sum(req.estimated_memory_mb for req in self.requests)
        if total_memory > 2048:  # 2GB limit
            raise ValueError('Total estimated memory usage cannot exceed 2GB')
        
        return self

class DiffusionResponse(BaseModel):
    """
    Response model for single image generation.
    Using Pydantic v2 features for better serialization.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_encoders={
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        }
    )
    
    image_url: str = Field(..., description="URL to generated image")
    image_id: str = Field(..., description="Unique image identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Generation metadata"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    seed: Optional[int] = Field(None, description="Random seed used")
    model_used: str = Field(..., description="Model used for generation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    
    @computed_field
    @property
    def is_cached(self) -> bool:
        """Computed field indicating if result was cached."""
        return self.image_id.startswith("cached_")

class BatchDiffusionResponse(BaseModel):
    """Response model for batch image generation."""
    model_config = ConfigDict(extra="forbid")
    
    images: List[DiffusionResponse] = Field(..., description="List of generated images")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_id: str = Field(..., description="Unique batch identifier")
    successful_generations: int = Field(..., description="Number of successful generations")
    failed_generations: int = Field(..., description="Number of failed generations")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Computed field for success rate."""
        total = self.successful_generations + self.failed_generations
        return self.successful_generations / total if total > 0 else 0.0

class ErrorResponse(BaseModel):
    """Response model for errors with detailed information."""
    model_config = ConfigDict(extra="forbid")
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")
    path: Optional[str] = Field(None, description="Request path")
    method: Optional[str] = Field(None, description="HTTP method")

class HealthResponse(BaseModel):
    """Response model for health check with comprehensive system info."""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    gpu_available: bool = Field(..., description="GPU availability")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage in MB")
    services: Dict[str, str] = Field(..., description="Service health status")

# ============================================================================
# CUSTOM MIDDLEWARE - Following FastAPI Best Practices
# ============================================================================

class RequestIDMiddleware:
    """
    Middleware to add unique request ID to all requests.
    Following FastAPI middleware best practices.
    """
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            # Generate unique request ID
            request_id = hashlib.md5(
                f"{scope['client'][0]}:{time.time()}".encode()
            ).hexdigest()[:8]
            
            # Add to scope for use in route handlers
            scope["request_id"] = request_id
            
            # Add to response headers
            async async def send_with_request_id(message) -> Any:
                if message["type"] == "http.response.start":
                    message["headers"].append(
                        (b"x-request-id", request_id.encode())
                    )
                await send(message)
            
            await self.app(scope, receive, send_with_request_id)
        else:
            await self.app(scope, receive, send)

class LoggingMiddleware:
    """
    Comprehensive logging middleware with structured logging.
    Following FastAPI middleware patterns.
    """
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = time.time()
            request_id = scope.get("request_id", "unknown")
            
            # Log request
            logger.info(
                f"Request started",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "client": scope["client"][0] if scope["client"] else "unknown"
                }
            )
            
            async def send_with_logging(message) -> Any:
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    status_code = message["status"]
                    
                    # Log response
                    logger.info(
                        f"Request completed",
                        extra={
                            "request_id": request_id,
                            "method": scope["method"],
                            "path": scope["path"],
                            "status_code": status_code,
                            "duration": duration,
                            "client": scope["client"][0] if scope["client"] else "unknown"
                        }
                    )
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)

class PerformanceMiddleware:
    """
    Performance monitoring middleware with Prometheus metrics.
    Following FastAPI middleware best practices.
    """
    
    def __init__(self, app) -> Any:
        self.app = app
        # Prometheus metrics
        self.request_counter = Counter(
            'http_requests_total', 
            'Total HTTP requests',
            ['method', 'path', 'status']
        )
        self.request_duration = Histogram(
            'http_request_duration_seconds', 
            'HTTP request duration',
            ['method', 'path']
        )
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'path']
        )
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'path']
        )
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = time.time()
            method = scope["method"]
            path = scope["path"]
            
            # Track request size
            content_length = 0
            for name, value in scope["headers"]:
                if name == b"content-length":
                    content_length = int(value)
                    break
            
            self.request_size.labels(method=method, path=path).observe(content_length)
            
            async def send_with_metrics(message) -> Any:
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    status = message["status"]
                    
                    # Record metrics
                    self.request_counter.labels(
                        method=method, 
                        path=path, 
                        status=status
                    ).inc()
                    self.request_duration.labels(
                        method=method, 
                        path=path
                    ).observe(duration)
                    
                    # Track response size
                    content_length = 0
                    for name, value in message["headers"]:
                        if name == b"content-length":
                            content_length = int(value)
                            break
                    
                    self.response_size.labels(
                        method=method, 
                        path=path
                    ).observe(content_length)
                
                await send(message)
            
            await self.app(scope, receive, send_with_metrics)
        else:
            await self.app(scope, receive, send)

class SecurityMiddleware:
    """
    Security middleware with headers and validation.
    Following FastAPI security best practices.
    """
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            async def send_with_security_headers(message) -> Any:
                if message["type"] == "http.response.start":
                    # Add security headers
                    security_headers = [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"x-xss-protection", b"1; mode=block"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()")
                    ]
                    
                    for header_name, header_value in security_headers:
                        message["headers"].append((header_name, header_value))
                
                await send(message)
            
            await self.app(scope, receive, send_with_security_headers)
        else:
            await self.app(scope, receive, send)

# ============================================================================
# DEPENDENCY INJECTION - Advanced Patterns
# ============================================================================

# Type aliases for better readability
DatabaseDep = Annotated[Database, Depends()]
CacheDep = Annotated[redis.Redis, Depends()]
UserDep = Annotated[str, Depends()]

class DatabaseService:
    """Database service with dependency injection."""
    
    def __init__(self, database_url: str):
        
    """__init__ function."""
self.database_url = database_url
        self._database: Optional[Database] = None
    
    async def get_database(self) -> Database:
        if self._database is None:
            self._database = Database(self.database_url)
            await self._database.connect()
        return self._database
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        db = await self.get_database()
        try:
            result = await db.fetch_all(text(query), params or {})
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise
    
    async def close(self) -> Any:
        if self._database:
            await self._database.disconnect()

class CacheService:
    """Cache service with dependency injection."""
    
    def __init__(self, redis_url: str):
        
    """__init__ function."""
self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def get(self, key: str) -> Optional[str]:
        redis_client = await self.get_redis()
        try:
            return await redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600) -> bool:
        redis_client = await self.get_redis()
        try:
            return await redis_client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def close(self) -> Any:
        if self._redis:
            await self._redis.close()

# Dependency functions
async def get_database_service() -> DatabaseService:
    """Get database service instance."""
    return DatabaseService("postgresql://user:pass@localhost/db")

async def get_cache_service() -> CacheService:
    """Get cache service instance."""
    return CacheService("redis://localhost:6379")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> str:
    """Get current user from JWT token."""
    # In production, validate JWT token
    return "default_user"

async def get_rate_limit_info(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """Get rate limiting information."""
    client_ip = request.client.host
    rate_limit_key = f"rate_limit:{client_ip}"
    
    redis_client = await cache_service.get_redis()
    current_count = await redis_client.get(rate_limit_key)
    current_count = int(current_count) if current_count else 0
    
    limit_per_minute = 60
    if current_count >= limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    await redis_client.set(rate_limit_key, str(current_count + 1), ex=60)
    
    return {
        "requests_per_minute": limit_per_minute,
        "remaining_requests": limit_per_minute - current_count - 1,
        "reset_time": datetime.now(timezone.utc).timestamp() + 60
    }

# ============================================================================
# PATH OPERATIONS - Following FastAPI Best Practices
# ============================================================================

class DiffusionAPI:
    """Diffusion API with organized path operations."""
    
    def __init__(self, db_service: DatabaseService, cache_service: CacheService):
        
    """__init__ function."""
self.db_service = db_service
        self.cache_service = cache_service
    
    async def generate_single_image(
        self,
        request: DiffusionRequest,
        current_user: UserDep = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> DiffusionResponse:
        """Generate single image from text prompt."""
        start_time = time.time()
        
        try:
            # Check cache first
            prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
            cached_result = await self.cache_service.get(f"generation:{prompt_hash}")
            
            if cached_result:
                processing_time = time.time() - start_time
                return DiffusionResponse(
                    image_url=cached_result,
                    image_id=f"cached_{prompt_hash}",
                    processing_time=processing_time,
                    model_used=request.model_type.value
                )
            
            # Generate new image (simulated)
            await asyncio.sleep(1)
            
            # Save to database
            image_id = await self._save_generation_result(
                current_user, request.prompt, f"https://example.com/generated/{prompt_hash}.png"
            )
            
            # Cache result
            await self.cache_service.set(
                f"generation:{prompt_hash}", 
                f"https://example.com/generated/{prompt_hash}.png"
            )
            
            processing_time = time.time() - start_time
            
            return DiffusionResponse(
                image_url=f"https://example.com/generated/{prompt_hash}.png",
                image_id=image_id or prompt_hash,
                processing_time=processing_time,
                seed=request.seed,
                model_used=request.model_type.value,
                metadata={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "pipeline_type": request.pipeline_type.value,
                    "parameters": request.model_dump()
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image generation failed"
            )
    
    async def generate_batch_images(
        self,
        request: BatchDiffusionRequest,
        current_user: UserDep = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> BatchDiffusionResponse:
        """Generate multiple images in batch."""
        start_time = time.time()
        
        try:
            # Process requests in parallel
            tasks = [
                self.generate_single_image(
                    req,
                    current_user,
                    rate_limit
                )
                for req in request.requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful and failed generations
            successful = []
            failed = 0
            
            for result in results:
                if isinstance(result, DiffusionResponse):
                    successful.append(result)
                else:
                    failed += 1
            
            total_processing_time = time.time() - start_time
            
            return BatchDiffusionResponse(
                images=successful,
                total_processing_time=total_processing_time,
                batch_id=f"batch_{int(time.time())}",
                successful_generations=len(successful),
                failed_generations=failed
            )
            
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch generation failed"
            )
    
    async def _save_generation_result(self, user_id: str, prompt: str, result_url: str) -> str:
        """Save generation result to database."""
        query = """
            INSERT INTO image_generations (user_id, prompt, result_url, created_at)
            VALUES (:user_id, :prompt, :result_url, NOW())
            RETURNING id
        """
        result = await self.db_service.execute_query(
            query, 
            {"user_id": user_id, "prompt": prompt, "result_url": result_url}
        )
        return result[0]['id'] if result else None

# ============================================================================
# APPLICATION FACTORY - Following FastAPI Best Practices
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Application starting up...")
    
    # Initialize services
    app.state.db_service = DatabaseService("postgresql://user:pass@localhost/db")
    app.state.cache_service = CacheService("redis://localhost:6379")
    app.state.diffusion_api = DiffusionAPI(
        app.state.db_service, 
        app.state.cache_service
    )
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    await app.state.db_service.close()
    await app.state.cache_service.close()

def create_application() -> FastAPI:
    """Create and configure FastAPI application following best practices."""
    
    app = FastAPI(
        title="NotebookLM AI API",
        description="Advanced AI diffusion model API with best practices",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware in order (last added = first executed)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # Configure properly in production
    )
    
    # Register routes
    register_routes(app)
    
    # Custom OpenAPI schema
    def custom_openapi():
        
    """custom_openapi function."""
if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app

def register_routes(app: FastAPI):
    """Register all application routes following FastAPI best practices."""
    
    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        summary="Health Check",
        description="Check API health and system status",
        response_description="System health information",
        tags=["Health"]
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=time.time() - 0,  # Would be actual uptime
            gpu_available=True,
            models_loaded={
                "stable-diffusion-v1-5": True,
                "stable-diffusion-xl": True
            },
            memory_usage={
                "gpu": 2048.0,
                "ram": 8192.0
            },
            services={
                "database": "healthy",
                "cache": "healthy",
                "api": "healthy"
            }
        )
    
    # Metrics endpoint for Prometheus
    @app.get(
        "/metrics",
        summary="Prometheus Metrics",
        description="Prometheus metrics endpoint",
        tags=["Monitoring"]
    )
    async def metrics():
        
    """metrics function."""
return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Diffusion endpoints
    @app.post(
        "/api/v1/diffusion/generate",
        response_model=DiffusionResponse,
        status_code=status.HTTP_200_OK,
        summary="Generate Single Image",
        description="Generate an image from a text prompt using diffusion models",
        response_description="Generated image information",
        tags=["Diffusion"],
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
            status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}
        }
    )
    async def generate_image(
        request: DiffusionRequest,
        current_user: UserDep = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> DiffusionResponse:
        return await app.state.diffusion_api.generate_single_image(
            request, current_user, rate_limit
        )
    
    @app.post(
        "/api/v1/diffusion/generate-batch",
        response_model=BatchDiffusionResponse,
        status_code=status.HTTP_200_OK,
        summary="Generate Batch Images",
        description="Generate multiple images in a single batch request",
        response_description="Batch generation results",
        tags=["Diffusion"],
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
            status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}
        }
    )
    async def generate_batch_images(
        request: BatchDiffusionRequest,
        current_user: UserDep = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> BatchDiffusionResponse:
        return await app.state.diffusion_api.generate_batch_images(
            request, current_user, rate_limit
        )
    
    # File upload endpoint
    @app.post(
        "/api/v1/diffusion/upload",
        summary="Upload Image",
        description="Upload an image for processing",
        tags=["Diffusion"],
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse}
        }
    )
    async def upload_image(
        file: UploadFile = File(
            ...,
            description="Image file to upload",
            max_length=10 * 1024 * 1024  # 10MB limit
        ),
        current_user: UserDep = Depends(get_current_user)
    ):
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Process file
        content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        file_hash = hashlib.md5(content).hexdigest()
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "hash": file_hash,
            "uploaded_by": current_user
        }

# ============================================================================
# ERROR HANDLERS - Comprehensive Error Management
# ============================================================================

def register_error_handlers(app: FastAPI):
    """Register comprehensive error handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=str(exc),
                error_code="VALIDATION_ERROR",
                request_id=request.scope.get("request_id"),
                path=request.url.path,
                method=request.method
            ).model_dump()
        )
    
    @app.exception_handler(HTTPException)
    async async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=exc.detail,
                error_code="HTTP_ERROR",
                request_id=request.scope.get("request_id"),
                path=request.url.path,
                method=request.method
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="Internal server error",
                error_code="INTERNAL_ERROR",
                request_id=request.scope.get("request_id"),
                path=request.url.path,
                method=request.method
            ).model_dump()
        )

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

app = create_application()
register_error_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_best_practices_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 