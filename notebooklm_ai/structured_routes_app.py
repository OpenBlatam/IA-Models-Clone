from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps
from contextlib import asynccontextmanager
import json
import hashlib
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum
import aiohttp
import aiofiles
import redis.asyncio as redis
from databases import Database
from sqlalchemy import text
import motor.motor_asyncio
from bson import ObjectId
from prometheus_client import Counter, Histogram, Gauge
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Structured FastAPI Application with Clear Route Organization and Dependencies
- Modular architecture with separate concerns
- Clear dependency injection patterns
- Organized route structure by domain
- Improved readability and maintainability
"""


# FastAPI and async dependencies

# Async database and external operations

# Performance monitoring

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

class AppConfig:
    """Application configuration with environment-based settings."""
    
    def __init__(self) -> Any:
        self.app_name: str = "notebooklm_ai"
        self.version: str = "1.0.0"
        self.debug: bool = False
        self.database_url: str = "postgresql://user:pass@localhost/db"
        self.redis_url: str = "redis://localhost:6379"
        self.api_timeout: int = 30
        self.max_connections: int = 100
        self.cache_ttl: int = 3600
        self.rate_limit_per_minute: int = 60
        self.allowed_origins: List[str] = ["*"]
        
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create config from environment variables."""
        config = cls()
        # In production, load from environment variables
        return config

# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class DependencyContainer:
    """Centralized dependency injection container."""
    
    def __init__(self, config: AppConfig):
        
    """__init__ function."""
self.config = config
        self._database_manager: Optional['AsyncDatabaseManager'] = None
        self._cache_manager: Optional['AsyncCacheManager'] = None
        self._api_manager: Optional['AsyncExternalAPIManager'] = None
        self._diffusion_service: Optional['AsyncDiffusionService'] = None
        self._external_api_service: Optional['AsyncExternalAPIService'] = None
        
    async def get_database_manager(self) -> 'AsyncDatabaseManager':
        """Get or create database manager."""
        if self._database_manager is None:
            self._database_manager = AsyncDatabaseManager(self.config.database_url)
        return self._database_manager
    
    async def get_cache_manager(self) -> 'AsyncCacheManager':
        """Get or create cache manager."""
        if self._cache_manager is None:
            self._cache_manager = AsyncCacheManager(self.config.redis_url)
        return self._cache_manager
    
    async async def get_api_manager(self) -> 'AsyncExternalAPIManager':
        """Get or create external API manager."""
        if self._api_manager is None:
            self._api_manager = AsyncExternalAPIManager(
                timeout=self.config.api_timeout,
                max_connections=self.config.max_connections
            )
        return self._api_manager
    
    async def get_diffusion_service(self) -> 'AsyncDiffusionService':
        """Get or create diffusion service."""
        if self._diffusion_service is None:
            db_manager = await self.get_database_manager()
            cache_manager = await self.get_cache_manager()
            self._diffusion_service = AsyncDiffusionService(db_manager, cache_manager)
        return self._diffusion_service
    
    async async def get_external_api_service(self) -> 'AsyncExternalAPIService':
        """Get or create external API service."""
        if self._external_api_service is None:
            api_manager = await self.get_api_manager()
            self._external_api_service = AsyncExternalAPIService(api_manager)
        return self._external_api_service
    
    async def cleanup(self) -> Any:
        """Cleanup all resources."""
        if self._database_manager:
            await self._database_manager.close()
        if self._cache_manager:
            await self._cache_manager.close()
        if self._api_manager:
            await self._api_manager.close()

# ============================================================================
# PYDANTIC MODELS - Input/Output Schemas
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
    """Request model for single image generation."""
    model_config = ConfigDict(extra="forbid")
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Text prompt for image generation"
    )
    negative_prompt: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Negative prompt to avoid certain elements"
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
    
    @validator('width', 'height')
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions are divisible by 8."""
        if v % 8 != 0:
            raise ValueError('Width and height must be divisible by 8')
        return v

class BatchDiffusionRequest(BaseModel):
    """Request model for batch image generation."""
    model_config = ConfigDict(extra="forbid")
    
    requests: List[DiffusionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=10, 
        description="List of diffusion requests"
    )

class DiffusionResponse(BaseModel):
    """Response model for single image generation."""
    model_config = ConfigDict(extra="forbid")
    
    image_url: str = Field(..., description="URL to generated image")
    image_id: str = Field(..., description="Unique image identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Generation metadata"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    seed: Optional[int] = Field(None, description="Random seed used")
    model_used: str = Field(..., description="Model used for generation")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BatchDiffusionResponse(BaseModel):
    """Response model for batch image generation."""
    model_config = ConfigDict(extra="forbid")
    
    images: List[DiffusionResponse] = Field(..., description="List of generated images")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_id: str = Field(..., description="Unique batch identifier")
    successful_generations: int = Field(..., description="Number of successful generations")
    failed_generations: int = Field(..., description="Number of failed generations")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    model_config = ConfigDict(extra="forbid")
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")

class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    gpu_available: bool = Field(..., description="GPU availability")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage in MB")

# ============================================================================
# ASYNC MANAGERS - Core Services
# ============================================================================

class AsyncDatabaseManager:
    """Dedicated async database manager for non-blocking operations."""
    
    def __init__(self, database_url: str):
        
    """__init__ function."""
self.database_url = database_url
        self._database: Optional[Database] = None
        self._mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        
    async def get_database(self) -> Database:
        """Get async database connection."""
        if self._database is None:
            self._database = Database(self.database_url)
            await self._database.connect()
        return self._database
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute async database query."""
        db = await self.get_database()
        try:
            result = await db.fetch_all(text(query), params or {})
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise
    
    async def close(self) -> Any:
        """Close database connections."""
        if self._database:
            await self._database.disconnect()
        if self._mongo_client:
            self._mongo_client.close()

class AsyncCacheManager:
    """Dedicated async cache manager for non-blocking operations."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        
    async def get_redis(self) -> redis.Redis:
        """Get async Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        redis_client = await self.get_redis()
        try:
            return await redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in cache."""
        redis_client = await self.get_redis()
        try:
            return await redis_client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def close(self) -> Any:
        """Close cache connection."""
        if self._redis:
            await self._redis.close()

class AsyncExternalAPIManager:
    """Dedicated async external API manager for non-blocking operations."""
    
    def __init__(self, timeout: int = 30, max_connections: int = 100):
        
    """__init__ function."""
self.timeout = timeout
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get async HTTP session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async async def make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make async HTTP request."""
        session = await self.get_session()
        start_time = time.time()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                duration = time.time() - start_time
                
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
                
                return {
                    'status_code': response.status,
                    'data': data,
                    'headers': dict(response.headers),
                    'duration': duration,
                    'url': url,
                    'method': method
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            raise HTTPException(status_code=504, detail="External API timeout")
        except Exception as e:
            logger.error(f"External API error: {e}")
            raise HTTPException(status_code=502, detail="External API error")
    
    async def close(self) -> Any:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

# ============================================================================
# BUSINESS LOGIC SERVICES
# ============================================================================

class AsyncDiffusionService:
    """Business logic service for diffusion operations."""
    
    def __init__(self, db_manager: AsyncDatabaseManager, cache_manager: AsyncCacheManager):
        
    """__init__ function."""
self.db_manager = db_manager
        self.cache_manager = cache_manager
    
    async def save_generation_result(self, user_id: str, prompt: str, result_url: str) -> str:
        """Save generation result to database."""
        query = """
            INSERT INTO image_generations (user_id, prompt, result_url, created_at)
            VALUES (:user_id, :prompt, :result_url, NOW())
            RETURNING id
        """
        result = await self.db_manager.execute_query(
            query, 
            {"user_id": user_id, "prompt": prompt, "result_url": result_url}
        )
        return result[0]['id'] if result else None
    
    async def get_cached_result(self, prompt_hash: str) -> Optional[str]:
        """Get cached generation result."""
        return await self.cache_manager.get(f"generation:{prompt_hash}")
    
    async def cache_generation_result(self, prompt_hash: str, result_url: str) -> bool:
        """Cache generation result."""
        return await self.cache_manager.set(f"generation:{prompt_hash}", result_url)
    
    async def get_user_generations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's generation history."""
        query = """
            SELECT * FROM image_generations 
            WHERE user_id = :user_id 
            ORDER BY created_at DESC 
            LIMIT :limit
        """
        return await self.db_manager.execute_query(
            query, 
            {"user_id": user_id, "limit": limit}
        )

class AsyncExternalAPIService:
    """Business logic service for external API operations."""
    
    def __init__(self, api_manager: AsyncExternalAPIManager):
        
    """__init__ function."""
self.api_manager = api_manager
    
    async async def call_diffusion_api(self, prompt: str, parameters: Dict) -> Dict:
        """Call external diffusion API."""
        # Simulate external API call
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            'status_code': 200,
            'data': {
                'image_url': f"https://example.com/generated/{hash(prompt)}.png",
                'prompt': prompt,
                'parameters': parameters
            }
        }
    
    async async def call_multiple_apis(self, requests: List[Dict]) -> List[Dict]:
        """Call multiple external APIs in parallel."""
        tasks = [self.api_manager.make_request(**req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ============================================================================
# DEPENDENCY FUNCTIONS - Route Dependencies
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> str:
    """Get current user from JWT token."""
    # In production, validate JWT token
    return "default_user"

async def get_rate_limit_info() -> Dict[str, Any]:
    """Get rate limiting information."""
    return {
        "requests_per_minute": 60,
        "remaining_requests": 59,
        "reset_time": datetime.utcnow().timestamp() + 60
    }

async def get_dependency_container(request: Request) -> DependencyContainer:
    """Get dependency container from request state."""
    return request.app.state.dependency_container

async def get_diffusion_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> AsyncDiffusionService:
    """Get diffusion service from dependency container."""
    return await container.get_diffusion_service()

async def get_external_api_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> AsyncExternalAPIService:
    """Get external API service from dependency container."""
    return await container.get_external_api_service()

# ============================================================================
# ROUTE HANDLERS - Organized by Domain
# ============================================================================

class DiffusionRoutes:
    """Route handlers for diffusion operations."""
    
    @staticmethod
    async def generate_single_image(
        request: DiffusionRequest,
        diffusion_service: AsyncDiffusionService = Depends(get_diffusion_service),
        current_user: str = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> DiffusionResponse:
        """Generate single image from text prompt."""
        start_time = time.time()
        
        try:
            # Check cache first
            prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
            cached_result = await diffusion_service.get_cached_result(prompt_hash)
            
            if cached_result:
                processing_time = time.time() - start_time
                return DiffusionResponse(
                    image_url=cached_result,
                    image_id=f"cached_{prompt_hash}",
                    processing_time=processing_time,
                    model_used=request.model_type.value
                )
            
            # Generate new image (simulated)
            await asyncio.sleep(1)  # Simulate generation time
            
            # Save to database
            image_id = await diffusion_service.save_generation_result(
                current_user, request.prompt, f"https://example.com/generated/{prompt_hash}.png"
            )
            
            # Cache result
            await diffusion_service.cache_generation_result(
                prompt_hash, f"https://example.com/generated/{prompt_hash}.png"
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
                    "parameters": request.dict()
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise HTTPException(status_code=500, detail="Image generation failed")
    
    @staticmethod
    async def generate_batch_images(
        request: BatchDiffusionRequest,
        diffusion_service: AsyncDiffusionService = Depends(get_diffusion_service),
        current_user: str = Depends(get_current_user),
        rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
    ) -> BatchDiffusionResponse:
        """Generate multiple images in batch."""
        start_time = time.time()
        
        try:
            # Process requests in parallel
            tasks = [
                DiffusionRoutes.generate_single_image(
                    DiffusionRequest(**req.dict()),
                    diffusion_service,
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
            raise HTTPException(status_code=500, detail="Batch generation failed")

class HealthRoutes:
    """Route handlers for health and monitoring."""
    
    @staticmethod
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=time.time() - 0,  # Would be actual uptime in production
            gpu_available=True,
            models_loaded={
                "stable-diffusion-v1-5": True,
                "stable-diffusion-xl": True
            },
            memory_usage={
                "gpu": 2048.0,
                "ram": 8192.0
            }
        )

# ============================================================================
# MIDDLEWARE - Request/Response Processing
# ============================================================================

class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_with_logging(message) -> Any:
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    logger.info(
                        f"{scope['method']} {scope['path']} - "
                        f"{message['status']} - {duration:.3f}s"
                    )
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)

class PerformanceMiddleware:
    """Middleware for performance monitoring."""
    
    def __init__(self, app) -> Any:
        self.app = app
        self.request_counter = Counter('http_requests_total', 'Total HTTP requests')
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            self.request_counter.inc()
            start_time = time.time()
            
            async def send_with_metrics(message) -> Any:
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    self.request_duration.observe(duration)
                await send(message)
            
            await self.app(scope, receive, send_with_metrics)
        else:
            await self.app(scope, receive, send)

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = AppConfig.from_env()
    container = DependencyContainer(config)
    app.state.dependency_container = container
    app.state.config = config
    
    logger.info("Application starting up...")
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    await container.cleanup()

def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    config = AppConfig.from_env()
    
    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description="Structured FastAPI application with clear route organization",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(PerformanceMiddleware)
    
    # Register routes
    register_routes(app)
    
    return app

def register_routes(app: FastAPI):
    """Register all application routes."""
    
    # Diffusion routes
    app.add_api_route(
        "/api/v1/diffusion/generate",
        DiffusionRoutes.generate_single_image,
        methods=["POST"],
        response_model=DiffusionResponse,
        status_code=200,
        summary="Generate single image from text prompt",
        description="Generate an image from a text prompt using diffusion models",
        responses={
            400: {"model": ErrorResponse, "description": "Bad request"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        }
    )
    
    app.add_api_route(
        "/api/v1/diffusion/generate-batch",
        DiffusionRoutes.generate_batch_images,
        methods=["POST"],
        response_model=BatchDiffusionResponse,
        status_code=200,
        summary="Generate multiple images in batch",
        description="Generate multiple images in a single batch request",
        responses={
            400: {"model": ErrorResponse, "description": "Bad request"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        }
    )
    
    # Health routes
    app.add_api_route(
        "/api/v1/health",
        HealthRoutes.health_check,
        methods=["GET"],
        response_model=HealthResponse,
        status_code=200,
        summary="Health check endpoint",
        description="Check API health and system status",
        responses={
            500: {"model": ErrorResponse, "description": "Service unhealthy"}
        }
    )

# ============================================================================
# ERROR HANDLERS
# ============================================================================

def register_error_handlers(app: FastAPI):
    """Register global error handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError exceptions."""
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exc),
                "error_code": "VALIDATION_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, 'request_id', None)
            }
        )

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

app = create_application()
register_error_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "structured_routes_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 