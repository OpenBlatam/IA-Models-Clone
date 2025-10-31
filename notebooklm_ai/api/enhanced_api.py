from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import json
import os
import sys
from typing import Dict, Any, List, Optional, Union, Callable
from contextlib import asynccontextmanager
from pathlib import Path
import traceback
from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic_settings import BaseSettings
import httpx
import aiofiles
from asyncio import Semaphore
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog
import jwt
from passlib.context import CryptContext
import secrets
from datetime import datetime, timedelta
import hashlib
from integration_master import IntegrationMaster
from production_config import get_config, ProductionConfig
            import base64
        import tempfile
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Enhanced FastAPI Application
============================

Advanced FastAPI application with enterprise-grade features:
- Enhanced error handling and validation
- Performance optimizations
- Advanced middleware
- Comprehensive logging
- Rate limiting and security
- Caching strategies
- Health checks and monitoring
"""


# FastAPI and core imports

# Pydantic for validation

# Async and performance

# Monitoring and metrics

# Security and utilities

# Local imports

# Setup structured logging
def setup_logging():
    """Setup structured logging with JSON format"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Setup Prometheus metrics
def setup_metrics():
    """Setup Prometheus metrics for monitoring"""
    return {
        'request_counter': Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status']),
        'request_duration': Histogram('api_request_duration_seconds', 'Request duration', ['endpoint']),
        'error_counter': Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type']),
        'active_connections': Gauge('api_active_connections', 'Active connections'),
        'cache_hits': Counter('api_cache_hits_total', 'Cache hits', ['cache_type']),
        'cache_misses': Counter('api_cache_misses_total', 'Cache misses', ['cache_type']),
        'rate_limit_hits': Counter('api_rate_limit_hits_total', 'Rate limit hits', ['endpoint']),
    }

# Pydantic models for request/response validation
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(default=False, description="Operation success status")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")

class SuccessResponse(BaseResponse):
    """Success response model"""
    success: bool = Field(default=True, description="Operation success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

class TextProcessingRequest(BaseModel):
    """Text processing request model"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    operations: List[str] = Field(
        default=["statistics", "sentiment", "keywords"],
        description="Processing operations to perform"
    )
    language: Optional[str] = Field(None, description="Text language code")
    cache_result: bool = Field(default=True, description="Cache processing result")

    @validator('operations')
    def validate_operations(cls, v) -> bool:
        valid_operations = {
            "statistics", "sentiment", "keywords", "topics", 
            "entities", "summarization", "classification"
        }
        invalid_ops = set(v) - valid_operations
        if invalid_ops:
            raise ValueError(f"Invalid operations: {invalid_ops}")
        return v

class ImageProcessingRequest(BaseModel):
    """Image processing request model"""
    image_url: Optional[str] = Field(None, description="Image URL")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    operations: List[str] = Field(
        default=["properties", "face_detection"],
        description="Processing operations to perform"
    )
    quality: int = Field(default=80, ge=1, le=100, description="Processing quality")

    @validator('image_url', 'image_base64')
    def validate_image_source(cls, v, values) -> bool:
        if not values.get('image_url') and not values.get('image_base64'):
            raise ValueError("Either image_url or image_base64 must be provided")
        return v

class VectorSearchRequest(BaseModel):
    """Vector search request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")

class BatchProcessingRequest(BaseModel):
    """Batch processing request model"""
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000, description="Items to process")
    operation_type: str = Field(..., description="Type of operation to perform")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for processing")
    parallel: bool = Field(default=True, description="Process batches in parallel")

class PerformanceOptimizationRequest(BaseModel):
    """Performance optimization request model"""
    task_type: str = Field(..., description="Type of task to optimize")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")
    target_metric: str = Field(default="speed", description="Target optimization metric")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")

# Custom exception classes
class APIException(HTTPException):
    """Custom API exception with error code"""
    def __init__(self, status_code: int, error_code: str, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        self.details = details

class ValidationException(APIException):
    """Validation error exception"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message=message,
            details=details
        )

class RateLimitException(APIException):
    """Rate limit exceeded exception"""
    def __init__(self, retry_after: int = 60):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            message="Rate limit exceeded",
            details={"retry_after": retry_after}
        )

class CacheException(APIException):
    """Cache operation exception"""
    def __init__(self, message: str):
        
    """__init__ function."""
super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="CACHE_ERROR",
            message=message
        )

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis backend"""
    
    def __init__(self, app, redis_client: aioredis.Redis, requests_per_minute: int = 60):
        
    """__init__ function."""
super().__init__(app)
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.logger = structlog.get_logger()
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        # Create rate limit key
        rate_limit_key = f"rate_limit:{client_ip}:{endpoint}:{int(time.time() // 60)}"
        
        try:
            # Check current request count
            current_requests = await self.redis.get(rate_limit_key)
            current_count = int(current_requests) if current_requests else 0
            
            if current_count >= self.requests_per_minute:
                self.logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    endpoint=endpoint,
                    current_count=current_count
                )
                raise RateLimitException()
            
            # Increment request count
            await self.redis.incr(rate_limit_key)
            await self.redis.expire(rate_limit_key, 60)  # Expire after 1 minute
            
            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - current_count - 1)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() // 60 * 60 + 60))
            
            return response
            
        except RateLimitException:
            raise
        except Exception as e:
            self.logger.error(f"Rate limiting error: {e}")
            # Continue without rate limiting on error
            return await call_next(request)

# Caching middleware
class CacheMiddleware(BaseHTTPMiddleware):
    """Caching middleware with Redis backend"""
    
    def __init__(self, app, redis_client: aioredis.Redis, default_ttl: int = 300):
        
    """__init__ function."""
super().__init__(app)
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = structlog.get_logger()
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Create cache key
        cache_key = f"cache:{request.url.path}:{hash(str(request.query_params))}"
        
        try:
            # Try to get from cache
            cached_response = await self.redis.get(cache_key)
            if cached_response:
                metrics['cache_hits'].labels(cache_type='api').inc()
                return JSONResponse(
                    content=json.loads(cached_response),
                    headers={"X-Cache": "HIT"}
                )
            
            metrics['cache_misses'].labels(cache_type='api').inc()
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                response_content = response.body.decode() if hasattr(response, 'body') else ""
                await self.redis.setex(cache_key, self.default_ttl, response_content)
                response.headers["X-Cache"] = "MISS"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Caching error: {e}")
            # Continue without caching on error
            return await call_next(request)

# Security middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for additional protection"""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.logger = structlog.get_logger()
    
    async def dispatch(self, request: StarletteRequest, call_next: Callable) -> StarletteResponse:
        # Add security headers
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

# Authentication utilities
class AuthService:
    """Authentication service with JWT tokens"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        
    """__init__ function."""
self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.logger = structlog.get_logger()
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise APIException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code="TOKEN_EXPIRED",
                message="Token has expired"
            )
        except jwt.JWTError:
            raise APIException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code="INVALID_TOKEN",
                message="Invalid token"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

# Dependency injection
async def get_integration_master() -> IntegrationMaster:
    """Get integration master instance"""
    if not hasattr(get_integration_master, 'instance'):
        get_integration_master.instance = IntegrationMaster()
        await get_integration_master.instance.start()
    return get_integration_master.instance

async def get_redis_client() -> aioredis.Redis:
    """Get Redis client instance"""
    if not hasattr(get_redis_client, 'client'):
        config = get_config()
        get_redis_client.client = aioredis.from_url(config.get_redis_url())
    return get_redis_client.client

async def get_auth_service() -> AuthService:
    """Get authentication service instance"""
    if not hasattr(get_auth_service, 'instance'):
        config = get_config()
        get_auth_service.instance = AuthService(config.security.secret_key)
    return get_auth_service.instance

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        payload = auth_service.verify_token(credentials.credentials)
        return payload
    except APIException:
        raise
    except Exception as e:
        raise APIException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_FAILED",
            message="Authentication failed"
        )

# Error handlers
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors"""
    logger = structlog.get_logger()
    logger.error("Validation error", errors=exc.errors(), path=request.url.path)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            message="Validation error",
            error_code="VALIDATION_ERROR",
            details={"errors": exc.errors()}
        ).dict()
    )

async async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger = structlog.get_logger()
    logger.error(f"HTTP error {exc.status_code}", detail=exc.detail, path=request.url.path)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code="HTTP_ERROR"
        ).dict()
    )

async async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle custom API exceptions"""
    logger = structlog.get_logger()
    logger.error(
        f"API error {exc.status_code}",
        error_code=exc.error_code,
        message=exc.detail,
        details=exc.details,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=exc.error_code,
            details=exc.details
        ).dict()
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions"""
    logger = structlog.get_logger()
    logger.error(
        "Unhandled exception",
        error=str(exc),
        traceback=traceback.format_exc(),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            message="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger = structlog.get_logger()
    logger.info("🚀 Starting Enhanced API Server")
    
    # Initialize Redis connection
    redis_client = await get_redis_client()
    await redis_client.ping()
    logger.info("✅ Redis connection established")
    
    # Initialize integration master
    integration_master = await get_integration_master()
    logger.info("✅ Integration master initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced API Server")
    
    # Close Redis connection
    if hasattr(get_redis_client, 'client'):
        await get_redis_client.client.close()
    
    # Shutdown integration master
    if hasattr(get_integration_master, 'instance'):
        await get_integration_master.instance.shutdown()
    
    logger.info("✅ Enhanced API Server shutdown completed")

# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Setup logging and metrics
    setup_logging()
    global metrics
    metrics = setup_metrics()
    
    # Get configuration
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="Enhanced Advanced Library Integration API",
        description="Enterprise-grade API with advanced AI capabilities",
        version="2.0.0",
        docs_url="/docs" if config.environment.value != "production" else None,
        redoc_url="/redoc" if config.environment.value != "production" else None,
        openapi_url="/openapi.json" if config.environment.value != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(SecurityMiddleware)
    
    # Add custom middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        
    """metrics_middleware function."""
start_time = time.time()
        
        # Update metrics
        metrics['active_connections'].inc()
        metrics['request_counter'].labels(
            endpoint=request.url.path,
            method=request.method,
            status="pending"
        ).inc()
        
        try:
            response = await call_next(request)
            
            # Log successful request
            duration = time.time() - start_time
            metrics['request_duration'].labels(endpoint=request.url.path).observe(duration)
            metrics['request_counter'].labels(
                endpoint=request.url.path,
                method=request.method,
                status=str(response.status_code)
            ).inc()
            
            logger = structlog.get_logger()
            logger.info(
                "Request processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                client_ip=request.client.host if request.client else None
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            metrics['error_counter'].labels(
                endpoint=request.url.path,
                error_type=type(e).__name__
            ).inc()
            
            logger = structlog.get_logger()
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration=duration,
                traceback=traceback.format_exc()
            )
            
            raise
        finally:
            metrics['active_connections'].dec()
    
    # Add exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    return app

# Create app instance
app = create_app()

# API Routes
@app.get("/", response_model=SuccessResponse)
async def root() -> SuccessResponse:
    """Root endpoint with API information"""
    return SuccessResponse(
        message="Enhanced Advanced Library Integration API",
        data={
            "version": "2.0.0",
            "status": "running",
            "features": [
                "Advanced AI Processing",
                "Real-time Analytics",
                "Performance Optimization",
                "Enterprise Security",
                "Comprehensive Monitoring"
            ],
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
                "text_processing": "/api/v1/text/process",
                "image_processing": "/api/v1/image/process",
                "vector_search": "/api/v1/vector/search",
                "batch_processing": "/api/v1/batch/process",
                "performance_optimization": "/api/v1/optimize/performance"
            }
        }
    )

@app.get("/health", response_model=SuccessResponse)
async def health_check(
    integration_master: IntegrationMaster = Depends(get_integration_master),
    redis_client: aioredis.Redis = Depends(get_redis_client)
) -> SuccessResponse:
    """Comprehensive health check"""
    try:
        # Check integration master health
        health = await integration_master.health_check()
        
        # Check Redis health
        await redis_client.ping()
        redis_health = "healthy"
        
        # Check system info
        system_info = integration_master.get_system_info()
        
        # Determine overall health
        overall_health = "healthy"
        if health['overall'] != 'healthy' or redis_health != 'healthy':
            overall_health = "degraded"
        
        return SuccessResponse(
            message="Health check completed",
            data={
                "status": overall_health,
                "timestamp": datetime.utcnow().isoformat(),
                "integration_master": health,
                "redis": {"status": redis_health},
                "system_info": system_info
            }
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Health check failed: {e}")
        raise APIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="HEALTH_CHECK_FAILED",
            message=f"Health check failed: {str(e)}"
        )

@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/api/v1/text/process", response_model=SuccessResponse)
async def process_text(
    request: TextProcessingRequest,
    integration_master: IntegrationMaster = Depends(get_integration_master),
    redis_client: aioredis.Redis = Depends(get_redis_client),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Process text with advanced NLP"""
    start_time = time.time()
    
    try:
        # Check cache if enabled
        if request.cache_result:
            cache_key = f"text_process:{hash(request.text + str(request.operations))}"
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                metrics['cache_hits'].labels(cache_type='text_processing').inc()
                return SuccessResponse(
                    message="Text processed (cached)",
                    data=json.loads(cached_result)
                )
        
        # Process text
        results = await integration_master.process_text(request.text, request.operations)
        
        # Cache result if enabled
        if request.cache_result:
            cache_key = f"text_process:{hash(request.text + str(request.operations))}"
            await redis_client.setex(cache_key, 3600, json.dumps(results))  # Cache for 1 hour
        
        duration = time.time() - start_time
        
        return SuccessResponse(
            message="Text processed successfully",
            data={
                "results": results,
                "processing_time": duration,
                "operations": request.operations,
                "text_length": len(request.text)
            }
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Text processing failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="TEXT_PROCESSING_FAILED",
            message=f"Text processing failed: {str(e)}"
        )

@app.post("/api/v1/image/process", response_model=SuccessResponse)
async def process_image(
    request: ImageProcessingRequest,
    integration_master: IntegrationMaster = Depends(get_integration_master),
    redis_client: aioredis.Redis = Depends(get_redis_client),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Process image with computer vision"""
    start_time = time.time()
    
    try:
        # Handle image source
        if request.image_url:
            # Download image from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url)
                response.raise_for_status()
                image_data = response.content
        elif request.image_base64:
            # Decode base64 image
            image_data = base64.b64decode(request.image_base64)
        else:
            raise ValidationException("Either image_url or image_base64 must be provided")
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file_path = temp_file.name
        
        try:
            # Process image
            results = await integration_master.process_image(temp_file_path, request.operations)
            
            duration = time.time() - start_time
            
            return SuccessResponse(
                message="Image processed successfully",
                data={
                    "results": results,
                    "processing_time": duration,
                    "operations": request.operations,
                    "quality": request.quality
                }
            )
        finally:
            # Cleanup temporary file
            os.unlink(temp_file_path)
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Image processing failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="IMAGE_PROCESSING_FAILED",
            message=f"Image processing failed: {str(e)}"
        )

@app.post("/api/v1/vector/search", response_model=SuccessResponse)
async def vector_search(
    request: VectorSearchRequest,
    integration_master: IntegrationMaster = Depends(get_integration_master),
    redis_client: aioredis.Redis = Depends(get_redis_client),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Perform vector similarity search"""
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = f"vector_search:{hash(request.query + str(request.top_k))}"
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            metrics['cache_hits'].labels(cache_type='vector_search').inc()
            return SuccessResponse(
                message="Vector search completed (cached)",
                data=json.loads(cached_result)
            )
        
        # Perform search
        results = await integration_master.vector_search(request.query, request.top_k)
        
        # Cache result
        await redis_client.setex(cache_key, 1800, json.dumps(results))  # Cache for 30 minutes
        
        duration = time.time() - start_time
        
        return SuccessResponse(
            message="Vector search completed",
            data={
                "results": results,
                "query": request.query,
                "top_k": request.top_k,
                "similarity_threshold": request.similarity_threshold,
                "processing_time": duration
            }
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Vector search failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="VECTOR_SEARCH_FAILED",
            message=f"Vector search failed: {str(e)}"
        )

@app.post("/api/v1/batch/process", response_model=SuccessResponse)
async def batch_process(
    request: BatchProcessingRequest,
    integration_master: IntegrationMaster = Depends(get_integration_master),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Process items in batches"""
    start_time = time.time()
    
    try:
        # Define processor function based on operation type
        async def processor_func(item) -> Any:
            if request.operation_type == "text":
                text = str(item.get('text', item))
                return await integration_master.process_text(text, ["statistics", "sentiment"])
            else:
                return {"processed": item, "type": request.operation_type}
        
        # Process batches
        results = await integration_master.batch_process(
            request.items, 
            processor_func, 
            request.batch_size
        )
        
        duration = time.time() - start_time
        
        return SuccessResponse(
            message="Batch processing completed",
            data={
                "results": results,
                "total_items": len(request.items),
                "batch_size": request.batch_size,
                "operation_type": request.operation_type,
                "parallel": request.parallel,
                "processing_time": duration
            }
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Batch processing failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="BATCH_PROCESSING_FAILED",
            message=f"Batch processing failed: {str(e)}"
        )

@app.post("/api/v1/optimize/performance", response_model=SuccessResponse)
async def optimize_performance(
    request: PerformanceOptimizationRequest,
    integration_master: IntegrationMaster = Depends(get_integration_master),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Optimize performance for specific tasks"""
    start_time = time.time()
    
    try:
        # Perform optimization
        results = await integration_master.optimize_performance(
            request.task_type, 
            **request.parameters
        )
        
        duration = time.time() - start_time
        
        return SuccessResponse(
            message="Performance optimization completed",
            data={
                "results": results,
                "task_type": request.task_type,
                "target_metric": request.target_metric,
                "constraints": request.constraints,
                "processing_time": duration
            }
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Performance optimization failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="PERFORMANCE_OPTIMIZATION_FAILED",
            message=f"Performance optimization failed: {str(e)}"
        )

@app.get("/api/v1/system/info", response_model=SuccessResponse)
async def get_system_info(
    integration_master: IntegrationMaster = Depends(get_integration_master),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> SuccessResponse:
    """Get comprehensive system information"""
    try:
        system_info = integration_master.get_system_info()
        
        return SuccessResponse(
            message="System information retrieved",
            data=system_info
        )
        
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Failed to get system info: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="SYSTEM_INFO_FAILED",
            message=f"Failed to get system info: {str(e)}"
        )

# Admin routes (protected)
@app.post("/api/v1/admin/cache/clear")
async def clear_cache(
    redis_client: aioredis.Redis = Depends(get_redis_client),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> SuccessResponse:
    """Clear all cache (admin only)"""
    try:
        # Check admin permissions
        if current_user.get('role') != 'admin':
            raise APIException(
                status_code=status.HTTP_403_FORBIDDEN,
                error_code="INSUFFICIENT_PERMISSIONS",
                message="Admin permissions required"
            )
        
        # Clear cache
        await redis_client.flushdb()
        
        return SuccessResponse(
            message="Cache cleared successfully",
            data={"cleared_at": datetime.utcnow().isoformat()}
        )
        
    except APIException:
        raise
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(f"Cache clear failed: {e}")
        raise APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CACHE_CLEAR_FAILED",
            message=f"Cache clear failed: {str(e)}"
        )

if __name__ == "__main__":
    
    config = get_config()
    
    uvicorn.run(
        "enhanced_api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level="info",
        reload=config.api.reload
    ) 