from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any, Annotated
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import (
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict, validator
from pydantic_settings import BaseSettings
import structlog
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .sqlalchemy_2_implementation import (
        from httpx import AsyncClient
from typing import Any, List, Dict, Optional
"""
🚀 FastAPI-Specific Guidelines for SQLAlchemy 2.0
================================================

Comprehensive FastAPI guidelines with:
- Dependency injection patterns
- Request/response models
- Error handling
- Middleware configuration
- Performance optimization
- Security best practices
- Testing strategies
- Production deployment
"""


    FastAPI, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header, Cookie, Form, File,
    UploadFile, APIRouter, middleware, Middleware
)

    DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, TextAnalysisUpdate, BatchAnalysisCreate,
    TextAnalysisResponse, BatchAnalysisResponse,
    AnalysisType, AnalysisStatus, OptimizationTier,
    DatabaseError, ValidationError
)

# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/nlp_db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    database_enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    # API
    api_title: str = Field(default="Blatam Academy NLP API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_description: str = Field(default="NLP Analysis API with SQLAlchemy 2.0", env="API_DESCRIPTION")
    
    # Security
    secret_key: str = Field(default="your-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Performance
    enable_gzip: bool = Field(default=True, env="ENABLE_GZIP")
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    
    model_config = ConfigDict(env_file=".env", case_sensitive=False)


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Current timestamp")
    version: str = Field(description="API version")
    database: Dict[str, Any] = Field(description="Database health status")
    performance: Dict[str, Any] = Field(description="Performance metrics")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: datetime = Field(description="Error timestamp")
    request_id: Optional[str] = Field(description="Request ID for tracking")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    order_by: Optional[str] = Field(default="created_at", description="Order by field")
    order_desc: bool = Field(default=True, description="Descending order")


class AnalysisFilters(BaseModel):
    """Analysis filtering parameters."""
    analysis_type: Optional[AnalysisType] = Field(description="Analysis type filter")
    status: Optional[AnalysisStatus] = Field(description="Status filter")
    optimization_tier: Optional[OptimizationTier] = Field(description="Optimization tier filter")
    date_from: Optional[datetime] = Field(description="Start date filter")
    date_to: Optional[datetime] = Field(description="End date filter")


class BatchCreateRequest(BaseModel):
    """Batch creation request with multiple texts."""
    batch_name: str = Field(..., min_length=1, max_length=200, description="Batch name")
    analysis_type: AnalysisType = Field(description="Analysis type")
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="Texts to analyze")
    optimization_tier: OptimizationTier = Field(
        default=OptimizationTier.STANDARD, 
        description="Optimization tier"
    )
    
    @validator('texts')
    def validate_texts(cls, v) -> bool:
        """Validate text lengths."""
        for text in v:
            if len(text) > 10000:
                raise ValueError("Text content too long (max 10000 characters)")
        return v


class AnalysisUpdateRequest(BaseModel):
    """Analysis update request."""
    status: Optional[AnalysisStatus] = Field(description="Analysis status")
    sentiment_score: Optional[float] = Field(ge=-1.0, le=1.0, description="Sentiment score")
    quality_score: Optional[float] = Field(ge=0.0, le=1.0, description="Quality score")
    processing_time_ms: Optional[float] = Field(ge=0.0, description="Processing time")
    model_used: Optional[str] = Field(description="Model used for analysis")
    error_message: Optional[str] = Field(description="Error message if failed")


# ============================================================================
# Dependency Injection
# ============================================================================

class DatabaseDependency:
    """Database dependency manager."""
    
    def __init__(self) -> Any:
        self.settings = Settings()
        self._db_manager: Optional[SQLAlchemy2Manager] = None
    
    async def get_db_manager(self) -> SQLAlchemy2Manager:
        """Get database manager instance."""
        if self._db_manager is None:
            config = DatabaseConfig(
                url=self.settings.database_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                enable_caching=self.settings.database_enable_caching
            )
            self._db_manager = SQLAlchemy2Manager(config)
            await self._db_manager.initialize()
        
        return self._db_manager
    
    async def cleanup(self) -> Any:
        """Cleanup database manager."""
        if self._db_manager:
            await self._db_manager.cleanup()
            self._db_manager = None


# Global database dependency
db_dependency = DatabaseDependency()


async def get_db() -> SQLAlchemy2Manager:
    """Database dependency for FastAPI."""
    return await db_dependency.get_db_manager()


# Type alias for dependency injection
DB = Annotated[SQLAlchemy2Manager, Depends(get_db)]


# ============================================================================
# Middleware and Security
# ============================================================================

class SecurityMiddleware:
    """Security middleware for request validation."""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        # Add security headers
        if scope["type"] == "http":
            async def send_with_headers(message) -> Any:
                if message["type"] == "http.response.start":
                    message["headers"].extend([
                        (b"X-Content-Type-Options", b"nosniff"),
                        (b"X-Frame-Options", b"DENY"),
                        (b"X-XSS-Protection", b"1; mode=block"),
                        (b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains"),
                        (b"Content-Security-Policy", b"default-src 'self'"),
                    ])
                await send(message)
            
            await self.app(scope, receive, send_with_headers)
        else:
            await self.app(scope, receive, send)


class RequestLoggingMiddleware:
    """Request logging middleware."""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.logger = structlog.get_logger(__name__)
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = datetime.now()
            request_id = f"req_{start_time.timestamp()}"
            
            # Log request
            self.logger.info(
                "Request started",
                request_id=request_id,
                method=scope["method"],
                path=scope["path"],
                client=scope.get("client", ("unknown", 0))[0]
            )
            
            async def send_with_logging(message) -> Any:
                if message["type"] == "http.response.start":
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    self.logger.info(
                        "Request completed",
                        request_id=request_id,
                        status_code=message["status"],
                        duration=duration
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)


# ============================================================================
# Error Handling
# ============================================================================

class CustomHTTPException(HTTPException):
    """Custom HTTP exception with additional context."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        
    """__init__ function."""
super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.request_id = request_id


async def http_exception_handler(request: Request, exc: CustomHTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=exc.error_code,
            timestamp=datetime.now(),
            request_id=exc.request_id
        ).model_dump()
    )


async def database_exception_handler(request: Request, exc: DatabaseError):
    """Database exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Database operation failed",
            detail=str(exc),
            timestamp=datetime.now(),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump()
    )


async def validation_exception_handler(request: Request, exc: ValidationError):
    """Validation exception handler."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            timestamp=datetime.now(),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump()
    )


# ============================================================================
# Rate Limiting
# ============================================================================

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Rate limit exceeded handler."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse(
            error="Rate limit exceeded",
            detail=f"Too many requests. Try again in {exc.retry_after} seconds.",
            timestamp=datetime.now(),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump(),
        headers={"Retry-After": str(exc.retry_after)}
    )


# ============================================================================
# API Routes
# ============================================================================

# Create API router
api_router = APIRouter(prefix="/api/v1", tags=["NLP Analysis"])


@api_router.get("/health", response_model=HealthResponse)
async def health_check(db: DB):
    """Health check endpoint."""
    try:
        # Check database health
        health = await db.health_check()
        metrics = await db.get_performance_metrics()
        
        return HealthResponse(
            status="healthy" if health.is_healthy else "unhealthy",
            timestamp=datetime.now(),
            version=Settings().api_version,
            database={
                "is_healthy": health.is_healthy,
                "connection_count": health.connection_count,
                "pool_size": health.pool_size,
                "avg_query_time": health.avg_query_time
            },
            performance=metrics
        )
    except Exception as e:
        raise CustomHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy",
            error_code=str(e)
        )


@api_router.post("/analyses", response_model=TextAnalysisResponse)
@limiter.limit("100/minute")
async def create_analysis(
    request: Request,
    data: TextAnalysisCreate,
    db: DB,
    background_tasks: BackgroundTasks
):
    """Create a new text analysis."""
    try:
        # Create analysis
        analysis = await db.create_text_analysis(data)
        
        # Add background task for processing (if needed)
        # background_tasks.add_task(process_analysis, analysis.id)
        
        return analysis
    except ValidationError as e:
        raise CustomHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data",
            error_code=str(e)
        )
    except DatabaseError as e:
        raise CustomHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
            error_code=str(e)
        )


@api_router.get("/analyses/{analysis_id}", response_model=TextAnalysisResponse)
@limiter.limit("200/minute")
async def get_analysis(
    analysis_id: int = Path(..., description="Analysis ID"),
    db: DB = Depends(get_db)
):
    """Get analysis by ID."""
    analysis = await db.get_text_analysis(analysis_id)
    if not analysis:
        raise CustomHTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    return analysis


@api_router.put("/analyses/{analysis_id}", response_model=TextAnalysisResponse)
@limiter.limit("50/minute")
async def update_analysis(
    analysis_id: int = Path(..., description="Analysis ID"),
    data: AnalysisUpdateRequest = Body(...),
    db: DB = Depends(get_db)
):
    """Update analysis results."""
    try:
        # Convert to TextAnalysisUpdate
        update_data = TextAnalysisUpdate(**data.model_dump(exclude_unset=True))
        
        analysis = await db.update_text_analysis(analysis_id, update_data)
        if not analysis:
            raise CustomHTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        return analysis
    except ValidationError as e:
        raise CustomHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid update data",
            error_code=str(e)
        )


@api_router.get("/analyses", response_model=List[TextAnalysisResponse])
@limiter.limit("100/minute")
async def list_analyses(
    pagination: PaginationParams = Depends(),
    filters: AnalysisFilters = Depends(),
    db: DB = Depends(get_db)
):
    """List analyses with filtering and pagination."""
    try:
        # Calculate offset
        offset = (pagination.page - 1) * pagination.size
        
        # Build filters
        filter_params = {}
        if filters.analysis_type:
            filter_params["analysis_type"] = filters.analysis_type
        if filters.status:
            filter_params["status"] = filters.status
        if filters.optimization_tier:
            filter_params["optimization_tier"] = filters.optimization_tier
        
        analyses = await db.list_text_analyses(
            limit=pagination.size,
            offset=offset,
            order_by=pagination.order_by,
            order_desc=pagination.order_desc,
            **filter_params
        )
        
        return analyses
    except Exception as e:
        raise CustomHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analyses",
            error_code=str(e)
        )


@api_router.delete("/analyses/{analysis_id}")
@limiter.limit("20/minute")
async def delete_analysis(
    analysis_id: int = Path(..., description="Analysis ID"),
    db: DB = Depends(get_db)
):
    """Delete analysis by ID."""
    success = await db.delete_text_analysis(analysis_id)
    if not success:
        raise CustomHTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    return {"message": "Analysis deleted successfully"}


@api_router.post("/batches", response_model=BatchAnalysisResponse)
@limiter.limit("10/minute")
async def create_batch(
    request: BatchCreateRequest,
    db: DB = Depends(get_db),
    background_tasks: BackgroundTasks
):
    """Create batch analysis with multiple texts."""
    try:
        # Create batch
        batch_data = BatchAnalysisCreate(
            batch_name=request.batch_name,
            analysis_type=request.analysis_type,
            optimization_tier=request.optimization_tier
        )
        batch = await db.create_batch_analysis(batch_data)
        
        # Add background task to process texts
        background_tasks.add_task(
            process_batch_texts,
            batch.id,
            request.texts,
            request.analysis_type,
            db
        )
        
        return batch
    except Exception as e:
        raise CustomHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch",
            error_code=str(e)
        )


@api_router.get("/batches/{batch_id}", response_model=BatchAnalysisResponse)
@limiter.limit("50/minute")
async def get_batch(
    batch_id: int = Path(..., description="Batch ID"),
    db: DB = Depends(get_db)
):
    """Get batch analysis by ID."""
    batch = await db.get_batch_analysis(batch_id)
    if not batch:
        raise CustomHTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    return batch


@api_router.get("/batches", response_model=List[BatchAnalysisResponse])
@limiter.limit("50/minute")
async def list_batches(
    pagination: PaginationParams = Depends(),
    analysis_type: Optional[AnalysisType] = Query(None, description="Filter by analysis type"),
    status: Optional[AnalysisStatus] = Query(None, description="Filter by status"),
    db: DB = Depends(get_db)
):
    """List batch analyses with filtering and pagination."""
    try:
        # This would need to be implemented in the database manager
        # For now, return empty list
        return []
    except Exception as e:
        raise CustomHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batches",
            error_code=str(e)
        )


@api_router.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(db: DB = Depends(get_db)):
    """Get performance metrics."""
    try:
        metrics = await db.get_performance_metrics()
        return metrics
    except Exception as e:
        raise CustomHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics",
            error_code=str(e)
        )


# ============================================================================
# Background Tasks
# ============================================================================

async def process_batch_texts(
    batch_id: int,
    texts: List[str],
    analysis_type: AnalysisType,
    db: SQLAlchemy2Manager
):
    """Background task to process batch texts."""
    try:
        completed_count = 0
        error_count = 0
        
        for text in texts:
            try:
                # Create analysis
                analysis_data = TextAnalysisCreate(
                    text_content=text,
                    analysis_type=analysis_type
                )
                analysis = await db.create_text_analysis(analysis_data)
                
                # Simulate processing (replace with actual NLP processing)
                await asyncio.sleep(0.1)
                
                # Update with results
                update_data = TextAnalysisUpdate(
                    status=AnalysisStatus.COMPLETED,
                    sentiment_score=0.5,  # Replace with actual result
                    processing_time_ms=100.0,
                    model_used="test-model"
                )
                await db.update_text_analysis(analysis.id, update_data)
                
                completed_count += 1
                
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing text in batch {batch_id}: {e}")
        
        # Update batch progress
        await db.update_batch_progress(batch_id, completed_count, error_count)
        
    except Exception as e:
        logging.error(f"Error in batch processing {batch_id}: {e}")


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    settings = Settings()
    
    # Configure logging
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
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        docs_url="/docs" if settings.log_level == "DEBUG" else None,
        redoc_url="/redoc" if settings.log_level == "DEBUG" else None,
        openapi_url="/openapi.json" if settings.log_level == "DEBUG" else None
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    if settings.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    
    # Add exception handlers
    app.add_exception_handler(CustomHTTPException, http_exception_handler)
    app.add_exception_handler(DatabaseError, database_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Include routers
    app.include_router(api_router)
    
    # Custom OpenAPI schema
    def custom_openapi():
        
    """custom_openapi function."""
if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.api_title,
            version=settings.api_version,
            description=settings.api_description,
            routes=app.routes,
        )
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logging.info("Starting FastAPI application...")
        
        # Initialize database
        await db_dependency.get_db_manager()
        
        logging.info("FastAPI application started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logging.info("Shutting down FastAPI application...")
        
        # Cleanup database
        await db_dependency.cleanup()
        
        logging.info("FastAPI application shut down successfully")
    
    return app


# ============================================================================
# Testing Utilities
# ============================================================================

class TestClient:
    """Test client for FastAPI application."""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.client = None
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        self.client = AsyncClient(app=self.app, base_url="http://test")
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()


# ============================================================================
# Production Configuration
# ============================================================================

def run_production():
    """Run application in production mode."""
    settings = Settings()
    
    uvicorn.run(
        "fastapi_sqlalchemy_2_guidelines:create_app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=False,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    # Create application
    app = create_app()
    
    # Run development server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 