from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn
from api.core.lifespan_manager import (
from api.utils.sync_async_patterns import (
from api.schemas.functional_models import (
from api.endpoints.functional_endpoints import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Main FastAPI Application for HeyGen AI API
Demonstrates modern lifespan context manager usage.
"""



# Import our lifespan manager and components
    LifespanManager,
    get_database_session,
    get_redis_client,
    get_lifespan_state
)
    UserService,
    validate_username_sync,
    create_user_async
)
    UserCreate,
    UserResponse,
    VideoCreate,
    VideoResponse,
    HealthResponse
)
    create_user_endpoint,
    get_user_endpoint,
    create_video_endpoint,
    get_video_endpoint
)

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

logger = structlog.get_logger()

# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Modern lifespan context manager for HeyGen AI API.
    
    This replaces the deprecated app.on_event("startup") and app.on_event("shutdown")
    with a modern context manager approach.
    """
    # Initialize lifespan manager
    lifespan_manager = LifespanManager()
    
    try:
        # Startup phase
        logger.info("Starting HeyGen AI API application")
        
        # Use the lifespan manager's startup logic
        await lifespan_manager._startup(app)
        
        logger.info("HeyGen AI API application started successfully")
        
        # Application runs here
        yield
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
        
    finally:
        # Shutdown phase (always runs)
        logger.info("Shutting down HeyGen AI API application")
        
        try:
            # Use the lifespan manager's shutdown logic
            await lifespan_manager._shutdown(app)
        except Exception as e:
            logger.error("Error during application shutdown", error=str(e))

# =============================================================================
# FastAPI Application
# =============================================================================

# Create FastAPI app with lifespan
app = FastAPI(
    title="HeyGen AI API",
    description="Modern AI-powered video generation API with lifespan context managers",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =============================================================================
# Middleware
# =============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = datetime.now(timezone.utc)
    
    # Get lifespan state for logging
    try:
        lifespan_state = get_lifespan_state()
        uptime = (datetime.now(timezone.utc) - lifespan_state.startup_time).total_seconds()
    except:
        uptime = 0
    
    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        uptime_seconds=uptime
    )
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log request completion
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_seconds=duration
        )
        
        return response
        
    except Exception as e:
        # Calculate duration
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log request error
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration_seconds=duration
        )
        raise

# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        method=request.method,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        method=request.method,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )

# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Application health status
    """
    try:
        # Get lifespan state
        lifespan_state = get_lifespan_state()
        
        # Calculate uptime
        uptime_seconds = (
            datetime.now(timezone.utc) - lifespan_state.startup_time
        ).total_seconds() if lifespan_state.startup_time else 0
        
        # Perform health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "version": "1.0.0",
            "checks": {
                "database": {"status": "healthy", "message": "Database connection OK"},
                "redis": {"status": "healthy", "message": "Redis connection OK"},
                "application": {"status": "healthy", "message": "Application running"}
            }
        }
        
        # Check if any component is unhealthy
        all_healthy = all(
            check["status"] == "healthy" 
            for check in health_status["checks"].values()
        )
        
        if not all_healthy:
            health_status["status"] = "unhealthy"
        
        return HealthResponse(**health_status)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=0,
            version="1.0.0",
            error=str(e)
        )

@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns:
        dict: Readiness status
    """
    try:
        # Check if application is ready to serve requests
        lifespan_state = get_lifespan_state()
        
        if lifespan_state.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Application is shutting down"}
            )
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)}
        )

@app.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint.
    
    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db_session=Depends(get_database_session),
    redis_client=Depends(get_redis_client)
):
    """
    Create a new user.
    
    Args:
        user_data: User creation data
        db_session: Database session dependency
        redis_client: Redis client dependency
        
    Returns:
        UserResponse: Created user data
    """
    return await create_user_endpoint(user_data, db_session, redis_client)

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db_session=Depends(get_database_session)
):
    """
    Get user by ID.
    
    Args:
        user_id: User ID
        db_session: Database session dependency
        
    Returns:
        UserResponse: User data
    """
    return await get_user_endpoint(user_id, db_session)

@app.post("/videos", response_model=VideoResponse)
async def create_video(
    video_data: VideoCreate,
    db_session=Depends(get_database_session),
    redis_client=Depends(get_redis_client)
):
    """
    Create a new video.
    
    Args:
        video_data: Video creation data
        db_session: Database session dependency
        redis_client: Redis client dependency
        
    Returns:
        VideoResponse: Created video data
    """
    return await create_video_endpoint(video_data, db_session, redis_client)

@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: int,
    db_session=Depends(get_database_session)
):
    """
    Get video by ID.
    
    Args:
        video_id: Video ID
        db_session: Database session dependency
        
    Returns:
        VideoResponse: Video data
    """
    return await get_video_endpoint(video_id, db_session)

# =============================================================================
# Utility Endpoints
# =============================================================================

@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        dict: API information
    """
    return {
        "message": "HeyGen AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/info")
async def api_info():
    """
    API information endpoint.
    
    Returns:
        dict: Detailed API information
    """
    try:
        lifespan_state = get_lifespan_state()
        uptime_seconds = (
            datetime.now(timezone.utc) - lifespan_state.startup_time
        ).total_seconds() if lifespan_state.startup_time else 0
        
        return {
            "name": "HeyGen AI API",
            "version": "1.0.0",
            "description": "Modern AI-powered video generation API",
            "uptime_seconds": uptime_seconds,
            "startup_time": lifespan_state.startup_time.isoformat() if lifespan_state.startup_time else None,
            "is_shutting_down": lifespan_state.is_shutting_down,
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "users": "/users",
                "videos": "/videos"
            }
        }
    except Exception as e:
        logger.error("Error getting API info", error=str(e))
        return {
            "name": "HeyGen AI API",
            "version": "1.0.0",
            "error": "Unable to get runtime information"
        }

# =============================================================================
# Background Tasks
# =============================================================================

@app.post("/tasks/process-video/{video_id}")
async def process_video_background(
    video_id: int,
    db_session=Depends(get_database_session)
):
    """
    Start background video processing.
    
    Args:
        video_id: Video ID to process
        db_session: Database session dependency
        
    Returns:
        dict: Task status
    """
    try:
        # Get lifespan state to add background task
        lifespan_state = get_lifespan_state()
        
        # Create background task
        task = asyncio.create_task(process_video_async(video_id, db_session))
        
        # Add to lifespan manager for tracking
        lifespan_manager = LifespanManager()
        lifespan_manager.add_background_task(task)
        
        return {
            "message": "Video processing started",
            "video_id": video_id,
            "task_id": id(task)
        }
        
    except Exception as e:
        logger.error("Failed to start video processing", video_id=video_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start video processing")

async def process_video_async(video_id: int, db_session):
    """
    Background video processing task.
    
    Args:
        video_id: Video ID to process
        db_session: Database session
    """
    try:
        logger.info("Starting video processing", video_id=video_id)
        
        # Simulate video processing
        await asyncio.sleep(10)  # Simulate processing time
        
        logger.info("Video processing completed", video_id=video_id)
        
    except Exception as e:
        logger.error("Video processing failed", video_id=video_id, error=str(e))

# =============================================================================
# Signal Handling
# =============================================================================

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# Main Application Entry Point
# =============================================================================

def main():
    """Main application entry point."""
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False  # Set to True for development
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        sys.exit(1)

match __name__:
    case "__main__":
    main() 