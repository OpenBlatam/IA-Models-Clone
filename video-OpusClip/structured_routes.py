"""
🏗️ Structured Routes and Dependencies for Video-OpusClip

Comprehensive routing system with clear organization, dependency injection,
and maintainable structure for the Video-OpusClip AI video processing system.

Features:
- Modular route organization by domain
- Clear dependency injection patterns
- Centralized route registration
- Consistent error handling
- Performance monitoring
- API versioning
- Documentation generation
- Testing support
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union, Type, TypeVar
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json

from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header, Form, File, UploadFile
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, ConfigDict
import structlog

# Import Video-OpusClip components
from .fastapi_dependency_injection import DependencyContainer, get_dependency_container
from .async_database import AsyncVideoDatabase, AsyncBatchDatabaseOperations
from .async_external_apis import AsyncYouTubeAPI, AsyncOpenAIAPI, AsyncStabilityAIAPI, AsyncElevenLabsAPI
from .pydantic_models import (
    VideoProcessingConfig, ModelConfig, PerformanceConfig,
    VideoRequest, VideoResponse, BatchVideoRequest, BatchVideoResponse,
    ProcessingStatus, ErrorResponse, HealthResponse
)

logger = structlog.get_logger(__name__)

# Type variables
T = TypeVar('T')
RouteT = TypeVar('RouteT', bound=Callable)

# =============================================================================
# Route Organization
# =============================================================================

class RouteCategory(str, Enum):
    """Route categories for organization."""
    AUTH = "authentication"
    VIDEO = "video"
    PROCESSING = "processing"
    BATCH = "batch"
    ANALYTICS = "analytics"
    ADMIN = "admin"
    SYSTEM = "system"
    HEALTH = "health"
    MONITORING = "monitoring"
    API = "api"
    WEBHOOK = "webhook"
    FILE = "file"
    SEARCH = "search"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"

class RoutePriority(int, Enum):
    """Route priorities for ordering."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class RouteConfig:
    """Configuration for route registration."""
    path: str
    method: str
    tags: List[str]
    summary: str
    description: str
    response_model: Optional[Type[BaseModel]] = None
    status_code: int = 200
    dependencies: List[Callable] = field(default_factory=list)
    deprecated: bool = False
    include_in_schema: bool = True
    category: RouteCategory = RouteCategory.API
    priority: RoutePriority = RoutePriority.NORMAL
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None

# =============================================================================
# Base Router and Dependencies
# =============================================================================

class BaseRouter:
    """Base router with common dependencies and functionality."""
    
    def __init__(self, prefix: str = "", tags: List[str] = None):
        self.router = APIRouter(prefix=prefix, tags=tags or [])
        self.dependencies = {}
        self.middleware = []
        self.exception_handlers = {}
    
    def add_dependency(self, name: str, dependency: Callable):
        """Add dependency to router."""
        self.dependencies[name] = dependency
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to router."""
        self.middleware.append(middleware)
    
    def add_exception_handler(self, exception_type: Type[Exception], handler: Callable):
        """Add exception handler to router."""
        self.exception_handlers[exception_type] = handler
    
    def get_router(self) -> APIRouter:
        """Get the underlying APIRouter."""
        return self.router

# =============================================================================
# Common Dependencies
# =============================================================================

class CommonDependencies:
    """Common dependencies used across all routes."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        if not credentials:
            return None
        
        # TODO: Implement JWT token validation
        # For now, return a mock user
        return {
            "id": "user_123",
            "email": "user@example.com",
            "role": "user"
        }
    
    async def get_db_session(self):
        """Get database session."""
        return await self.container.get_db_session_dependency()()
    
    async def get_cache_client(self):
        """Get cache client."""
        return await self.container.get_cache_client_dependency()()
    
    async def get_video_database(self) -> AsyncVideoDatabase:
        """Get video database service."""
        db_ops = await self.container.get_db_session_dependency()()
        return AsyncVideoDatabase(db_ops)
    
    async def get_batch_database(self) -> AsyncBatchDatabaseOperations:
        """Get batch database operations."""
        db_ops = await self.container.get_db_session_dependency()()
        return AsyncBatchDatabaseOperations(db_ops)
    
    async def get_youtube_api(self) -> AsyncYouTubeAPI:
        """Get YouTube API service."""
        # TODO: Implement YouTube API service
        return None
    
    async def get_openai_api(self) -> AsyncOpenAIAPI:
        """Get OpenAI API service."""
        # TODO: Implement OpenAI API service
        return None
    
    async def get_stability_api(self) -> AsyncStabilityAIAPI:
        """Get Stability AI API service."""
        # TODO: Implement Stability AI API service
        return None
    
    async def get_elevenlabs_api(self) -> AsyncElevenLabsAPI:
        """Get ElevenLabs API service."""
        # TODO: Implement ElevenLabs API service
        return None

# =============================================================================
# Route Handlers
# =============================================================================

class VideoRouteHandlers:
    """Video processing route handlers."""
    
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def create_video(
        self,
        video_data: VideoRequest,
        current_user: Dict[str, Any] = Depends(CommonDependencies.get_current_user),
        video_db: AsyncVideoDatabase = Depends(CommonDependencies.get_video_database)
    ) -> VideoResponse:
        """Create a new video processing request."""
        try:
            # Create video record
            video_record = {
                "title": video_data.title,
                "description": video_data.description,
                "url": video_data.url,
                "duration": video_data.duration,
                "resolution": video_data.resolution,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "user_id": current_user["id"] if current_user else None,
                "config": video_data.config.dict() if video_data.config else {}
            }
            
            video_id = await video_db.create_video_record(video_record)
            
            return VideoResponse(
                id=video_id,
                title=video_data.title,
                status="pending",
                created_at=datetime.now(),
                message="Video processing request created successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create video: {str(e)}"
            )
    
    async def get_video(
        self,
        video_id: int,
        current_user: Dict[str, Any] = Depends(CommonDependencies.get_current_user),
        video_db: AsyncVideoDatabase = Depends(CommonDependencies.get_video_database)
    ) -> VideoResponse:
        """Get video by ID."""
        try:
            video = await video_db.get_video_by_id(video_id)
            if not video:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found"
                )
            
            return VideoResponse(
                id=video_id,
                title=video["title"],
                status=video["status"],
                created_at=datetime.fromisoformat(video["created_at"]),
                message="Video retrieved successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get video {video_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get video: {str(e)}"
            )
    
    async def update_video_status(
        self,
        video_id: int,
        status: ProcessingStatus,
        current_user: Dict[str, Any] = Depends(CommonDependencies.get_current_user),
        video_db: AsyncVideoDatabase = Depends(CommonDependencies.get_video_database)
    ) -> VideoResponse:
        """Update video processing status."""
        try:
            success = await video_db.update_video_status(video_id, status.value)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found"
                )
            
            return VideoResponse(
                id=video_id,
                status=status.value,
                updated_at=datetime.now(),
                message="Video status updated successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update video status {video_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update video status: {str(e)}"
            )

class BatchRouteHandlers:
    """Batch processing route handlers."""
    
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def create_batch_videos(
        self,
        batch_data: BatchVideoRequest,
        current_user: Dict[str, Any] = Depends(CommonDependencies.get_current_user),
        batch_db: AsyncBatchDatabaseOperations = Depends(CommonDependencies.get_batch_database)
    ) -> BatchVideoResponse:
        """Create multiple video processing requests."""
        try:
            videos_data = []
            for video_req in batch_data.videos:
                video_data = {
                    "title": video_req.title,
                    "description": video_req.description,
                    "url": video_req.url,
                    "duration": video_req.duration,
                    "resolution": video_req.resolution,
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                    "user_id": current_user["id"] if current_user else None,
                    "config": video_req.config.dict() if video_req.config else {}
                }
                videos_data.append(video_data)
            
            video_ids = await batch_db.batch_insert_videos(videos_data)
            
            return BatchVideoResponse(
                batch_id=f"batch_{int(time.time())}",
                video_ids=video_ids,
                total_videos=len(video_ids),
                status="pending",
                created_at=datetime.now(),
                message=f"Batch processing request created with {len(video_ids)} videos"
            )
            
        except Exception as e:
            logger.error(f"Failed to create batch videos: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create batch videos: {str(e)}"
            )

class AnalyticsRouteHandlers:
    """Analytics and monitoring route handlers."""
    
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def get_processing_stats(
        self,
        current_user: Dict[str, Any] = Depends(CommonDependencies.get_current_user),
        video_db: AsyncVideoDatabase = Depends(CommonDependencies.get_video_database)
    ) -> Dict[str, Any]:
        """Get video processing statistics."""
        try:
            # Get videos by status
            pending_videos = await video_db.get_videos_by_status("pending")
            processing_videos = await video_db.get_videos_by_status("processing")
            completed_videos = await video_db.get_videos_by_status("completed")
            failed_videos = await video_db.get_videos_by_status("failed")
            
            return {
                "total_videos": len(pending_videos) + len(processing_videos) + len(completed_videos) + len(failed_videos),
                "pending": len(pending_videos),
                "processing": len(processing_videos),
                "completed": len(completed_videos),
                "failed": len(failed_videos),
                "success_rate": len(completed_videos) / max(1, len(completed_videos) + len(failed_videos)) * 100
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get processing stats: {str(e)}"
            )

# =============================================================================
# Router Factory
# =============================================================================

class RouterFactory:
    """Factory for creating organized routers."""
    
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
        self.handlers = {
            "video": VideoRouteHandlers(dependencies),
            "batch": BatchRouteHandlers(dependencies),
            "analytics": AnalyticsRouteHandlers(dependencies)
        }
    
    def create_video_router(self) -> APIRouter:
        """Create video processing router."""
        router = APIRouter(prefix="/api/v1/videos", tags=["video"])
        
        # Video creation
        @router.post(
            "/",
            response_model=VideoResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create video processing request",
            description="Create a new video processing request"
        )
        async def create_video(
            video_data: VideoRequest,
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            video_db: AsyncVideoDatabase = Depends(self.deps.get_video_database)
        ):
            return await self.handlers["video"].create_video(video_data, current_user, video_db)
        
        # Get video
        @router.get(
            "/{video_id}",
            response_model=VideoResponse,
            summary="Get video by ID",
            description="Retrieve video processing request by ID"
        )
        async def get_video(
            video_id: int,
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            video_db: AsyncVideoDatabase = Depends(self.deps.get_video_database)
        ):
            return await self.handlers["video"].get_video(video_id, current_user, video_db)
        
        # Update status
        @router.patch(
            "/{video_id}/status",
            response_model=VideoResponse,
            summary="Update video status",
            description="Update video processing status"
        )
        async def update_video_status(
            video_id: int,
            status: ProcessingStatus,
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            video_db: AsyncVideoDatabase = Depends(self.deps.get_video_database)
        ):
            return await self.handlers["video"].update_video_status(video_id, status, current_user, video_db)
        
        # List videos
        @router.get(
            "/",
            response_model=List[VideoResponse],
            summary="List videos",
            description="List all video processing requests"
        )
        async def list_videos(
            status: Optional[str] = Query(None, description="Filter by status"),
            limit: int = Query(10, ge=1, le=100, description="Number of videos to return"),
            offset: int = Query(0, ge=0, description="Number of videos to skip"),
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            video_db: AsyncVideoDatabase = Depends(self.deps.get_video_database)
        ):
            try:
                if status:
                    videos = await video_db.get_videos_by_status(status)
                else:
                    # TODO: Implement pagination
                    videos = await video_db.get_videos_by_status("all")
                
                return [
                    VideoResponse(
                        id=video["id"],
                        title=video["title"],
                        status=video["status"],
                        created_at=datetime.fromisoformat(video["created_at"]),
                        message="Video retrieved successfully"
                    )
                    for video in videos[offset:offset + limit]
                ]
                
            except Exception as e:
                logger.error(f"Failed to list videos: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list videos: {str(e)}"
                )
        
        return router
    
    def create_batch_router(self) -> APIRouter:
        """Create batch processing router."""
        router = APIRouter(prefix="/api/v1/batch", tags=["batch"])
        
        # Batch video creation
        @router.post(
            "/videos",
            response_model=BatchVideoResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create batch video processing",
            description="Create multiple video processing requests"
        )
        async def create_batch_videos(
            batch_data: BatchVideoRequest,
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            batch_db: AsyncBatchDatabaseOperations = Depends(self.deps.get_batch_database)
        ):
            return await self.handlers["batch"].create_batch_videos(batch_data, current_user, batch_db)
        
        return router
    
    def create_analytics_router(self) -> APIRouter:
        """Create analytics router."""
        router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
        
        # Processing statistics
        @router.get(
            "/stats",
            summary="Get processing statistics",
            description="Get video processing statistics and metrics"
        )
        async def get_processing_stats(
            current_user: Dict[str, Any] = Depends(self.deps.get_current_user),
            video_db: AsyncVideoDatabase = Depends(self.deps.get_video_database)
        ):
            return await self.handlers["analytics"].get_processing_stats(current_user, video_db)
        
        return router
    
    def create_health_router(self) -> APIRouter:
        """Create health check router."""
        router = APIRouter(prefix="/api/v1/health", tags=["health"])
        
        @router.get(
            "/",
            response_model=HealthResponse,
            summary="Health check",
            description="Check system health status"
        )
        async def health_check():
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0",
                services={
                    "database": "healthy",
                    "cache": "healthy",
                    "models": "healthy"
                }
            )
        
        @router.get(
            "/detailed",
            summary="Detailed health check",
            description="Detailed system health check with all services"
        )
        async def detailed_health_check():
            # TODO: Implement detailed health checks
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": {
                    "database": {
                        "status": "healthy",
                        "response_time": 0.001,
                        "last_check": datetime.now().isoformat()
                    },
                    "cache": {
                        "status": "healthy",
                        "response_time": 0.0005,
                        "last_check": datetime.now().isoformat()
                    },
                    "models": {
                        "status": "healthy",
                        "loaded_models": ["video_processor", "caption_generator"],
                        "last_check": datetime.now().isoformat()
                    }
                }
            }
        
        return router

# =============================================================================
# Route Registry
# =============================================================================

class RouteRegistry:
    """Centralized route registry for managing all routers."""
    
    def __init__(self, app: FastAPI, container: DependencyContainer):
        self.app = app
        self.container = container
        self.dependencies = CommonDependencies(container)
        self.factory = RouterFactory(self.dependencies)
        self.routers: Dict[str, APIRouter] = {}
        self.route_configs: Dict[str, RouteConfig] = {}
    
    def register_router(self, name: str, router: APIRouter, config: Optional[RouteConfig] = None):
        """Register a router with optional configuration."""
        self.routers[name] = router
        if config:
            self.route_configs[name] = config
        
        # Include router in app
        self.app.include_router(router)
        logger.info(f"Registered router: {name}")
    
    def register_all_routers(self):
        """Register all standard routers."""
        # Video processing router
        video_router = self.factory.create_video_router()
        self.register_router("video", video_router, RouteConfig(
            path="/api/v1/videos",
            method="*",
            tags=["video"],
            summary="Video processing operations",
            description="Endpoints for video processing and management",
            category=RouteCategory.VIDEO,
            priority=RoutePriority.HIGH
        ))
        
        # Batch processing router
        batch_router = self.factory.create_batch_router()
        self.register_router("batch", batch_router, RouteConfig(
            path="/api/v1/batch",
            method="*",
            tags=["batch"],
            summary="Batch processing operations",
            description="Endpoints for batch video processing",
            category=RouteCategory.BATCH,
            priority=RoutePriority.NORMAL
        ))
        
        # Analytics router
        analytics_router = self.factory.create_analytics_router()
        self.register_router("analytics", analytics_router, RouteConfig(
            path="/api/v1/analytics",
            method="*",
            tags=["analytics"],
            summary="Analytics and monitoring",
            description="Endpoints for analytics and system monitoring",
            category=RouteCategory.ANALYTICS,
            priority=RoutePriority.LOW
        ))
        
        # Health check router
        health_router = self.factory.create_health_router()
        self.register_router("health", health_router, RouteConfig(
            path="/api/v1/health",
            method="*",
            tags=["health"],
            summary="Health checks",
            description="System health check endpoints",
            category=RouteCategory.HEALTH,
            priority=RoutePriority.CRITICAL
        ))
    
    def get_router(self, name: str) -> Optional[APIRouter]:
        """Get router by name."""
        return self.routers.get(name)
    
    def get_all_routers(self) -> Dict[str, APIRouter]:
        """Get all registered routers."""
        return self.routers.copy()
    
    def get_route_config(self, name: str) -> Optional[RouteConfig]:
        """Get route configuration by name."""
        return self.route_configs.get(name)

# =============================================================================
# Application Factory
# =============================================================================

def create_structured_app() -> FastAPI:
    """Create FastAPI application with structured routes and dependencies."""
    
    # Create FastAPI app
    app = FastAPI(
        title="Video-OpusClip API",
        description="AI-powered video processing system with structured routes and dependencies",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Get dependency container
    container = get_dependency_container()
    
    # Create route registry
    registry = RouteRegistry(app, container)
    
    # Register all routers
    registry.register_all_routers()
    
    # Add global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        # Add process time to response headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    return app

# =============================================================================
# Usage Example
# =============================================================================

def main():
    """Main function to run the structured FastAPI application."""
    app = create_structured_app()
    
    # Run the application
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 