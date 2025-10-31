from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.core.async_database import (
from api.core.repositories import (
from api.core.migrations import MigrationManager, CommonMigrations
from config.database_config import get_database_config_by_environment
        import uuid
    import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Async Database System Example for HeyGen AI API
Demonstrates integration with FastAPI, repositories, and health monitoring
"""



# Import our async database system
    AsyncDatabaseManager,
    DatabaseConnectionPool,
    get_db_session,
    check_database_health
)
    UserRepository,
    VideoRepository,
    ModelUsageRepository,
    RepositoryFactory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: str
    api_key: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

class VideoCreate(BaseModel):
    script: str
    voice_id: str
    language: str = "en"
    quality: str = "medium"

class VideoResponse(BaseModel):
    id: int
    video_id: str
    script: str
    voice_id: str
    language: str
    quality: str
    status: str
    created_at: datetime

# FastAPI application
app = FastAPI(title="HeyGen AI API - Async Database Example")

# Global database manager
db_manager: Optional[AsyncDatabaseManager] = None
db_pool: Optional[DatabaseConnectionPool] = None
repository_factory: Optional[RepositoryFactory] = None


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    global db_manager, db_pool, repository_factory
    
    logger.info("🚀 Starting HeyGen AI API with async database...")
    
    try:
        # Get database configuration based on environment
        environment = "development"  # Change based on your environment
        config = get_database_config_by_environment(environment)
        
        # Initialize database manager
        db_manager = AsyncDatabaseManager(config)
        await db_manager.initialize()
        
        # Setup migrations
        migration_manager = MigrationManager(db_manager)
        await create_initial_schema(migration_manager)
        
        # Initialize repository factory
        repository_factory = RepositoryFactory(db_pool if db_pool else db_manager)
        
        logger.info("✅ Database initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup database on shutdown."""
    global db_manager, db_pool
    
    logger.info("🛑 Shutting down database connections...")
    
    if db_manager:
        await db_manager.close()
    
    if db_pool:
        await db_pool.close_all()
    
    logger.info("✅ Database connections closed")


async def create_initial_schema(migration_manager: MigrationManager):
    """Create initial database schema."""
    logger.info("Creating initial database schema...")
    
    # Create users table
    users_migration = migration_manager.create_migration(
        name="create_users_table",
        description="Create users table",
        sql_up=CommonMigrations.create_users_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Create videos table
    videos_migration = migration_manager.create_migration(
        name="create_videos_table",
        description="Create videos table",
        sql_up=CommonMigrations.create_videos_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Create model usage table
    usage_migration = migration_manager.create_migration(
        name="create_model_usage_table",
        description="Create model usage table",
        sql_up=CommonMigrations.create_model_usage_table(),
        sql_down=CommonMigrations.drop_tables()
    )
    
    # Run migrations
    result = await migration_manager.migrate()
    
    if result["success"]:
        logger.info(f"✓ Schema created: {result['executed']} migrations")
    else:
        logger.error(f"✗ Schema creation failed: {result['failed']} migrations")
        raise RuntimeError("Failed to create database schema")


# Dependency to get repositories
async def get_user_repository() -> UserRepository:
    """Get user repository instance."""
    if not repository_factory:
        raise HTTPException(status_code=500, detail="Repository factory not initialized")
    return repository_factory.get_user_repository()


async def get_video_repository() -> VideoRepository:
    """Get video repository instance."""
    if not repository_factory:
        raise HTTPException(status_code=500, detail="Repository factory not initialized")
    return repository_factory.get_video_repository()


async def get_usage_repository() -> ModelUsageRepository:
    """Get model usage repository instance."""
    if not repository_factory:
        raise HTTPException(status_code=500, detail="Repository factory not initialized")
    return repository_factory.get_model_usage_repository()


# Health check endpoints
@app.get("/health/database")
async def database_health():
    """Check database health status."""
    if not db_manager:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Database not initialized"}
        )
    
    try:
        is_healthy = await db_manager.is_healthy()
        stats = await db_manager.get_connection_stats()
        info = await db_manager.get_database_info()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database": {
                "type": info.get("type"),
                "version": info.get("version"),
                "is_healthy": is_healthy,
                "connection_stats": {
                    "total": stats.total_connections,
                    "active": stats.active_connections,
                    "idle": stats.idle_connections,
                    "overflow": stats.overflow_connections
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/health/database/stats")
async def database_stats():
    """Get detailed database statistics."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        stats = await db_manager.get_connection_stats()
        info = await db_manager.get_database_info()
        
        return {
            "connection_stats": {
                "total_connections": stats.total_connections,
                "active_connections": stats.active_connections,
                "idle_connections": stats.idle_connections,
                "overflow_connections": stats.overflow_connections,
                "checked_out_connections": stats.checked_out_connections,
                "checked_in_connections": stats.checked_in_connections,
                "invalid_connections": stats.invalid_connections,
                "last_health_check": stats.last_health_check.isoformat() if stats.last_health_check else None,
                "health_check_status": stats.health_check_status
            },
            "database_info": info
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User endpoints
@app.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """Create a new user."""
    try:
        # Check if user already exists
        existing_user = await user_repo.get_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )
        
        existing_user = await user_repo.get_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already exists"
            )
        
        # Create user
        user = await user_repo.create(
            username=user_data.username,
            email=user_data.email,
            api_key=user_data.api_key,
            is_active=True
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """Get user by ID."""
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user")


@app.get("/users/{user_id}/stats")
async def get_user_stats(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """Get user statistics."""
    try:
        stats = await user_repo.get_user_stats(user_id)
        if not stats:
            raise HTTPException(status_code=404, detail="User not found")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user stats {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user stats")


# Video endpoints
@app.post("/videos", response_model=VideoResponse)
async def create_video(
    video_data: VideoCreate,
    user_id: int = 1,  # In real app, get from authentication
    video_repo: VideoRepository = Depends(get_video_repository)
):
    """Create a new video generation request."""
    try:
        
        video_id = f"video_{uuid.uuid4().hex[:8]}"
        
        video = await video_repo.create(
            video_id=video_id,
            user_id=user_id,
            script=video_data.script,
            voice_id=video_data.voice_id,
            language=video_data.language,
            quality=video_data.quality,
            status="processing"
        )
        
        return VideoResponse(
            id=video.id,
            video_id=video.video_id,
            script=video.script,
            voice_id=video.voice_id,
            language=video.language,
            quality=video.quality,
            status=video.status,
            created_at=video.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        raise HTTPException(status_code=500, detail="Failed to create video")


@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    video_repo: VideoRepository = Depends(get_video_repository)
):
    """Get video by video_id."""
    try:
        video = await video_repo.get_by_video_id(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return VideoResponse(
            id=video.id,
            video_id=video.video_id,
            script=video.script,
            voice_id=video.voice_id,
            language=video.language,
            quality=video.quality,
            status=video.status,
            created_at=video.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get video")


@app.put("/videos/{video_id}/status")
async def update_video_status(
    video_id: str,
    status: str,
    video_repo: VideoRepository = Depends(get_video_repository)
):
    """Update video status."""
    try:
        success = await video_repo.update_status(video_id, status)
        if not success:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {"message": f"Video status updated to {status}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update video status {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update video status")


@app.get("/videos/user/{user_id}")
async def get_user_videos(
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    video_repo: VideoRepository = Depends(get_video_repository)
):
    """Get videos for a specific user."""
    try:
        videos = await video_repo.get_user_videos(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status
        )
        
        return [
            VideoResponse(
                id=video.id,
                video_id=video.video_id,
                script=video.script,
                voice_id=video.voice_id,
                language=video.language,
                quality=video.quality,
                status=video.status,
                created_at=video.created_at
            )
            for video in videos
        ]
        
    except Exception as e:
        logger.error(f"Failed to get user videos {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user videos")


# Analytics endpoints
@app.post("/analytics/usage")
async def log_model_usage(
    user_id: int,
    video_id: int,
    model_type: str,
    processing_time: float,
    memory_usage: Optional[float] = None,
    gpu_usage: Optional[float] = None,
    usage_repo: ModelUsageRepository = Depends(get_usage_repository)
):
    """Log model usage for analytics."""
    try:
        usage = await usage_repo.log_usage({
            "user_id": user_id,
            "video_id": video_id,
            "model_type": model_type,
            "processing_time": processing_time,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage
        })
        
        return {"message": "Usage logged successfully", "usage_id": usage.id}
        
    except Exception as e:
        logger.error(f"Failed to log usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to log usage")


@app.get("/analytics/usage/stats")
async def get_usage_stats(
    user_id: Optional[int] = None,
    days: int = 30,
    model_type: Optional[str] = None,
    usage_repo: ModelUsageRepository = Depends(get_usage_repository)
):
    """Get usage statistics."""
    try:
        stats = await usage_repo.get_usage_stats(
            user_id=user_id,
            days=days,
            model_type=model_type
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage stats")


@app.get("/analytics/usage/daily")
async def get_daily_usage(
    days: int = 30,
    usage_repo: ModelUsageRepository = Depends(get_usage_repository)
):
    """Get daily usage statistics."""
    try:
        daily_usage = await usage_repo.get_daily_usage(days=days)
        return daily_usage
        
    except Exception as e:
        logger.error(f"Failed to get daily usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to get daily usage")


# System endpoints
@app.get("/system/database/info")
async def get_database_info():
    """Get database information."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        info = await db_manager.get_database_info()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database info")


@app.get("/system/database/migrations")
async def get_migration_status():
    """Get migration status."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        migration_manager = MigrationManager(db_manager)
        status = await migration_manager.get_migration_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get migration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get migration status")


# Example usage function
async def create_sample_data():
    """Create sample data for testing."""
    try:
        user_repo = await get_user_repository()
        video_repo = await get_video_repository()
        usage_repo = await get_usage_repository()
        
        # Create sample user
        user = await user_repo.create(
            username="demo_user",
            email="demo@example.com",
            api_key="demo_api_key_12345",
            is_active=True
        )
        
        # Create sample video
        video = await video_repo.create(
            video_id="demo_video_001",
            user_id=user.id,
            script="This is a demo video script for testing the async database system.",
            voice_id="voice_001",
            language="en",
            quality="medium",
            status="completed"
        )
        
        # Log sample usage
        await usage_repo.log_usage({
            "user_id": user.id,
            "video_id": video.id,
            "model_type": "transformer",
            "processing_time": 45.2,
            "memory_usage": 1024.5,
            "gpu_usage": 85.3
        })
        
        logger.info("✓ Sample data created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")


if __name__ == "__main__":
    
    # Run the application
    uvicorn.run(
        "async_database_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 