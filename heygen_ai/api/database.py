from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Database models and operations for HeyGen AI API
SQLAlchemy-based database layer with async support.
"""


logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    """User model for authentication and user management."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    api_key = Column(String(255), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    videos = relationship("Video", back_populates="user")


class Video(Base):
    """Video model for storing video generation requests and results."""
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    script = Column(Text, nullable=False)
    voice_id = Column(String(50), nullable=False)
    language = Column(String(10), default="en")
    quality = Column(String(20), default="medium")
    status = Column(String(20), default="processing")  # processing, completed, failed
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="videos")


class ModelUsage(Base):
    """Model usage tracking for analytics and monitoring."""
    __tablename__ = "model_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # transformer, diffusion
    processing_time = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=True)
    gpu_usage = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatabaseManager:
    """Database manager for async operations."""
    
    def __init__(self, database_url: str):
        
    """__init__ function."""
self.database_url = database_url
        self.engine = None
        self.async_session = None
    
    async def initialize(self) -> Any:
        """Initialize database connection and create tables."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create async session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("✓ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self) -> Any:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("✓ Database connections closed")
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        if not self.async_session:
            raise RuntimeError("Database not initialized")
        return self.async_session()


class VideoRepository:
    """Repository for video-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        
    """__init__ function."""
self.db_manager = db_manager
    
    async def create_video(self, video_data: Dict[str, Any]) -> Video:
        """Create a new video record."""
        async with self.db_manager.get_session() as session:
            video = Video(**video_data)
            session.add(video)
            await session.commit()
            await session.refresh(video)
            return video
    
    async def get_video_by_id(self, video_id: str) -> Optional[Video]:
        """Get video by video_id."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(Video).filter(Video.video_id == video_id)
            )
            return result.scalar_one_or_none()
    
    async def update_video_status(self, video_id: str, status: str, **kwargs) -> bool:
        """Update video status and other fields."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(Video).filter(Video.video_id == video_id)
            )
            video = result.scalar_one_or_none()
            
            if not video:
                return False
            
            # Update fields
            video.status = status
            for key, value in kwargs.items():
                if hasattr(video, key):
                    setattr(video, key, value)
            
            await session.commit()
            return True
    
    async def get_user_videos(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Video]:
        """Get videos for a specific user."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(Video)
                .filter(Video.user_id == user_id)
                .order_by(Video.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    async def get_video_stats(self, user_id: int) -> Dict[str, Any]:
        """Get video statistics for a user."""
        async with self.db_manager.get_session() as session:
            # Total videos
            total_result = await session.execute(
                session.query(Video).filter(Video.user_id == user_id)
            )
            total_videos = len(total_result.scalars().all())
            
            # Completed videos
            completed_result = await session.execute(
                session.query(Video).filter(
                    Video.user_id == user_id,
                    Video.status == "completed"
                )
            )
            completed_videos = len(completed_result.scalars().all())
            
            # Failed videos
            failed_result = await session.execute(
                session.query(Video).filter(
                    Video.user_id == user_id,
                    Video.status == "failed"
                )
            )
            failed_videos = len(failed_result.scalars().all())
            
            # Average processing time
            avg_time_result = await session.execute(
                session.query(func.avg(Video.processing_time))
                .filter(
                    Video.user_id == user_id,
                    Video.status == "completed",
                    Video.processing_time.isnot(None)
                )
            )
            avg_processing_time = avg_time_result.scalar() or 0.0
            
            return {
                "total_videos": total_videos,
                "completed_videos": completed_videos,
                "failed_videos": failed_videos,
                "success_rate": (completed_videos / total_videos * 100) if total_videos > 0 else 0,
                "average_processing_time": avg_processing_time
            }


class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        
    """__init__ function."""
self.db_manager = db_manager
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        async with self.db_manager.get_session() as session:
            user = User(**user_data)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(User).filter(
                    User.api_key == api_key,
                    User.is_active == True
                )
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(User).filter(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user information."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                session.query(User).filter(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return False
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            await session.commit()
            return True


class ModelUsageRepository:
    """Repository for model usage tracking."""
    
    def __init__(self, db_manager: DatabaseManager):
        
    """__init__ function."""
self.db_manager = db_manager
    
    async def log_usage(self, usage_data: Dict[str, Any]) -> ModelUsage:
        """Log model usage."""
        async with self.db_manager.get_session() as session:
            usage = ModelUsage(**usage_data)
            session.add(usage)
            await session.commit()
            await session.refresh(usage)
            return usage
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        async with self.db_manager.get_session() as session:
            # Total processing time
            total_time_result = await session.execute(
                session.query(func.sum(ModelUsage.processing_time))
                .filter(ModelUsage.user_id == user_id)
            )
            total_processing_time = total_time_result.scalar() or 0.0
            
            # Average processing time
            avg_time_result = await session.execute(
                session.query(func.avg(ModelUsage.processing_time))
                .filter(ModelUsage.user_id == user_id)
            )
            avg_processing_time = avg_time_result.scalar() or 0.0
            
            # Usage by model type
            model_usage_result = await session.execute(
                session.query(
                    ModelUsage.model_type,
                    func.count(ModelUsage.id).label('count'),
                    func.sum(ModelUsage.processing_time).label('total_time')
                )
                .filter(ModelUsage.user_id == user_id)
                .group_by(ModelUsage.model_type)
            )
            model_usage = model_usage_result.all()
            
            return {
                "total_processing_time": total_processing_time,
                "average_processing_time": avg_processing_time,
                "model_usage": [
                    {
                        "model_type": row.model_type,
                        "count": row.count,
                        "total_time": row.total_time or 0.0
                    }
                    for row in model_usage
                ]
            }


# Database configuration
DATABASE_URL = "sqlite+aiosqlite:///./heygen_ai.db"

# Global database manager instance
db_manager = DatabaseManager(DATABASE_URL)

# Repository instances
video_repo = VideoRepository(db_manager)
user_repo = UserRepository(db_manager)
usage_repo = ModelUsageRepository(db_manager)


async def init_database():
    """Initialize database on startup."""
    await db_manager.initialize()


async def close_database():
    """Close database on shutdown."""
    await db_manager.close()


# Database dependency for FastAPI
async def get_db() -> AsyncSession:
    """Database dependency for FastAPI routes."""
    async with db_manager.get_session() as session:
        yield session 