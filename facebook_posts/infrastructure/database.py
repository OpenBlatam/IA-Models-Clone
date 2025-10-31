"""
Advanced database layer for Facebook Posts API
Async database operations with connection pooling and migrations
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import structlog

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    from sqlalchemy import String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey, Index
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy import select, update, delete, func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from ..core.config import get_settings
from ..core.models import PostStatus, ContentType, AudienceType, OptimizationLevel, QualityTier

logger = structlog.get_logger(__name__)


if SQLALCHEMY_AVAILABLE:
    class Base(DeclarativeBase):
        """Base class for all database models"""
        pass


    class FacebookPostModel(Base):
        """Database model for Facebook posts"""
        __tablename__ = "facebook_posts"
        
        id: Mapped[str] = mapped_column(String(255), primary_key=True)
        content: Mapped[str] = mapped_column(Text, nullable=False)
        status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft")
        content_type: Mapped[str] = mapped_column(String(50), nullable=False)
        audience_type: Mapped[str] = mapped_column(String(50), nullable=False)
        created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
        updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Optional fields
        optimization_level: Mapped[Optional[str]] = mapped_column(String(50), default="standard")
        quality_tier: Mapped[Optional[str]] = mapped_column(String(50))
        tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
        hashtags: Mapped[Optional[List[str]]] = mapped_column(JSON)
        mentions: Mapped[Optional[List[str]]] = mapped_column(JSON)
        urls: Mapped[Optional[List[str]]] = mapped_column(JSON)
        metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
        
        # Metrics
        engagement_score: Mapped[Optional[float]] = mapped_column(Float)
        quality_score: Mapped[Optional[float]] = mapped_column(Float)
        readability_score: Mapped[Optional[float]] = mapped_column(Float)
        sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
        creativity_score: Mapped[Optional[float]] = mapped_column(Float)
        relevance_score: Mapped[Optional[float]] = mapped_column(Float)
        
        # Relationships
        analytics: Mapped[List["PostAnalyticsModel"]] = relationship("PostAnalyticsModel", back_populates="post", cascade="all, delete-orphan")
        
        # Indexes
        __table_args__ = (
            Index('idx_posts_status', 'status'),
            Index('idx_posts_content_type', 'content_type'),
            Index('idx_posts_audience_type', 'audience_type'),
            Index('idx_posts_created_at', 'created_at'),
            Index('idx_posts_quality_tier', 'quality_tier'),
        )


    class PostAnalyticsModel(Base):
        """Database model for post analytics"""
        __tablename__ = "post_analytics"
        
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        post_id: Mapped[str] = mapped_column(String(255), ForeignKey("facebook_posts.id"), nullable=False)
        analytics_type: Mapped[str] = mapped_column(String(100), nullable=False)
        data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
        generated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
        valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime)
        
        # Relationships
        post: Mapped["FacebookPostModel"] = relationship("FacebookPostModel", back_populates="analytics")
        
        # Indexes
        __table_args__ = (
            Index('idx_analytics_post_id', 'post_id'),
            Index('idx_analytics_type', 'analytics_type'),
            Index('idx_analytics_generated_at', 'generated_at'),
        )


    class SystemMetricsModel(Base):
        """Database model for system metrics"""
        __tablename__ = "system_metrics"
        
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
        metric_value: Mapped[float] = mapped_column(Float, nullable=False)
        labels: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)
        timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
        
        # Indexes
        __table_args__ = (
            Index('idx_metrics_name', 'metric_name'),
            Index('idx_metrics_timestamp', 'timestamp'),
        )


class DatabaseManager:
    """Database manager for async operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection"""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, using mock database")
            return
        
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.database_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                echo=self.settings.debug,
                future=True
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


class PostRepository:
    """Repository for Facebook post operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_post(self, post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new post"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            post_data["id"] = f"mock_post_{int(datetime.now().timestamp())}"
            post_data["created_at"] = datetime.now()
            post_data["updated_at"] = datetime.now()
            return post_data
        
        try:
            async with self.db_manager.get_session() as session:
                post = FacebookPostModel(**post_data)
                session.add(post)
                await session.flush()
                
                # Convert to dict
                result = {
                    "id": post.id,
                    "content": post.content,
                    "status": post.status,
                    "content_type": post.content_type,
                    "audience_type": post.audience_type,
                    "created_at": post.created_at,
                    "updated_at": post.updated_at,
                    "optimization_level": post.optimization_level,
                    "quality_tier": post.quality_tier,
                    "tags": post.tags,
                    "hashtags": post.hashtags,
                    "mentions": post.mentions,
                    "urls": post.urls,
                    "metadata": post.metadata,
                    "engagement_score": post.engagement_score,
                    "quality_score": post.quality_score,
                    "readability_score": post.readability_score,
                    "sentiment_score": post.sentiment_score,
                    "creativity_score": post.creativity_score,
                    "relevance_score": post.relevance_score,
                }
                
                return result
                
        except Exception as e:
            logger.error("Failed to create post", error=str(e))
            return None
    
    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get post by ID"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return {
                "id": post_id,
                "content": "Mock post content",
                "status": "draft",
                "content_type": "educational",
                "audience_type": "professionals",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(FacebookPostModel).where(FacebookPostModel.id == post_id)
                )
                post = result.scalar_one_or_none()
                
                if not post:
                    return None
                
                return {
                    "id": post.id,
                    "content": post.content,
                    "status": post.status,
                    "content_type": post.content_type,
                    "audience_type": post.audience_type,
                    "created_at": post.created_at,
                    "updated_at": post.updated_at,
                    "optimization_level": post.optimization_level,
                    "quality_tier": post.quality_tier,
                    "tags": post.tags,
                    "hashtags": post.hashtags,
                    "mentions": post.mentions,
                    "urls": post.urls,
                    "metadata": post.metadata,
                    "engagement_score": post.engagement_score,
                    "quality_score": post.quality_score,
                    "readability_score": post.readability_score,
                    "sentiment_score": post.sentiment_score,
                    "creativity_score": post.creativity_score,
                    "relevance_score": post.relevance_score,
                }
                
        except Exception as e:
            logger.error("Failed to get post", post_id=post_id, error=str(e))
            return None
    
    async def update_post(self, post_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update post"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return await self.get_post(post_id)
        
        try:
            async with self.db_manager.get_session() as session:
                # Add updated_at timestamp
                update_data["updated_at"] = datetime.utcnow()
                
                result = await session.execute(
                    update(FacebookPostModel)
                    .where(FacebookPostModel.id == post_id)
                    .values(**update_data)
                    .returning(FacebookPostModel)
                )
                post = result.scalar_one_or_none()
                
                if not post:
                    return None
                
                return {
                    "id": post.id,
                    "content": post.content,
                    "status": post.status,
                    "content_type": post.content_type,
                    "audience_type": post.audience_type,
                    "created_at": post.created_at,
                    "updated_at": post.updated_at,
                    "optimization_level": post.optimization_level,
                    "quality_tier": post.quality_tier,
                    "tags": post.tags,
                    "hashtags": post.hashtags,
                    "mentions": post.mentions,
                    "urls": post.urls,
                    "metadata": post.metadata,
                    "engagement_score": post.engagement_score,
                    "quality_score": post.quality_score,
                    "readability_score": post.readability_score,
                    "sentiment_score": post.sentiment_score,
                    "creativity_score": post.creativity_score,
                    "relevance_score": post.relevance_score,
                }
                
        except Exception as e:
            logger.error("Failed to update post", post_id=post_id, error=str(e))
            return None
    
    async def delete_post(self, post_id: str) -> bool:
        """Delete post"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return True
        
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    delete(FacebookPostModel).where(FacebookPostModel.id == post_id)
                )
                return result.rowcount > 0
                
        except Exception as e:
            logger.error("Failed to delete post", post_id=post_id, error=str(e))
            return False
    
    async def list_posts(
        self,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List posts with filters"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return [
                {
                    "id": f"mock_post_{i}",
                    "content": f"Mock post content {i}",
                    "status": "draft",
                    "content_type": "educational",
                    "audience_type": "professionals",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                for i in range(min(limit, 5))
            ]
        
        try:
            async with self.db_manager.get_session() as session:
                query = select(FacebookPostModel)
                
                # Apply filters
                if filters:
                    if "status" in filters:
                        query = query.where(FacebookPostModel.status == filters["status"])
                    if "content_type" in filters:
                        query = query.where(FacebookPostModel.content_type == filters["content_type"])
                    if "audience_type" in filters:
                        query = query.where(FacebookPostModel.audience_type == filters["audience_type"])
                    if "quality_tier" in filters:
                        query = query.where(FacebookPostModel.quality_tier == filters["quality_tier"])
                
                # Apply pagination
                query = query.offset(skip).limit(limit)
                
                # Order by created_at desc
                query = query.order_by(FacebookPostModel.created_at.desc())
                
                result = await session.execute(query)
                posts = result.scalars().all()
                
                return [
                    {
                        "id": post.id,
                        "content": post.content,
                        "status": post.status,
                        "content_type": post.content_type,
                        "audience_type": post.audience_type,
                        "created_at": post.created_at,
                        "updated_at": post.updated_at,
                        "optimization_level": post.optimization_level,
                        "quality_tier": post.quality_tier,
                        "tags": post.tags,
                        "hashtags": post.hashtags,
                        "mentions": post.mentions,
                        "urls": post.urls,
                        "metadata": post.metadata,
                        "engagement_score": post.engagement_score,
                        "quality_score": post.quality_score,
                        "readability_score": post.readability_score,
                        "sentiment_score": post.sentiment_score,
                        "creativity_score": post.creativity_score,
                        "relevance_score": post.relevance_score,
                    }
                    for post in posts
                ]
                
        except Exception as e:
            logger.error("Failed to list posts", error=str(e))
            return []
    
    async def count_posts(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count posts with filters"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return 100
        
        try:
            async with self.db_manager.get_session() as session:
                query = select(func.count(FacebookPostModel.id))
                
                # Apply filters
                if filters:
                    if "status" in filters:
                        query = query.where(FacebookPostModel.status == filters["status"])
                    if "content_type" in filters:
                        query = query.where(FacebookPostModel.content_type == filters["content_type"])
                    if "audience_type" in filters:
                        query = query.where(FacebookPostModel.audience_type == filters["audience_type"])
                    if "quality_tier" in filters:
                        query = query.where(FacebookPostModel.quality_tier == filters["quality_tier"])
                
                result = await session.execute(query)
                return result.scalar()
                
        except Exception as e:
            logger.error("Failed to count posts", error=str(e))
            return 0


class AnalyticsRepository:
    """Repository for analytics operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_analytics(self, post_id: str, analytics_type: str, data: Dict[str, Any]) -> bool:
        """Create analytics record"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return True
        
        try:
            async with self.db_manager.get_session() as session:
                analytics = PostAnalyticsModel(
                    post_id=post_id,
                    analytics_type=analytics_type,
                    data=data
                )
                session.add(analytics)
                return True
                
        except Exception as e:
            logger.error("Failed to create analytics", post_id=post_id, error=str(e))
            return False
    
    async def get_analytics(self, post_id: str, analytics_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get analytics for a post"""
        if not SQLALCHEMY_AVAILABLE:
            # Mock implementation
            return [
                {
                    "id": 1,
                    "post_id": post_id,
                    "analytics_type": analytics_type or "engagement",
                    "data": {"likes": 100, "shares": 50, "comments": 25},
                    "generated_at": datetime.now()
                }
            ]
        
        try:
            async with self.db_manager.get_session() as session:
                query = select(PostAnalyticsModel).where(PostAnalyticsModel.post_id == post_id)
                
                if analytics_type:
                    query = query.where(PostAnalyticsModel.analytics_type == analytics_type)
                
                query = query.order_by(PostAnalyticsModel.generated_at.desc())
                
                result = await session.execute(query)
                analytics = result.scalars().all()
                
                return [
                    {
                        "id": a.id,
                        "post_id": a.post_id,
                        "analytics_type": a.analytics_type,
                        "data": a.data,
                        "generated_at": a.generated_at,
                        "valid_until": a.valid_until
                    }
                    for a in analytics
                ]
                
        except Exception as e:
            logger.error("Failed to get analytics", post_id=post_id, error=str(e))
            return []


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager


async def initialize_database():
    """Initialize global database"""
    db_manager = get_db_manager()
    await db_manager.initialize()


async def close_database():
    """Close global database connections"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


# Export all classes and functions
__all__ = [
    # Database models (if SQLAlchemy is available)
    'Base',
    'FacebookPostModel',
    'PostAnalyticsModel',
    'SystemMetricsModel',
    
    # Main classes
    'DatabaseManager',
    'PostRepository',
    'AnalyticsRepository',
    
    # Utility functions
    'get_db_manager',
    'initialize_database',
    'close_database',
]






























