"""
Database Layer for OpusClip Improved
===================================

Advanced database management with async support and connection pooling.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy import select, update, delete, func
import redis.asyncio as redis

from .schemas import get_settings
from .exceptions import DatabaseError, create_database_error

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# Database Models
class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(50), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    projects: Mapped[List["Project"]] = relationship("Project", back_populates="owner")
    analyses: Mapped[List["VideoAnalysis"]] = relationship("VideoAnalysis", back_populates="user")
    generations: Mapped[List["ClipGeneration"]] = relationship("ClipGeneration", back_populates="user")


class Project(Base):
    """Project model"""
    __tablename__ = "projects"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    settings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="projects")
    analyses: Mapped[List["VideoAnalysis"]] = relationship("VideoAnalysis", back_populates="project")
    generations: Mapped[List["ClipGeneration"]] = relationship("ClipGeneration", back_populates="project")


class VideoAnalysis(Base):
    """Video analysis model"""
    __tablename__ = "video_analyses"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("projects.id"))
    video_url: Mapped[Optional[str]] = mapped_column(String(500))
    video_path: Mapped[Optional[str]] = mapped_column(String(500))
    video_filename: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    
    # Video metadata
    duration: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    resolution: Mapped[Optional[str]] = mapped_column(String(50))
    format: Mapped[Optional[str]] = mapped_column(String(20))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Analysis results
    transcript: Mapped[Optional[str]] = mapped_column(Text)
    sentiment_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    key_moments: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    scene_changes: Mapped[Optional[List[float]]] = mapped_column(JSON)
    face_detections: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    object_detections: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    
    # AI insights
    content_summary: Mapped[Optional[str]] = mapped_column(Text)
    topics: Mapped[Optional[List[str]]] = mapped_column(JSON)
    emotions: Mapped[Optional[List[str]]] = mapped_column(JSON)
    viral_potential: Mapped[Optional[float]] = mapped_column(Float)
    
    # Processing info
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")
    project: Mapped[Optional["Project"]] = relationship("Project", back_populates="analyses")
    generations: Mapped[List["ClipGeneration"]] = relationship("ClipGeneration", back_populates="analysis")


class ClipGeneration(Base):
    """Clip generation model"""
    __tablename__ = "clip_generations"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("projects.id"))
    analysis_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("video_analyses.id"), nullable=False)
    clip_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    
    # Generation settings
    target_duration: Mapped[int] = mapped_column(Integer, default=30)
    max_clips: Mapped[int] = mapped_column(Integer, default=5)
    include_intro: Mapped[bool] = mapped_column(Boolean, default=True)
    include_outro: Mapped[bool] = mapped_column(Boolean, default=True)
    add_captions: Mapped[bool] = mapped_column(Boolean, default=True)
    add_watermark: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # AI settings
    ai_provider: Mapped[str] = mapped_column(String(50), default="openai")
    custom_prompt: Mapped[Optional[str]] = mapped_column(Text)
    style_preference: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Platform settings
    target_platforms: Mapped[List[str]] = mapped_column(JSON, default=list)
    aspect_ratio: Mapped[Optional[str]] = mapped_column(String(20))
    quality: Mapped[str] = mapped_column(String(20), default="high")
    
    # Results
    clips: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="generations")
    project: Mapped[Optional["Project"]] = relationship("Project", back_populates="generations")
    analysis: Mapped["VideoAnalysis"] = relationship("VideoAnalysis", back_populates="generations")
    exports: Mapped[List["ClipExport"]] = relationship("ClipExport", back_populates="generation")


class ClipExport(Base):
    """Clip export model"""
    __tablename__ = "clip_exports"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    generation_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("clip_generations.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    
    # Export settings
    format: Mapped[str] = mapped_column(String(20), default="mp4")
    quality: Mapped[str] = mapped_column(String(20), default="high")
    resolution: Mapped[Optional[str]] = mapped_column(String(50))
    bitrate: Mapped[Optional[int]] = mapped_column(Integer)
    target_platform: Mapped[str] = mapped_column(String(50), nullable=False)
    optimize_for_platform: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Export options
    add_metadata: Mapped[bool] = mapped_column(Boolean, default=True)
    create_thumbnail: Mapped[bool] = mapped_column(Boolean, default=True)
    generate_subtitles: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Results
    exported_files: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    download_urls: Mapped[Optional[List[str]]] = mapped_column(JSON)
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    generation: Mapped["ClipGeneration"] = relationship("ClipGeneration", back_populates="exports")


class BatchProcessing(Base):
    """Batch processing model"""
    __tablename__ = "batch_processings"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    
    # Batch settings
    total_videos: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_videos: Mapped[int] = mapped_column(Integer, default=0)
    failed_videos: Mapped[int] = mapped_column(Integer, default=0)
    parallel_processing: Mapped[bool] = mapped_column(Boolean, default=True)
    max_concurrent: Mapped[int] = mapped_column(Integer, default=3)
    
    # Notification settings
    notify_on_completion: Mapped[bool] = mapped_column(Boolean, default=True)
    webhook_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Results
    analysis_results: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    generation_results: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    total_processing_time: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class Analytics(Base):
    """Analytics model"""
    __tablename__ = "analytics"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("users.id"))
    project_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID(as_uuid=True), ForeignKey("projects.id"))
    
    # Analytics data
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    trends: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    insights: Mapped[List[str]] = mapped_column(JSON, default=list)
    recommendations: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Metadata
    date_range_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    date_range_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """System metrics model"""
    __tablename__ = "system_metrics"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[Optional[str]] = mapped_column(String(20))
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Database connection and session management
class DatabaseManager:
    """Database manager with connection pooling and async support"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.database_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                pool_pre_ping=True,
                echo=self.settings.debug
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize Redis
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise create_database_error("initialize", "database", e)
    
    async def create_tables(self):
        """Create database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise create_database_error("create_tables", "database", e)
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            await self.initialize()
        return self.session_factory()
    
    async def get_redis(self):
        """Get Redis client"""
        if not self.redis_client:
            await self.initialize()
        return self.redis_client
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager
db_manager = DatabaseManager()


async def get_database_session() -> AsyncSession:
    """Get database session dependency"""
    return await db_manager.get_session()


async def get_redis_client():
    """Get Redis client dependency"""
    return await db_manager.get_redis()


async def init_database():
    """Initialize database"""
    await db_manager.initialize()
    await db_manager.create_tables()


# Database operations
class DatabaseOperations:
    """Database operations helper class"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            user = User(**user_data)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_user", "users", e)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_user_by_email", "users", e)
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_user_by_id", "users", e)
    
    async def create_project(self, project_data: Dict[str, Any]) -> Project:
        """Create a new project"""
        try:
            project = Project(**project_data)
            self.session.add(project)
            await self.session.commit()
            await self.session.refresh(project)
            return project
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_project", "projects", e)
    
    async def get_project_by_id(self, project_id: UUID) -> Optional[Project]:
        """Get project by ID"""
        try:
            result = await self.session.execute(
                select(Project).where(Project.id == project_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_project_by_id", "projects", e)
    
    async def get_user_projects(self, user_id: UUID, limit: int = 10, offset: int = 0) -> List[Project]:
        """Get user's projects"""
        try:
            result = await self.session.execute(
                select(Project)
                .where(Project.owner_id == user_id)
                .order_by(Project.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
        except Exception as e:
            raise create_database_error("get_user_projects", "projects", e)
    
    async def create_video_analysis(self, analysis_data: Dict[str, Any]) -> VideoAnalysis:
        """Create video analysis record"""
        try:
            analysis = VideoAnalysis(**analysis_data)
            self.session.add(analysis)
            await self.session.commit()
            await self.session.refresh(analysis)
            return analysis
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_video_analysis", "video_analyses", e)
    
    async def update_video_analysis(self, analysis_id: UUID, update_data: Dict[str, Any]) -> Optional[VideoAnalysis]:
        """Update video analysis"""
        try:
            result = await self.session.execute(
                update(VideoAnalysis)
                .where(VideoAnalysis.id == analysis_id)
                .values(**update_data)
                .returning(VideoAnalysis)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("update_video_analysis", "video_analyses", e)
    
    async def get_video_analysis_by_id(self, analysis_id: UUID) -> Optional[VideoAnalysis]:
        """Get video analysis by ID"""
        try:
            result = await self.session.execute(
                select(VideoAnalysis).where(VideoAnalysis.id == analysis_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_video_analysis_by_id", "video_analyses", e)
    
    async def create_clip_generation(self, generation_data: Dict[str, Any]) -> ClipGeneration:
        """Create clip generation record"""
        try:
            generation = ClipGeneration(**generation_data)
            self.session.add(generation)
            await self.session.commit()
            await self.session.refresh(generation)
            return generation
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_clip_generation", "clip_generations", e)
    
    async def update_clip_generation(self, generation_id: UUID, update_data: Dict[str, Any]) -> Optional[ClipGeneration]:
        """Update clip generation"""
        try:
            result = await self.session.execute(
                update(ClipGeneration)
                .where(ClipGeneration.id == generation_id)
                .values(**update_data)
                .returning(ClipGeneration)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("update_clip_generation", "clip_generations", e)
    
    async def get_clip_generation_by_id(self, generation_id: UUID) -> Optional[ClipGeneration]:
        """Get clip generation by ID"""
        try:
            result = await self.session.execute(
                select(ClipGeneration).where(ClipGeneration.id == generation_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_clip_generation_by_id", "clip_generations", e)
    
    async def create_clip_export(self, export_data: Dict[str, Any]) -> ClipExport:
        """Create clip export record"""
        try:
            export = ClipExport(**export_data)
            self.session.add(export)
            await self.session.commit()
            await self.session.refresh(export)
            return export
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_clip_export", "clip_exports", e)
    
    async def create_batch_processing(self, batch_data: Dict[str, Any]) -> BatchProcessing:
        """Create batch processing record"""
        try:
            batch = BatchProcessing(**batch_data)
            self.session.add(batch)
            await self.session.commit()
            await self.session.refresh(batch)
            return batch
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_batch_processing", "batch_processings", e)
    
    async def update_batch_processing(self, batch_id: UUID, update_data: Dict[str, Any]) -> Optional[BatchProcessing]:
        """Update batch processing"""
        try:
            result = await self.session.execute(
                update(BatchProcessing)
                .where(BatchProcessing.id == batch_id)
                .values(**update_data)
                .returning(BatchProcessing)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("update_batch_processing", "batch_processings", e)
    
    async def create_analytics(self, analytics_data: Dict[str, Any]) -> Analytics:
        """Create analytics record"""
        try:
            analytics = Analytics(**analytics_data)
            self.session.add(analytics)
            await self.session.commit()
            await self.session.refresh(analytics)
            return analytics
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("create_analytics", "analytics", e)
    
    async def get_analytics_by_id(self, analytics_id: UUID) -> Optional[Analytics]:
        """Get analytics by ID"""
        try:
            result = await self.session.execute(
                select(Analytics).where(Analytics.id == analytics_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise create_database_error("get_analytics_by_id", "analytics", e)
    
    async def record_system_metric(self, metric_name: str, metric_value: float, metric_unit: str = None, metadata: Dict[str, Any] = None):
        """Record system metric"""
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                metadata=metadata or {}
            )
            self.session.add(metric)
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise create_database_error("record_system_metric", "system_metrics", e)
    
    async def get_system_metrics(self, metric_name: str, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics for a specific time period"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            result = await self.session.execute(
                select(SystemMetrics)
                .where(
                    and_(
                        SystemMetrics.metric_name == metric_name,
                        SystemMetrics.timestamp >= start_time
                    )
                )
                .order_by(SystemMetrics.timestamp.desc())
            )
            return result.scalars().all()
        except Exception as e:
            raise create_database_error("get_system_metrics", "system_metrics", e)
    
    async def get_analytics_summary(self, user_id: Optional[UUID] = None, project_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get analytics summary"""
        try:
            # Get total counts
            total_analyses = await self.session.scalar(
                select(func.count(VideoAnalysis.id))
            )
            
            total_generations = await self.session.scalar(
                select(func.count(ClipGeneration.id))
            )
            
            total_exports = await self.session.scalar(
                select(func.count(ClipExport.id))
            )
            
            # Get success rates
            successful_analyses = await self.session.scalar(
                select(func.count(VideoAnalysis.id))
                .where(VideoAnalysis.status == "completed")
            )
            
            successful_generations = await self.session.scalar(
                select(func.count(ClipGeneration.id))
                .where(ClipGeneration.status == "completed")
            )
            
            # Calculate rates
            analysis_success_rate = successful_analyses / total_analyses if total_analyses > 0 else 0
            generation_success_rate = successful_generations / total_generations if total_generations > 0 else 0
            
            return {
                "total_analyses": total_analyses,
                "total_generations": total_generations,
                "total_exports": total_exports,
                "analysis_success_rate": analysis_success_rate,
                "generation_success_rate": generation_success_rate
            }
        except Exception as e:
            raise create_database_error("get_analytics_summary", "analytics", e)





























