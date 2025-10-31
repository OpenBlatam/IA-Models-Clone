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
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic, Type, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
from pathlib import Path
import weakref
from sqlalchemy.ext.asyncio import (
from sqlalchemy.orm import (
from sqlalchemy import (
from sqlalchemy.exc import (
from sqlalchemy.pool import QueuePool
from sqlalchemy.schema import CreateTable, DropTable
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json import pydantic_encoder
import structlog
from cachetools import TTLCache, LRUCache
import redis.asyncio as redis
            import psutil
from typing import Any, List, Dict, Optional
"""
🗄️ SQLAlchemy 2.0 Implementation - Production Ready
===================================================

Enterprise-grade SQLAlchemy 2.0 implementation with:
- Modern async/await patterns
- Type-safe ORM with Pydantic integration
- Connection pooling and optimization
- Comprehensive error handling
- Performance monitoring
- Migration support
- Multi-database support
- Advanced query optimization
"""


# SQLAlchemy 2.0 imports
    create_async_engine, AsyncSession, async_sessionmaker, 
    AsyncEngine, AsyncAttrs
)
    DeclarativeBase, Mapped, mapped_column, relationship,
    selectinload, joinedload, subqueryload
)
    String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    select, insert, update, delete, func, and_, or_, not_,
    desc, asc, distinct, case, cast, text
)
    SQLAlchemyError, OperationalError, IntegrityError,
    DataError, ProgrammingError, DisconnectionError
)

# Pydantic integration

# Performance and monitoring

logger = structlog.get_logger(__name__)

# Type variables
T = TypeVar('T')
ModelT = TypeVar('ModelT', bound='Base')

# ============================================================================
# Base Classes and Configuration
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models with async support."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def to_pydantic(self) -> Dict[str, Any]:
        """Convert model to Pydantic-compatible dictionary."""
        return json.loads(
            json.dumps(self.to_dict(), default=pydantic_encoder)
        )


@dataclass
class DatabaseConfig:
    """Database configuration with SQLAlchemy 2.0 optimizations."""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False
    echo_pool: bool = False
    json_serializer: callable = json.dumps
    json_deserializer: callable = json.loads
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_name: str
    query_sql: str
    execution_time: float
    memory_usage: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False
    rows_affected: int = 0


@dataclass
class HealthStatus:
    """Database health status."""
    is_healthy: bool
    connection_count: int
    pool_size: int
    avg_query_time: float
    error_rate: float
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# NLP System Models
# ============================================================================

class AnalysisType(str, Enum):
    """Types of NLP analysis."""
    SENTIMENT = "sentiment"
    QUALITY = "quality"
    EMOTION = "emotion"
    LANGUAGE = "language"
    KEYWORDS = "keywords"
    READABILITY = "readability"
    ENTITIES = "entities"
    TOPICS = "topics"


class AnalysisStatus(str, Enum):
    """Analysis processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CACHED = "cached"
    ERROR = "error"


class OptimizationTier(str, Enum):
    """Performance optimization tiers."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ULTRA = "ultra"


class TextAnalysis(Base):
    """Text analysis results model."""
    __tablename__ = "text_analyses"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Content and metadata
    text_content: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    content_length: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Analysis details
    analysis_type: Mapped[AnalysisType] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[AnalysisStatus] = mapped_column(String(20), nullable=False, default=AnalysisStatus.PENDING)
    
    # Results
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emotion_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    language_detected: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    readability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entities: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    topics: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    # Performance metrics
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Optimization tier
    optimization_tier: Mapped[OptimizationTier] = mapped_column(
        String(20), nullable=False, default=OptimizationTier.STANDARD
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Relationships
    batch_analysis_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("batch_analyses.id"), nullable=True
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_text_hash_analysis_type', 'text_hash', 'analysis_type'),
        Index('idx_status_created_at', 'status', 'created_at'),
        Index('idx_optimization_tier_processing_time', 'optimization_tier', 'processing_time_ms'),
        Index('idx_sentiment_score', 'sentiment_score'),
        Index('idx_quality_score', 'quality_score'),
        Index('idx_language_detected', 'language_detected'),
    )


class BatchAnalysis(Base):
    """Batch analysis management model."""
    __tablename__ = "batch_analyses"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Batch metadata
    batch_name: Mapped[str] = mapped_column(String(200), nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    analysis_type: Mapped[AnalysisType] = mapped_column(String(50), nullable=False)
    
    # Status and progress
    status: Mapped[AnalysisStatus] = mapped_column(String(20), nullable=False, default=AnalysisStatus.PENDING)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Performance metrics
    total_processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Configuration
    optimization_tier: Mapped[OptimizationTier] = mapped_column(
        String(20), nullable=False, default=OptimizationTier.STANDARD
    )
    model_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    text_analyses: Mapped[List["TextAnalysis"]] = relationship(
        "TextAnalysis", back_populates="batch_analysis", lazy="selectin"
    )


class ModelPerformance(Base):
    """Model performance tracking model."""
    __tablename__ = "model_performance"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Model information
    model_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Performance metrics
    avg_processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Accuracy metrics
    accuracy_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Resource usage
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'model_version'),
        Index('idx_model_type_performance', 'model_type', 'avg_processing_time_ms'),
        Index('idx_accuracy_score', 'accuracy_score'),
    )


class CacheEntry(Base):
    """Cache management model."""
    __tablename__ = "cache_entries"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Cache key and value
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    cache_value: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Cache metadata
    cache_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Usage statistics
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_key_type', 'cache_key', 'cache_type'),
        Index('idx_expires_at', 'expires_at'),
        Index('idx_last_accessed', 'last_accessed'),
    )


# ============================================================================
# Pydantic Models for API
# ============================================================================

class TextAnalysisCreate(BaseModel):
    """Pydantic model for creating text analysis."""
    text_content: str = Field(..., min_length=1, max_length=10000)
    analysis_type: AnalysisType
    optimization_tier: OptimizationTier = OptimizationTier.STANDARD
    model_config: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class TextAnalysisUpdate(BaseModel):
    """Pydantic model for updating text analysis."""
    status: Optional[AnalysisStatus] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotion_scores: Optional[Dict[str, float]] = None
    language_detected: Optional[str] = None
    keywords: Optional[List[str]] = None
    readability_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[str]] = None
    processing_time_ms: Optional[float] = Field(None, ge=0.0)
    model_used: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class TextAnalysisResponse(BaseModel):
    """Pydantic model for text analysis response."""
    id: int
    text_content: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    sentiment_score: Optional[float] = None
    quality_score: Optional[float] = None
    emotion_scores: Optional[Dict[str, float]] = None
    language_detected: Optional[str] = None
    keywords: Optional[List[str]] = None
    readability_score: Optional[float] = None
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None
    model_used: Optional[str] = None
    confidence_score: Optional[float] = None
    optimization_tier: OptimizationTier
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class BatchAnalysisCreate(BaseModel):
    """Pydantic model for creating batch analysis."""
    batch_name: str = Field(..., min_length=1, max_length=200)
    analysis_type: AnalysisType
    optimization_tier: OptimizationTier = OptimizationTier.STANDARD
    model_config: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class BatchAnalysisResponse(BaseModel):
    """Pydantic model for batch analysis response."""
    id: int
    batch_name: str
    batch_size: int
    analysis_type: AnalysisType
    status: AnalysisStatus
    completed_count: int
    error_count: int
    total_processing_time_ms: Optional[float] = None
    avg_processing_time_ms: Optional[float] = None
    optimization_tier: OptimizationTier
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Database Manager
# ============================================================================

class SQLAlchemy2Manager:
    """SQLAlchemy 2.0 database manager with modern async patterns."""
    
    def __init__(self, config: DatabaseConfig):
        
    """__init__ function."""
self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance monitoring
        self.query_metrics: List[QueryMetrics] = []
        self.query_cache = TTLCache(maxsize=1000, ttl=config.cache_ttl)
        self.performance_monitor = PerformanceMonitor()
        
        # Health monitoring
        self.health_status = HealthStatus(
            is_healthy=False,
            connection_count=0,
            pool_size=config.pool_size,
            avg_query_time=0.0,
            error_rate=0.0,
            last_check=datetime.now()
        )
        
        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.start_time = datetime.now()
        
        logger.info("SQLAlchemy 2.0 Manager initialized")
    
    async def initialize(self) -> Any:
        """Initialize database connections and components."""
        logger.info("Initializing SQLAlchemy 2.0 Manager...")
        
        try:
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.config.url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                json_serializer=self.config.json_serializer,
                json_deserializer=self.config.json_deserializer,
                future=True
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Initialize Redis for caching
            if self.config.enable_caching:
                await self._initialize_redis()
            
            # Test connection
            await self._test_connection()
            
            # Create tables
            await self._create_tables()
            
            # Initialize health monitoring
            await self._update_health_status()
            
            logger.info("SQLAlchemy 2.0 Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy 2.0 Manager: {e}")
            raise DatabaseError("initialization", reason=str(e))
    
    async def cleanup(self) -> Any:
        """Cleanup database connections and resources."""
        logger.info("Cleaning up SQLAlchemy 2.0 Manager...")
        
        try:
            # Close session factory
            if self.session_factory:
                await self.session_factory.close_all()
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Clear cache
            self.query_cache.clear()
            
            logger.info("SQLAlchemy 2.0 Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with automatic cleanup and monitoring."""
        session = None
        start_time = time.time()
        
        try:
            session = self.session_factory()
            self.health_status.connection_count += 1
            
            yield session
            
            # Auto-commit successful transactions
            await session.commit()
            self.successful_queries += 1
            
        except Exception as e:
            if session:
                await session.rollback()
            
            self.failed_queries += 1
            logger.error(f"Database session error: {e}")
            
            # Raise appropriate error
            if isinstance(e, IntegrityError):
                raise ValidationError("data_integrity", reason=str(e))
            elif isinstance(e, OperationalError):
                raise DatabaseError("operation", reason=str(e))
            else:
                raise DatabaseError("session", reason=str(e))
            
        finally:
            if session:
                await session.close()
                self.health_status.connection_count -= 1
            
            # Update metrics
            query_time = time.time() - start_time
            self.performance_monitor.record_query_time(query_time)
    
    # ============================================================================
    # Text Analysis Operations
    # ============================================================================
    
    async def create_text_analysis(self, data: TextAnalysisCreate) -> TextAnalysis:
        """Create a new text analysis record."""
        start_time = time.time()
        
        try:
            # Generate text hash
            text_hash = hashlib.sha256(data.text_content.encode()).hexdigest()
            
            # Check if analysis already exists
            existing = await self.get_text_analysis_by_hash(text_hash, data.analysis_type)
            if existing:
                logger.info(f"Text analysis already exists: {existing.id}")
                return existing
            
            # Create new analysis
            analysis = TextAnalysis(
                text_content=data.text_content,
                text_hash=text_hash,
                content_length=len(data.text_content),
                analysis_type=data.analysis_type,
                optimization_tier=data.optimization_tier,
                status=AnalysisStatus.PENDING
            )
            
            async with self.get_session() as session:
                session.add(analysis)
                await session.commit()
                await session.refresh(analysis)
            
            # Record metrics
            self._record_metrics(
                "create_text_analysis",
                f"INSERT INTO text_analyses",
                time.time() - start_time,
                True
            )
            
            logger.info(f"Created text analysis: {analysis.id}")
            return analysis
            
        except Exception as e:
            self._record_metrics(
                "create_text_analysis",
                f"INSERT INTO text_analyses",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def get_text_analysis(self, analysis_id: int) -> Optional[TextAnalysis]:
        """Get text analysis by ID."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(TextAnalysis).where(TextAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
            
            self._record_metrics(
                "get_text_analysis",
                f"SELECT FROM text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                True
            )
            
            return analysis
            
        except Exception as e:
            self._record_metrics(
                "get_text_analysis",
                f"SELECT FROM text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def get_text_analysis_by_hash(self, text_hash: str, analysis_type: AnalysisType) -> Optional[TextAnalysis]:
        """Get text analysis by text hash and analysis type."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(TextAnalysis).where(
                    and_(
                        TextAnalysis.text_hash == text_hash,
                        TextAnalysis.analysis_type == analysis_type
                    )
                )
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
            
            self._record_metrics(
                "get_text_analysis_by_hash",
                f"SELECT FROM text_analyses WHERE text_hash = {text_hash}",
                time.time() - start_time,
                True
            )
            
            return analysis
            
        except Exception as e:
            self._record_metrics(
                "get_text_analysis_by_hash",
                f"SELECT FROM text_analyses WHERE text_hash = {text_hash}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def update_text_analysis(self, analysis_id: int, data: TextAnalysisUpdate) -> Optional[TextAnalysis]:
        """Update text analysis record."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(TextAnalysis).where(TextAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    return None
                
                # Update fields
                update_data = data.model_dump(exclude_unset=True)
                for field, value in update_data.items():
                    setattr(analysis, field, value)
                
                # Update processed timestamp if status is completed
                if data.status == AnalysisStatus.COMPLETED:
                    analysis.processed_at = datetime.utcnow()
                
                await session.commit()
                await session.refresh(analysis)
            
            self._record_metrics(
                "update_text_analysis",
                f"UPDATE text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                True
            )
            
            logger.info(f"Updated text analysis: {analysis_id}")
            return analysis
            
        except Exception as e:
            self._record_metrics(
                "update_text_analysis",
                f"UPDATE text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def list_text_analyses(
        self,
        analysis_type: Optional[AnalysisType] = None,
        status: Optional[AnalysisStatus] = None,
        optimization_tier: Optional[OptimizationTier] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[TextAnalysis]:
        """List text analyses with filtering and pagination."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                # Build query
                stmt = select(TextAnalysis)
                
                # Add filters
                conditions = []
                if analysis_type:
                    conditions.append(TextAnalysis.analysis_type == analysis_type)
                if status:
                    conditions.append(TextAnalysis.status == status)
                if optimization_tier:
                    conditions.append(TextAnalysis.optimization_tier == optimization_tier)
                
                if conditions:
                    stmt = stmt.where(and_(*conditions))
                
                # Add ordering
                order_column = getattr(TextAnalysis, order_by, TextAnalysis.created_at)
                if order_desc:
                    stmt = stmt.order_by(desc(order_column))
                else:
                    stmt = stmt.order_by(asc(order_column))
                
                # Add pagination
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                analyses = result.scalars().all()
            
            self._record_metrics(
                "list_text_analyses",
                f"SELECT FROM text_analyses LIMIT {limit} OFFSET {offset}",
                time.time() - start_time,
                True,
                rows_affected=len(analyses)
            )
            
            return list(analyses)
            
        except Exception as e:
            self._record_metrics(
                "list_text_analyses",
                f"SELECT FROM text_analyses LIMIT {limit} OFFSET {offset}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def delete_text_analysis(self, analysis_id: int) -> bool:
        """Delete text analysis record."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(TextAnalysis).where(TextAnalysis.id == analysis_id)
                result = await session.execute(stmt)
                analysis = result.scalar_one_or_none()
                
                if not analysis:
                    return False
                
                await session.delete(analysis)
                await session.commit()
            
            self._record_metrics(
                "delete_text_analysis",
                f"DELETE FROM text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                True
            )
            
            logger.info(f"Deleted text analysis: {analysis_id}")
            return True
            
        except Exception as e:
            self._record_metrics(
                "delete_text_analysis",
                f"DELETE FROM text_analyses WHERE id = {analysis_id}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    # ============================================================================
    # Batch Analysis Operations
    # ============================================================================
    
    async def create_batch_analysis(self, data: BatchAnalysisCreate) -> BatchAnalysis:
        """Create a new batch analysis record."""
        start_time = time.time()
        
        try:
            batch = BatchAnalysis(
                batch_name=data.batch_name,
                batch_size=0,  # Will be updated when texts are added
                analysis_type=data.analysis_type,
                optimization_tier=data.optimization_tier,
                model_config=data.model_config,
                status=AnalysisStatus.PENDING
            )
            
            async with self.get_session() as session:
                session.add(batch)
                await session.commit()
                await session.refresh(batch)
            
            self._record_metrics(
                "create_batch_analysis",
                f"INSERT INTO batch_analyses",
                time.time() - start_time,
                True
            )
            
            logger.info(f"Created batch analysis: {batch.id}")
            return batch
            
        except Exception as e:
            self._record_metrics(
                "create_batch_analysis",
                f"INSERT INTO batch_analyses",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def get_batch_analysis(self, batch_id: int) -> Optional[BatchAnalysis]:
        """Get batch analysis by ID."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(BatchAnalysis).where(BatchAnalysis.id == batch_id)
                result = await session.execute(stmt)
                batch = result.scalar_one_or_none()
            
            self._record_metrics(
                "get_batch_analysis",
                f"SELECT FROM batch_analyses WHERE id = {batch_id}",
                time.time() - start_time,
                True
            )
            
            return batch
            
        except Exception as e:
            self._record_metrics(
                "get_batch_analysis",
                f"SELECT FROM batch_analyses WHERE id = {batch_id}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    async def update_batch_progress(self, batch_id: int, completed_count: int, error_count: int) -> Optional[BatchAnalysis]:
        """Update batch analysis progress."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                stmt = select(BatchAnalysis).where(BatchAnalysis.id == batch_id)
                result = await session.execute(stmt)
                batch = result.scalar_one_or_none()
                
                if not batch:
                    return None
                
                batch.completed_count = completed_count
                batch.error_count = error_count
                
                # Update status if all items are processed
                if completed_count + error_count >= batch.batch_size:
                    batch.status = AnalysisStatus.COMPLETED
                    batch.completed_at = datetime.utcnow()
                
                await session.commit()
                await session.refresh(batch)
            
            self._record_metrics(
                "update_batch_progress",
                f"UPDATE batch_analyses WHERE id = {batch_id}",
                time.time() - start_time,
                True
            )
            
            return batch
            
        except Exception as e:
            self._record_metrics(
                "update_batch_progress",
                f"UPDATE batch_analyses WHERE id = {batch_id}",
                time.time() - start_time,
                False,
                str(e)
            )
            raise
    
    # ============================================================================
    # Performance and Monitoring
    # ============================================================================
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "database": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "success_rate": self.successful_queries / max(self.total_queries, 1),
                "avg_query_time": self.performance_monitor.get_average_query_time(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            },
            "cache": {
                "size": len(self.query_cache),
                "hits": getattr(self.query_cache, 'hits', 0),
                "misses": getattr(self.query_cache, 'misses', 0),
                "hit_rate": getattr(self.query_cache, 'hits', 0) / max(getattr(self.query_cache, 'hits', 0) + getattr(self.query_cache, 'misses', 0), 1)
            },
            "health": {
                "is_healthy": self.health_status.is_healthy,
                "connection_count": self.health_status.connection_count,
                "pool_size": self.health_status.pool_size,
                "last_check": self.health_status.last_check.isoformat()
            }
        }
    
    async def health_check(self) -> HealthStatus:
        """Perform database health check."""
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                stmt = select(func.count(TextAnalysis.id))
                result = await session.execute(stmt)
                count = result.scalar()
            
            response_time = time.time() - start_time
            
            # Update health status
            self.health_status.is_healthy = True
            self.health_status.last_check = datetime.now()
            self.health_status.avg_query_time = response_time
            
            return self.health_status
            
        except Exception as e:
            self.health_status.is_healthy = False
            self.health_status.last_check = datetime.now()
            logger.error(f"Health check failed: {e}")
            return self.health_status
    
    # ============================================================================
    # Private Methods
    # ============================================================================
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    async def _test_connection(self) -> Any:
        """Test database connection."""
        async with self.get_session() as session:
            await session.execute(select(1))
    
    async def _create_tables(self) -> Any:
        """Create database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def _update_health_status(self) -> Any:
        """Update health status."""
        await self.health_check()
    
    def _record_metrics(self, query_name: str, query_sql: str, execution_time: float, 
                       success: bool, error_message: Optional[str] = None, rows_affected: int = 0):
        """Record query metrics."""
        self.total_queries += 1
        
        metric = QueryMetrics(
            query_name=query_name,
            query_sql=query_sql,
            execution_time=execution_time,
            memory_usage=self.performance_monitor.get_memory_usage(),
            success=success,
            error_message=error_message,
            rows_affected=rows_affected
        )
        
        self.query_metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]


class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self) -> Any:
        self.query_times: List[float] = []
        self.max_history = 1000
    
    def record_query_time(self, query_time: float):
        """Record query execution time."""
        self.query_times.append(query_time)
        
        # Keep only recent history
        if len(self.query_times) > self.max_history:
            self.query_times = self.query_times[-self.max_history:]
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_average_query_time(self) -> float:
        """Get average query execution time."""
        if not self.query_times:
            return 0.0
        return sum(self.query_times) / len(self.query_times)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.query_times:
            return {
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "total_queries": 0
            }
        
        return {
            "avg_time": sum(self.query_times) / len(self.query_times),
            "min_time": min(self.query_times),
            "max_time": max(self.query_times),
            "total_queries": len(self.query_times),
            "memory_usage_mb": self.get_memory_usage()
        }


# ============================================================================
# Error Classes
# ============================================================================

class DatabaseError(Exception):
    """Base database error."""
    
    def __init__(self, operation: str, reason: str, **kwargs):
        
    """__init__ function."""
self.operation = operation
        self.reason = reason
        self.details = kwargs
        super().__init__(f"Database {operation} failed: {reason}")


class ValidationError(Exception):
    """Validation error."""
    
    def __init__(self, field: str, reason: str, **kwargs):
        
    """__init__ function."""
self.field = field
        self.reason = reason
        self.details = kwargs
        super().__init__(f"Validation error in {field}: {reason}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_database_config(
    url: str,
    pool_size: int = 20,
    max_overflow: int = 30,
    enable_caching: bool = True
) -> DatabaseConfig:
    """Create database configuration with sensible defaults."""
    return DatabaseConfig(
        url=url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        enable_caching=enable_caching
    )


async def create_database_manager(config: DatabaseConfig) -> SQLAlchemy2Manager:
    """Create and initialize database manager."""
    manager = SQLAlchemy2Manager(config)
    await manager.initialize()
    return manager


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example usage of SQLAlchemy 2.0 implementation."""
    
    # Create configuration
    config = create_database_config(
        url="postgresql+asyncpg://user:password@localhost/nlp_db",
        pool_size=10,
        enable_caching=True
    )
    
    # Create manager
    db_manager = await create_database_manager(config)
    
    try:
        # Create text analysis
        analysis_data = TextAnalysisCreate(
            text_content="This is a sample text for sentiment analysis.",
            analysis_type=AnalysisType.SENTIMENT,
            optimization_tier=OptimizationTier.STANDARD
        )
        
        analysis = await db_manager.create_text_analysis(analysis_data)
        print(f"Created analysis: {analysis.id}")
        
        # Update analysis with results
        update_data = TextAnalysisUpdate(
            status=AnalysisStatus.COMPLETED,
            sentiment_score=0.8,
            processing_time_ms=150.5,
            model_used="distilbert-sentiment",
            confidence_score=0.95
        )
        
        updated_analysis = await db_manager.update_text_analysis(analysis.id, update_data)
        print(f"Updated analysis: {updated_analysis.sentiment_score}")
        
        # List analyses
        analyses = await db_manager.list_text_analyses(
            analysis_type=AnalysisType.SENTIMENT,
            limit=10
        )
        print(f"Found {len(analyses)} sentiment analyses")
        
        # Get performance metrics
        metrics = await db_manager.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
    finally:
        await db_manager.cleanup()


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 