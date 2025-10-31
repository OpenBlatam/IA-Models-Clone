"""
Database Integration for Email Sequence System

This module provides async database integration using SQLAlchemy 2.0
with proper connection management and session handling.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# Database Models
class EmailSequenceModel(Base):
    """Email sequence database model"""
    __tablename__ = "email_sequences"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft")
    
    # Configuration
    personalization_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    personalization_variables: Mapped[Optional[dict]] = mapped_column(JSON)
    ab_testing_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    ab_test_variants: Mapped[Optional[dict]] = mapped_column(JSON)
    tracking_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    conversion_tracking: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timing
    max_duration_days: Mapped[Optional[int]] = mapped_column(Integer)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    
    # Metadata
    tags: Mapped[Optional[list]] = mapped_column(JSON)
    category: Mapped[Optional[str]] = mapped_column(String(100))
    priority: Mapped[int] = mapped_column(Integer, default=1)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Statistics
    total_subscribers: Mapped[int] = mapped_column(Integer, default=0)
    active_subscribers: Mapped[int] = mapped_column(Integer, default=0)
    completed_subscribers: Mapped[int] = mapped_column(Integer, default=0)


class SequenceStepModel(Base):
    """Sequence step database model"""
    __tablename__ = "sequence_steps"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sequence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("email_sequences.id"), nullable=False)
    step_type: Mapped[str] = mapped_column(String(50), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Email-specific fields
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    subject: Mapped[Optional[str]] = mapped_column(String(255))
    content: Mapped[Optional[str]] = mapped_column(Text)
    
    # Delay-specific fields
    delay_hours: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    delay_days: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    
    # Condition-specific fields
    condition_expression: Mapped[Optional[str]] = mapped_column(Text)
    condition_variables: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Action-specific fields
    action_type: Mapped[Optional[str]] = mapped_column(String(100))
    action_data: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Webhook-specific fields
    webhook_url: Mapped[Optional[str]] = mapped_column(String(500))
    webhook_method: Mapped[Optional[str]] = mapped_column(String(10), default="POST")
    webhook_headers: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Common fields
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SequenceTriggerModel(Base):
    """Sequence trigger database model"""
    __tablename__ = "sequence_triggers"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sequence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("email_sequences.id"), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)
    delay_hours: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    delay_days: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    event_name: Mapped[Optional[str]] = mapped_column(String(100))
    conditions: Mapped[Optional[dict]] = mapped_column(JSON)
    scheduled_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmailTemplateModel(Base):
    """Email template database model"""
    __tablename__ = "email_templates"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    html_content: Mapped[str] = mapped_column(Text, nullable=False)
    text_content: Mapped[Optional[str]] = mapped_column(Text)
    variables: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    category: Mapped[Optional[str]] = mapped_column(String(100))
    tags: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SubscriberModel(Base):
    """Subscriber database model"""
    __tablename__ = "subscribers"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    company: Mapped[Optional[str]] = mapped_column(String(200))
    job_title: Mapped[Optional[str]] = mapped_column(String(200))
    custom_fields: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="active")
    tags: Mapped[Optional[list]] = mapped_column(JSON)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class SequenceSubscriberModel(Base):
    """Sequence subscriber relationship model"""
    __tablename__ = "sequence_subscribers"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sequence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("email_sequences.id"), nullable=False)
    subscriber_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("subscribers.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active")
    current_step: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class EmailCampaignModel(Base):
    """Email campaign database model"""
    __tablename__ = "email_campaigns"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    sequence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("email_sequences.id"), nullable=False)
    target_segments: Mapped[list] = mapped_column(JSON)
    scheduled_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    status: Mapped[str] = mapped_column(String(50), default="draft")
    tags: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    launched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class AnalyticsEventModel(Base):
    """Analytics event database model"""
    __tablename__ = "analytics_events"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    sequence_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("email_sequences.id"))
    step_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("sequence_steps.id"))
    subscriber_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("subscribers.id"))
    campaign_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("email_campaigns.id"))
    email_address: Mapped[Optional[str]] = mapped_column(String(255))
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Database connection management
class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_maker: Optional[async_sessionmaker[AsyncSession]] = None
    
    async def initialize(self) -> None:
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.debug,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session"""
        if not self.session_maker:
            raise RuntimeError("Database not initialized")
        
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            if not self.engine:
                return False
            
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with db_manager.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database"""
    await db_manager.initialize()


async def close_database() -> None:
    """Close database"""
    await db_manager.close()


async def check_database_health() -> bool:
    """Check database health"""
    return await db_manager.health_check()






























