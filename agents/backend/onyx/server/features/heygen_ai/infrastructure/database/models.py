from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import (
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SQLAlchemy Database Models

ORM models that map domain entities to database tables.
"""

    String, Integer, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class TimestampMixin:
    """Mixin for timestamp fields."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class UserModel(Base, TimestampMixin):
    """
    User database model.
    
    Maps to the User domain entity.
    """
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )
    
    # User identity
    username: Mapped[Optional[str]] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=True
    )
    
    email: Mapped[Optional[str]] = mapped_column(
        String(254),
        unique=True,
        index=True,
        nullable=True
    )
    
    # Personal information
    first_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    last_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    is_premium: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    is_suspended: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )
    
    # Video credits and limits
    video_credits: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False
    )
    
    max_video_duration: Mapped[int] = mapped_column(
        Integer,
        default=60,  # seconds
        nullable=False
    )
    
    # Usage tracking
    videos_created_today: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    last_video_created: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    videos: Mapped[list["VideoModel"]] = relationship(
        "VideoModel",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_users_email_active", "email", "is_active"),
        Index("idx_users_username_active", "username", "is_active"),
        Index("idx_users_premium_active", "is_premium", "is_active"),
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("username", name="uq_users_username"),
    )
    
    def __repr__(self) -> str:
        return f"UserModel(id={self.id}, username={self.username}, email={self.email})"


class VideoModel(Base, TimestampMixin):
    """
    Video database model.
    
    Maps to the Video domain entity.
    """
    __tablename__ = "videos"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )
    
    # Foreign key to user
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Video metadata
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Video specifications
    duration_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    
    quality_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium"
    )
    
    quality_config: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Processing status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True
    )
    
    progress_percentage: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    status_message: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    error_details: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Processing timestamps
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # File information
    input_data: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    
    output_file_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    output_file_size: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )
    
    # Relationships
    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="videos"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_videos_user_status", "user_id", "status"),
        Index("idx_videos_user_created", "user_id", "created_at"),
        Index("idx_videos_status_created", "status", "created_at"),
        Index("idx_videos_user_active", "user_id", "is_deleted"),
    )
    
    def __repr__(self) -> str:
        return f"VideoModel(id={self.id}, title={self.title}, status={self.status})"


class APIKeyModel(Base, TimestampMixin):
    """
    API Key database model.
    
    For API key authentication.
    """
    __tablename__ = "api_keys"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )
    
    # Foreign key to user
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # API key data
    key_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        unique=True,
        index=True,
        nullable=False
    )
    
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    
    # Key status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    # Usage tracking
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Permissions (JSON array of permissions)
    permissions: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_api_keys_user_active", "user_id", "is_active"),
        Index("idx_api_keys_hash_active", "key_hash", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"APIKeyModel(id={self.id}, name={self.name}, active={self.is_active})"


class AuditLogModel(Base):
    """
    Audit log model for tracking user actions.
    """
    __tablename__ = "audit_logs"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )
    
    # User information
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Action details
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    
    resource_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    
    # Request context
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 support
        nullable=True
    )
    
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Additional data
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Status
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )
    
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Timestamp (no auto-update)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_audit_logs_user_action", "user_id", "action"),
        Index("idx_audit_logs_action_created", "action", "created_at"),
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
    )
    
    def __repr__(self) -> str:
        return f"AuditLogModel(id={self.id}, action={self.action}, user_id={self.user_id})" 