from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Database models for HeyGen AI API
SQLAlchemy ORM models for all database entities.
"""


Base = declarative_base()


class User(Base):
    """User model for authentication and user management"""
    __tablename__ = "users"
    
    user_id = Column(String(50), primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "api_key": self.api_key,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class Video(Base):
    """Video model for video generation and management"""
    __tablename__ = "videos"
    
    video_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    script = Column(Text, nullable=False)
    voice_id = Column(String(50), nullable=False)
    language = Column(String(10), nullable=False)
    quality = Column(String(20), nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    file_path = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    estimated_duration = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="videos")
    processing_jobs = relationship("VideoProcessingJob", back_populates="video", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "video_id": self.video_id,
            "user_id": self.user_id,
            "script": self.script,
            "voice_id": self.voice_id,
            "language": self.language,
            "quality": self.quality,
            "status": self.status,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "estimated_duration": self.estimated_duration,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class APIKey(Base):
    """API Key model for API authentication"""
    __tablename__ = "api_keys"
    
    key_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    permissions = Column(JSON, nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "api_key": self.api_key,
            "name": self.name,
            "is_active": self.is_active,
            "permissions": self.permissions,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class UserSession(Base):
    """User session model for session management"""
    __tablename__ = "user_sessions"
    
    session_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    access_token = Column(String(255), nullable=False)
    refresh_token = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "is_active": self.is_active,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


class VideoProcessingJob(Base):
    """Video processing job model for background task management"""
    __tablename__ = "video_processing_jobs"
    
    job_id = Column(String(50), primary_key=True, index=True)
    video_id = Column(String(50), ForeignKey("videos.video_id"), nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    priority = Column(Integer, default=0, nullable=False)
    worker_id = Column(String(50), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    progress = Column(Float, default=0.0, nullable=False)
    settings = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="processing_jobs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status,
            "priority": self.priority,
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress": self.progress,
            "settings": self.settings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class SystemLog(Base):
    """System log model for logging and monitoring"""
    __tablename__ = "system_logs"
    
    log_id = Column(String(50), primary_key=True, index=True)
    level = Column(String(10), nullable=False, index=True)
    module = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "log_id": self.log_id,
            "level": self.level,
            "module": self.module,
            "message": self.message,
            "details": self.details,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AnalyticsEvent(Base):
    """Analytics event model for tracking user behavior"""
    __tablename__ = "analytics_events"
    
    event_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSON, nullable=True)
    session_id = Column(String(50), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# Named exports
__all__ = [
    "Base",
    "User",
    "Video", 
    "APIKey",
    "UserSession",
    "VideoProcessingJob",
    "SystemLog",
    "AnalyticsEvent"
] 