from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Database models for OS Content UGC Video Generator
SQLAlchemy models for data persistence
"""


Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(254), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    video_requests = relationship("VideoRequest", back_populates="user")

class VideoRequest(Base):
    """Video request model"""
    __tablename__ = "video_requests"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    text_prompt = Column(Text, nullable=False)
    description = Column(Text)
    ugc_type = Column(String(50), default="ugc_video_ad")
    language = Column(String(10), default="es")
    duration_per_image = Column(Float, default=3.0)
    resolution_width = Column(Integer, default=1080)
    resolution_height = Column(Integer, default=1920)
    status = Column(String(20), default="queued")
    progress = Column(Float, default=0.0)
    estimated_duration = Column(Float)
    video_url = Column(String(500))
    local_path = Column(String(500))
    cdn_url = Column(String(500))
    nlp_analysis = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="video_requests")
    processing_tasks = relationship("ProcessingTask", back_populates="video_request")
    uploaded_files = relationship("UploadedFile", back_populates="video_request")

class ProcessingTask(Base):
    """Processing task model"""
    __tablename__ = "processing_tasks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_request_id = Column(String(36), ForeignKey("video_requests.id"), nullable=False)
    task_type = Column(String(50), nullable=False)  # video_creation, nlp_analysis, etc.
    priority = Column(String(20), default="normal")
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    retries = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    timeout = Column(Integer)  # seconds
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    result_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    video_request = relationship("VideoRequest", back_populates="processing_tasks")

class UploadedFile(Base):
    """Uploaded file model"""
    __tablename__ = "uploaded_files"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_request_id = Column(String(36), ForeignKey("video_requests.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    file_type = Column(String(20), nullable=False)  # image, video, audio
    checksum = Column(String(64))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)
    
    # Relationships
    video_request = relationship("VideoRequest", back_populates="uploaded_files")

class SystemMetrics(Base):
    """System metrics model"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_io = Column(JSON)
    cache_hit_rate = Column(Float)
    request_count = Column(Integer)
    error_count = Column(Integer)
    average_response_time = Column(Float)
    active_connections = Column(Integer)
    queue_size = Column(Integer)
    metrics_data = Column(JSON)

class CacheEntry(Base):
    """Cache entry model"""
    __tablename__ = "cache_entries"
    
    id = Column(String(64), primary_key=True)
    key = Column(String(255), nullable=False, index=True)
    value = Column(JSON)
    ttl = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    is_compressed = Column(Boolean, default=False)
    cache_level = Column(String(20), default="memory")  # memory, disk, redis 