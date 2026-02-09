"""
Database Models
===============

Ultra-advanced database models with SQLAlchemy.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    optimization_sessions = relationship("OptimizationSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<User {self.username}>'

class Document(Base):
    """Document model."""
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default='pending', nullable=False, index=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    optimization_sessions = relationship("OptimizationSession", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Document {self.title}>'

class OptimizationSession(Base):
    """Optimization session model."""
    __tablename__ = 'optimization_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    session_name = Column(String(255), nullable=False)
    optimization_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default='pending', nullable=False, index=True)
    parameters = Column(JSON)
    results = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="optimization_sessions")
    document = relationship("Document", back_populates="optimization_sessions")
    
    def __repr__(self):
        return f'<OptimizationSession {self.session_name}>'

class PerformanceMetric(Base):
    """Performance metric model."""
    __tablename__ = 'performance_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('optimization_sessions.id'), nullable=False)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON)
    
    # Relationships
    session = relationship("OptimizationSession")
    
    def __repr__(self):
        return f'<PerformanceMetric {self.metric_name}: {self.metric_value}>'

class SystemLog(Base):
    """System log model."""
    __tablename__ = 'system_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    level = Column(String(20), nullable=False, index=True)
    logger_name = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module = Column(String(100))
    function = Column(String(100))
    line_number = Column(Integer)
    exception_info = Column(Text)
    request_id = Column(String(100), index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f'<SystemLog {self.level}: {self.message[:50]}...>'

class CacheEntry(Base):
    """Cache entry model."""
    __tablename__ = 'cache_entries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    cache_value = Column(Text, nullable=False)
    cache_type = Column(String(50), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<CacheEntry {self.cache_key}>'

class APIRateLimit(Base):
    """API rate limit model."""
    __tablename__ = 'api_rate_limits'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint = Column(String(255), nullable=False, index=True)
    ip_address = Column(String(45), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    request_count = Column(Integer, default=1, nullable=False)
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f'<APIRateLimit {self.endpoint}: {self.request_count}>'

class SecurityEvent(Base):
    """Security event model."""
    __tablename__ = 'security_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    description = Column(Text, nullable=False)
    ip_address = Column(String(45), index=True)
    user_agent = Column(String(500))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    request_id = Column(String(100), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f'<SecurityEvent {self.event_type}: {self.severity}>'

class OptimizationConfig(Base):
    """Optimization configuration model."""
    __tablename__ = 'optimization_configs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_name = Column(String(100), unique=True, nullable=False, index=True)
    config_type = Column(String(50), nullable=False, index=True)
    parameters = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User")
    
    def __repr__(self):
        return f'<OptimizationConfig {self.config_name}>'

class ModelVersion(Base):
    """Model version model."""
    __tablename__ = 'model_versions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    model_path = Column(String(500), nullable=False)
    model_size = Column(Integer)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)
    
    # Relationships
    creator = relationship("User")
    
    def __repr__(self):
        return f'<ModelVersion {self.model_name}: {self.version}>'

class TaskQueue(Base):
    """Task queue model."""
    __tablename__ = 'task_queue'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_name = Column(String(100), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default='pending', nullable=False, index=True)
    priority = Column(Integer, default=1, nullable=False, index=True)
    parameters = Column(JSON)
    result = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, index=True)
    completed_at = Column(DateTime, index=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    creator = relationship("User")
    
    def __repr__(self):
        return f'<TaskQueue {self.task_name}: {self.status}>'