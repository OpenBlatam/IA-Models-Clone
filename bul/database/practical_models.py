"""
BUL System - Practical Database Models
Real, practical database models for the BUL system
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """Real user model"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    usage_stats = relationship("UsageStats", back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    """Real document model"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    template_type = Column(String(100), nullable=False, index=True)
    language = Column(String(10), default="es", nullable=False)
    format = Column(String(20), default="pdf", nullable=False)
    status = Column(String(20), default="draft", nullable=False, index=True)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")

class DocumentVersion(Base):
    """Real document version model"""
    __tablename__ = "document_versions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    changes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="versions")
    creator = relationship("User")

class APIKey(Base):
    """Real API key model"""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    permissions = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class UsageStats(Base):
    """Real usage statistics model"""
    __tablename__ = "usage_stats"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    date = Column(DateTime, default=func.now(), nullable=False, index=True)
    documents_created = Column(Integer, default=0, nullable=False)
    api_calls = Column(Integer, default=0, nullable=False)
    processing_time = Column(Float, default=0.0, nullable=False)
    errors_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="usage_stats")

class Template(Base):
    """Real template model"""
    __tablename__ = "templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    template_type = Column(String(100), nullable=False, index=True)
    content = Column(Text, nullable=False)
    variables = Column(JSON, nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    creator = relationship("User")

class SystemLog(Base):
    """Real system log model"""
    __tablename__ = "system_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship("User")

class RateLimit(Base):
    """Real rate limit model"""
    __tablename__ = "rate_limits"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    requests_count = Column(Integer, default=0, nullable=False)
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False, index=True)
    is_blocked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User")

class AIConfig(Base):
    """Real AI configuration model"""
    __tablename__ = "ai_configs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    api_key = Column(String(500), nullable=False)
    max_tokens = Column(Integer, default=2000, nullable=False)
    temperature = Column(Float, default=0.7, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    usage_stats = relationship("AIUsageStats", back_populates="ai_config", cascade="all, delete-orphan")

class AIUsageStats(Base):
    """Real AI usage statistics model"""
    __tablename__ = "ai_usage_stats"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ai_config_id = Column(String(36), ForeignKey("ai_configs.id"), nullable=False, index=True)
    date = Column(DateTime, default=func.now(), nullable=False, index=True)
    requests_count = Column(Integer, default=0, nullable=False)
    tokens_used = Column(Integer, default=0, nullable=False)
    cost = Column(Float, default=0.0, nullable=False)
    response_time = Column(Float, default=0.0, nullable=False)
    errors_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    ai_config = relationship("AIConfig", back_populates="usage_stats")

class Workflow(Base):
    """Real workflow model"""
    __tablename__ = "workflows"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    steps = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    creator = relationship("User")
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")

class WorkflowExecution(Base):
    """Real workflow execution model"""
    __tablename__ = "workflow_executions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(20), default="pending", nullable=False, index=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    user = relationship("User")













