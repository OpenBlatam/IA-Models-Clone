"""
AI Integration System - Database Models
SQLAlchemy models for tracking integrations, platforms, and content
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum

Base = declarative_base()

class IntegrationStatus(str, Enum):
    """Integration status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"

class ContentType(str, Enum):
    """Content type enumeration"""
    BLOG_POST = "blog_post"
    EMAIL_CAMPAIGN = "email_campaign"
    SOCIAL_MEDIA_POST = "social_media_post"
    PRODUCT_DESCRIPTION = "product_description"
    LANDING_PAGE = "landing_page"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    CONTACT = "contact"
    LEAD = "lead"
    OPPORTUNITY = "opportunity"

class PlatformType(str, Enum):
    """Platform type enumeration"""
    CRM = "crm"
    CMS = "cms"
    EMAIL_MARKETING = "email_marketing"
    SOCIAL_MEDIA = "social_media"
    ECOMMERCE = "ecommerce"
    COMMUNICATION = "communication"

class IntegrationRequest(Base):
    """Integration request model"""
    __tablename__ = "integration_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String(255), unique=True, index=True, nullable=False)
    content_type = Column(SQLEnum(ContentType), nullable=False)
    content_data = Column(JSON, nullable=False)
    target_platforms = Column(JSON, nullable=False)  # List of platform names
    priority = Column(Integer, default=1)
    max_retries = Column(Integer, default=3)
    retry_count = Column(Integer, default=0)
    status = Column(SQLEnum(IntegrationStatus), default=IntegrationStatus.PENDING)
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    results = relationship("IntegrationResult", back_populates="request", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<IntegrationRequest(id={self.id}, content_id='{self.content_id}', status='{self.status}')>"

class IntegrationResult(Base):
    """Integration result model"""
    __tablename__ = "integration_results"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("integration_requests.id"), nullable=False)
    platform = Column(String(100), nullable=False)
    status = Column(SQLEnum(IntegrationStatus), nullable=False)
    external_id = Column(String(255), nullable=True)  # ID from external platform
    error_message = Column(Text, nullable=True)
    response_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    request = relationship("IntegrationRequest", back_populates="results")
    
    def __repr__(self):
        return f"<IntegrationResult(id={self.id}, platform='{self.platform}', status='{self.status}')>"

class Platform(Base):
    """Platform configuration model"""
    __tablename__ = "platforms"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    platform_type = Column(SQLEnum(PlatformType), nullable=False)
    enabled = Column(Boolean, default=True)
    configuration = Column(JSON, nullable=True)  # Platform-specific config
    api_endpoints = Column(JSON, nullable=True)  # API endpoint configurations
    rate_limits = Column(JSON, nullable=True)  # Rate limiting configuration
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    health_status = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Platform(id={self.id}, name='{self.name}', enabled={self.enabled})>"

class WebhookEvent(Base):
    """Webhook event model"""
    __tablename__ = "webhook_events"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(100), nullable=False)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    
    # Timestamps
    received_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<WebhookEvent(id={self.id}, platform='{self.platform}', event_type='{self.event_type}')>"

class IntegrationMetrics(Base):
    """Integration metrics model"""
    __tablename__ = "integration_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(100), nullable=False)
    metric_type = Column(String(100), nullable=False)  # success_rate, response_time, etc.
    metric_value = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<IntegrationMetrics(id={self.id}, platform='{self.platform}', metric_type='{self.metric_type}')>"

class ContentTemplate(Base):
    """Content template model"""
    __tablename__ = "content_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    content_type = Column(SQLEnum(ContentType), nullable=False)
    template_data = Column(JSON, nullable=False)  # Template structure
    platform_mappings = Column(JSON, nullable=True)  # Platform-specific mappings
    enabled = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ContentTemplate(id={self.id}, name='{self.name}', content_type='{self.content_type}')>"

class IntegrationLog(Base):
    """Integration log model for audit trail"""
    __tablename__ = "integration_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("integration_requests.id"), nullable=True)
    platform = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)  # create, update, delete, etc.
    status = Column(SQLEnum(IntegrationStatus), nullable=False)
    message = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    request = relationship("IntegrationRequest")
    
    def __repr__(self):
        return f"<IntegrationLog(id={self.id}, platform='{self.platform}', action='{self.action}')>"

class UserSession(Base):
    """User session model for API authentication"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False)
    session_token = Column(String(500), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id='{self.user_id}', active={self.is_active})>"

class SystemConfiguration(Base):
    """System configuration model"""
    __tablename__ = "system_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(200), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    is_encrypted = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<SystemConfiguration(id={self.id}, key='{self.key}')>"

# Indexes for better performance
from sqlalchemy import Index

# Create indexes
Index('idx_integration_requests_content_id', IntegrationRequest.content_id)
Index('idx_integration_requests_status', IntegrationRequest.status)
Index('idx_integration_requests_created_at', IntegrationRequest.created_at)
Index('idx_integration_results_platform', IntegrationResult.platform)
Index('idx_integration_results_status', IntegrationResult.status)
Index('idx_webhook_events_platform', WebhookEvent.platform)
Index('idx_webhook_events_processed', WebhookEvent.processed)
Index('idx_integration_metrics_platform_timestamp', IntegrationMetrics.platform, IntegrationMetrics.timestamp)
Index('idx_integration_logs_platform', IntegrationLog.platform)
Index('idx_integration_logs_created_at', IntegrationLog.created_at)



























