"""
SQLAlchemy Models
=================

Database models for the workflow chain system.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from sqlalchemy import (
    Column, String, Integer, DateTime, Text, Boolean, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from ...domain.value_objects.workflow_status import WorkflowStatus
from ...domain.value_objects.priority import Priority


Base = declarative_base()


class WorkflowChainModel(Base):
    """Workflow chain database model"""
    __tablename__ = "workflow_chains"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default=WorkflowStatus.CREATED.value, index=True)
    
    # Settings and configuration
    settings = Column(JSONB, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Version for optimistic locking
    version = Column(Integer, nullable=False, default=1)
    
    # Relationships
    nodes = relationship("WorkflowNodeModel", back_populates="workflow", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('created', 'active', 'paused', 'completed', 'error', 'cancelled', 'deleted')", name="ck_workflow_status"),
        CheckConstraint("version > 0", name="ck_workflow_version"),
        Index("idx_workflow_name_status", "name", "status"),
        Index("idx_workflow_created_at", "created_at"),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate workflow status"""
        valid_statuses = [s.value for s in WorkflowStatus]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "settings": self.settings or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "nodes_count": len(self.nodes) if self.nodes else 0
        }


class WorkflowNodeModel(Base):
    """Workflow node database model"""
    __tablename__ = "workflow_nodes"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    
    # Foreign key to workflow
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflow_chains.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Basic information
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    
    # Hierarchy
    parent_id = Column(UUID(as_uuid=True), ForeignKey("workflow_nodes.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Priority and status
    priority = Column(Integer, nullable=False, default=Priority.NORMAL.value, index=True)
    status = Column(String(50), nullable=False, default=WorkflowStatus.CREATED.value, index=True)
    
    # Tags and metadata
    tags = Column(JSONB, nullable=True, default=[])
    metadata = Column(JSONB, nullable=True, default={})
    
    # Content metrics
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    sentence_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    reading_time_minutes = Column(Integer, nullable=True)
    
    # Quality scores
    overall_score = Column(Integer, nullable=True)
    readability_score = Column(Integer, nullable=True)
    sentiment_score = Column(Integer, nullable=True)
    seo_score = Column(Integer, nullable=True)
    grammar_score = Column(Integer, nullable=True)
    coherence_score = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Version for optimistic locking
    version = Column(Integer, nullable=False, default=1)
    
    # Relationships
    workflow = relationship("WorkflowChainModel", back_populates="nodes")
    parent = relationship("WorkflowNodeModel", remote_side=[id], backref="children")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("priority BETWEEN 1 AND 5", name="ck_node_priority"),
        CheckConstraint("status IN ('created', 'active', 'paused', 'completed', 'error', 'cancelled', 'deleted')", name="ck_node_status"),
        CheckConstraint("version > 0", name="ck_node_version"),
        CheckConstraint("word_count >= 0", name="ck_word_count"),
        CheckConstraint("character_count >= 0", name="ck_character_count"),
        CheckConstraint("sentence_count >= 0", name="ck_sentence_count"),
        CheckConstraint("paragraph_count >= 0", name="ck_paragraph_count"),
        CheckConstraint("reading_time_minutes >= 0", name="ck_reading_time"),
        CheckConstraint("overall_score BETWEEN 0 AND 100", name="ck_overall_score"),
        CheckConstraint("readability_score BETWEEN 0 AND 100", name="ck_readability_score"),
        CheckConstraint("sentiment_score BETWEEN -100 AND 100", name="ck_sentiment_score"),
        CheckConstraint("seo_score BETWEEN 0 AND 100", name="ck_seo_score"),
        CheckConstraint("grammar_score BETWEEN 0 AND 100", name="ck_grammar_score"),
        CheckConstraint("coherence_score BETWEEN 0 AND 100", name="ck_coherence_score"),
        Index("idx_node_workflow_priority", "workflow_id", "priority"),
        Index("idx_node_workflow_status", "workflow_id", "status"),
        Index("idx_node_parent", "parent_id"),
        Index("idx_node_created_at", "created_at"),
        Index("idx_node_title", "title"),
    )
    
    @validates('priority')
    def validate_priority(self, key, priority):
        """Validate node priority"""
        if not (1 <= priority <= 5):
            raise ValueError(f"Priority must be between 1 and 5, got {priority}")
        return priority
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate node status"""
        valid_statuses = [s.value for s in WorkflowStatus]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "title": self.title,
            "content": self.content,
            "prompt": self.prompt,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "priority": self.priority,
            "status": self.status,
            "tags": self.tags or [],
            "metadata": self.metadata or {},
            "content_metrics": {
                "word_count": self.word_count,
                "character_count": self.character_count,
                "sentence_count": self.sentence_count,
                "paragraph_count": self.paragraph_count,
                "reading_time_minutes": self.reading_time_minutes
            },
            "quality_scores": {
                "overall_score": self.overall_score,
                "readability_score": self.readability_score,
                "sentiment_score": self.sentiment_score,
                "seo_score": self.seo_score,
                "grammar_score": self.grammar_score,
                "coherence_score": self.coherence_score
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version
        }


class UserModel(Base):
    """User database model"""
    __tablename__ = "users"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    
    # Basic information
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(255), nullable=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    
    # Roles and permissions
    roles = Column(JSONB, nullable=True, default=[])
    permissions = Column(JSONB, nullable=True, default=[])
    
    # Profile information
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    timezone = Column(String(50), nullable=True, default="UTC")
    language = Column(String(10), nullable=True, default="en")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    workflows = relationship("WorkflowChainModel", backref="owner")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("length(username) >= 3", name="ck_username_length"),
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name="ck_email_format"),
        Index("idx_user_active", "is_active"),
        Index("idx_user_created_at", "created_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "roles": self.roles or [],
            "permissions": self.permissions or [],
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "timezone": self.timezone,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class AuditLogModel(Base):
    """Audit log database model"""
    __tablename__ = "audit_logs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    
    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_id = Column(String(100), nullable=False, index=True)
    
    # User and session information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Request information
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    
    # Resource information
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    action = Column(String(50), nullable=False, index=True)
    
    # Event details
    details = Column(JSONB, nullable=True, default={})
    result = Column(String(20), nullable=False, default="success", index=True)
    error_message = Column(Text, nullable=True)
    
    # Performance
    duration_ms = Column(Integer, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    # Relationships
    user = relationship("UserModel", backref="audit_logs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("result IN ('success', 'failure', 'error')", name="ck_audit_result"),
        CheckConstraint("duration_ms >= 0", name="ck_audit_duration"),
        Index("idx_audit_event_type_timestamp", "event_type", "timestamp"),
        Index("idx_audit_user_timestamp", "user_id", "timestamp"),
        Index("idx_audit_resource_timestamp", "resource_type", "resource_id", "timestamp"),
        Index("idx_audit_timestamp", "timestamp"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "event_id": self.event_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details or {},
            "result": self.result,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class SystemMetricsModel(Base):
    """System metrics database model"""
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    
    # Metric information
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)
    metric_value = Column(Integer, nullable=False)
    
    # Labels and metadata
    labels = Column(JSONB, nullable=True, default={})
    metadata = Column(JSONB, nullable=True, default={})
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("metric_type IN ('counter', 'gauge', 'histogram', 'summary')", name="ck_metric_type"),
        Index("idx_metrics_name_timestamp", "metric_name", "timestamp"),
        Index("idx_metrics_timestamp", "timestamp"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "metric_value": self.metric_value,
            "labels": self.labels or {},
            "metadata": self.metadata or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }




