"""
System Models
=============

System-related database models for metrics, alerts, integrations, etc.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, JSON
from datetime import datetime
import uuid
from typing import Dict, Any

from .base import Base

class Metric(Base):
    """Metric model for system metrics and analytics."""
    
    __tablename__ = 'metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False, index=True)
    tags = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'value': self.value,
            'metric_type': self.metric_type,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class Alert(Base):
    """Alert model for system alerts and notifications."""
    
    __tablename__ = 'alerts'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    level = Column(String(20), nullable=False, index=True)
    source = Column(String(100), nullable=False, index=True)
    is_resolved = Column(Boolean, default=False, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level,
            'source': self.source,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

class Integration(Base):
    """Integration model for external system integrations."""
    
    __tablename__ = 'integrations'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    integration_type = Column(String(50), nullable=False, index=True)
    base_url = Column(String(500), nullable=False)
    authentication_type = Column(String(50), nullable=False)
    credentials = Column(JSON, default=dict)
    headers = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'integration_type': self.integration_type,
            'base_url': self.base_url,
            'authentication_type': self.authentication_type,
            'credentials': self.credentials,
            'headers': self.headers,
            'configuration': self.configuration,
            'is_active': self.is_active,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class MLPipeline(Base):
    """ML Pipeline model for machine learning pipelines."""
    
    __tablename__ = 'ml_pipelines'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    feature_columns = Column(JSON, default=list)
    target_column = Column(String(100), nullable=True)
    model_configuration = Column(JSON, default=dict)
    training_metrics = Column(JSON, default=dict)
    is_trained = Column(Boolean, default=False, nullable=False, index=True)
    last_trained = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_configuration': self.model_configuration,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class EnhancementRequest(Base):
    """Enhancement request model for AI content enhancement."""
    
    __tablename__ = 'enhancement_requests'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    enhancement_type = Column(String(50), nullable=False, index=True)
    target_audience = Column(String(100), nullable=True)
    business_context = Column(JSON, default=dict)
    quality_requirements = Column(JSON, default=dict)
    language = Column(String(10), default="en", nullable=False)
    tone = Column(String(50), nullable=True)
    keywords = Column(JSON, default=list)
    max_length = Column(Integer, nullable=True)
    min_length = Column(Integer, nullable=True)
    status = Column(String(20), default="pending", nullable=False, index=True)
    result = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'enhancement_type': self.enhancement_type,
            'target_audience': self.target_audience,
            'business_context': self.business_context,
            'quality_requirements': self.quality_requirements,
            'language': self.language,
            'tone': self.tone,
            'keywords': self.keywords,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'status': self.status,
            'result': self.result,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata
        }
