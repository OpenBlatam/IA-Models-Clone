"""
Document Models
===============

Document and template-related database models.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import Dict, Any

from .base import Base, workflow_templates

class Document(Base):
    """Document model for generated business documents."""
    
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False, index=True)
    document_type = Column(String(50), nullable=False, index=True)
    business_area = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=True)
    format = Column(String(20), default="markdown", nullable=False)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    status = Column(String(20), default="draft", nullable=False, index=True)
    created_by = Column(String, ForeignKey('users.id'), nullable=False)
    workflow_id = Column(String, ForeignKey('workflows.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    created_by_user = relationship("User", back_populates="documents")
    workflow = relationship("Workflow", back_populates="documents")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'document_type': self.document_type,
            'business_area': self.business_area,
            'content': self.content,
            'format': self.format,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'status': self.status,
            'created_by': self.created_by,
            'workflow_id': self.workflow_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class Template(Base):
    """Template model for reusable workflow templates."""
    
    __tablename__ = 'templates'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    business_area = Column(String(50), nullable=False, index=True)
    template_data = Column(JSON, default=dict)
    category = Column(String(50), nullable=True, index=True)
    tags = Column(JSON, default=list)
    is_public = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    rating = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    workflows = relationship("Workflow", secondary=workflow_templates, back_populates="templates")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'business_area': self.business_area,
            'template_data': self.template_data,
            'category': self.category,
            'tags': self.tags,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'rating': self.rating,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
