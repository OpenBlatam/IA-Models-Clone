"""
Workflow Models
===============

Workflow-related database models.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import Dict, Any

from .base import Base, workflow_agents, workflow_templates

class Workflow(Base):
    """Workflow model for business process automation."""
    
    __tablename__ = 'workflows'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    business_area = Column(String(50), nullable=False, index=True)
    steps = Column(JSON, default=list)
    configuration = Column(JSON, default=dict)
    status = Column(String(20), default="draft", nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    version = Column(String(20), default="1.0.0", nullable=False)
    created_by = Column(String, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    created_by_user = relationship("User", back_populates="workflows")
    agents = relationship("BusinessAgent", secondary=workflow_agents, back_populates="workflows")
    templates = relationship("Template", secondary=workflow_templates, back_populates="workflows")
    executions = relationship("WorkflowExecution", back_populates="workflow")
    documents = relationship("Document", back_populates="workflow")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'business_area': self.business_area,
            'steps': self.steps,
            'configuration': self.configuration,
            'status': self.status,
            'is_active': self.is_active,
            'version': self.version,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class WorkflowExecution(Base):
    """Workflow execution model for tracking workflow runs."""
    
    __tablename__ = 'workflow_executions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey('workflows.id'), nullable=False)
    status = Column(String(20), default="running", nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    execution_log = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    agent_executions = relationship("AgentExecution", back_populates="workflow_execution")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'status': self.status,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration': self.duration,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'execution_log': self.execution_log,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
