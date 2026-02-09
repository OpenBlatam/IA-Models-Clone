"""
Agent Models
=============

Business agent-related database models.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import Dict, Any

from .base import Base, workflow_agents

class BusinessAgent(Base):
    """Business Agent model for different business areas."""
    
    __tablename__ = 'business_agents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    business_area = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)
    capabilities = Column(JSON, default=list)
    configuration = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    version = Column(String(20), default="1.0.0", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    workflows = relationship("Workflow", secondary=workflow_agents, back_populates="agents")
    executions = relationship("AgentExecution", back_populates="agent")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'business_area': self.business_area,
            'description': self.description,
            'capabilities': self.capabilities,
            'configuration': self.configuration,
            'is_active': self.is_active,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class AgentExecution(Base):
    """Agent execution model for tracking individual agent runs."""
    
    __tablename__ = 'agent_executions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('business_agents.id'), nullable=False)
    workflow_execution_id = Column(String, ForeignKey('workflow_executions.id'), nullable=True)
    capability_name = Column(String(100), nullable=False)
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
    agent = relationship("BusinessAgent", back_populates="executions")
    workflow_execution = relationship("WorkflowExecution", back_populates="agent_executions")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'workflow_execution_id': self.workflow_execution_id,
            'capability_name': self.capability_name,
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
