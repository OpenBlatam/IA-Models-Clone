"""
Workflow Models
===============

Simple and clear workflow models for the Document Workflow Chain system.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship

from .base import BaseModel


class WorkflowChain(BaseModel):
    """Workflow chain model - simple and clear"""
    __tablename__ = "workflow_chains"
    
    # Basic fields
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="draft")
    priority = Column(String(20), default="medium")
    
    # Configuration
    config = Column(JSON, nullable=True)
    
    # Relationships
    nodes = relationship("WorkflowNode", back_populates="workflow", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WorkflowChain(id={self.id}, name='{self.name}', status='{self.status}')>"


class WorkflowNode(BaseModel):
    """Workflow node model - simple and clear"""
    __tablename__ = "workflow_nodes"
    
    # Basic fields
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    node_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    
    # Configuration
    config = Column(JSON, nullable=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    
    # Relationships
    workflow_id = Column(Integer, ForeignKey("workflow_chains.id"), nullable=False)
    workflow = relationship("WorkflowChain", back_populates="nodes")
    
    def __repr__(self):
        return f"<WorkflowNode(id={self.id}, name='{self.name}', type='{self.node_type}', status='{self.status}')>"


