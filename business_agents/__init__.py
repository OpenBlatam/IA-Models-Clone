"""
Business Agents System
======================

A comprehensive agent system for all business areas with workflow management
and document generation capabilities.

Features:
- Multi-area business agents (Marketing, Sales, Operations, HR, Finance, etc.)
- Workflow creation and management
- Document generation system
- API endpoints for integration
- Real-time collaboration
"""

from .workflow_engine import WorkflowEngine
from .document_generator import DocumentGenerator
from .business_agents import BusinessAgentManager
from .api import BusinessAgentsAPI

__all__ = [
    'WorkflowEngine',
    'DocumentGenerator', 
    'BusinessAgentManager',
    'BusinessAgentsAPI'
]





























