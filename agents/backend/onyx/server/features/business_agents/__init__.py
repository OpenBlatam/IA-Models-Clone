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

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Comprehensive business agent system with workflow management and document generation"

# Data Models
try:
    from .agent_models import BusinessAgent, BusinessArea, AgentCapability
except Exception:
    BusinessAgent = None
    BusinessArea = None
    AgentCapability = None

# Core Components
try:
    from .workflow_engine import WorkflowEngine
except Exception:
    WorkflowEngine = None

try:
    from .document_generator import DocumentGenerator
except Exception:
    DocumentGenerator = None

# Manager
try:
    from .manager import BusinessAgentManager
except Exception:
    try:
        from .business_agents import BusinessAgentManager
    except Exception:
        BusinessAgentManager = None

# API
try:
    from .api import BusinessAgentsAPI
except Exception:
    print("Failed to import API")
    BusinessAgentsAPI = None

__all__ = [
    'BusinessAgent',
    'BusinessArea',
    'AgentCapability',
    'WorkflowEngine',
    'DocumentGenerator', 
    'BusinessAgentManager',
    'BusinessAgentsAPI'
]
