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

# Try to import components with error handling
try:
    from .workflow_engine import WorkflowEngine
except ImportError:
    WorkflowEngine = None

try:
    from .document_generator import DocumentGenerator
except ImportError:
    DocumentGenerator = None

try:
    from .business_agents import BusinessAgentManager
except ImportError:
    BusinessAgentManager = None

try:
    from .api import BusinessAgentsAPI
except ImportError:
    BusinessAgentsAPI = None

__all__ = [
    'WorkflowEngine',
    'DocumentGenerator', 
    'BusinessAgentManager',
    'BusinessAgentsAPI'
]























