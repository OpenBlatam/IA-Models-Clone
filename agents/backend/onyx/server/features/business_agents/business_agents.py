"""
Legacy shim for business_agents.py.
Please use the new modular structure:
- Models: .models
- Manager: .manager
- Defaults: .defaults
"""
import logging
import warnings

# Use relative imports from the new structure
from .agent_models import BusinessAgent, BusinessArea, AgentCapability
from .manager import BusinessAgentManager
from .workflow_engine import WorkflowEngine, Workflow, WorkflowStep, StepType
from .document_generator import DocumentGenerator, DocumentType, DocumentFormat

# Re-export everything for backward compatibility
__all__ = [
    "BusinessArea",
    "AgentCapability",
    "BusinessAgent",
    "BusinessAgentManager",
    "WorkflowEngine",
    "Workflow",
    "WorkflowStep",
    "StepType",
    "DocumentGenerator",
    "DocumentType",
    "DocumentFormat",
]

logger = logging.getLogger(__name__)

warnings.warn(
    "Importing from business_agents.py is deprecated. Use .manager and .models instead.",
    DeprecationWarning,
    stacklevel=2,
)
