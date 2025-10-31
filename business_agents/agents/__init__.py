"""
Business Agents Package
=======================

Modular business agent system with agent definitions, capabilities, and management.
"""

from .manager import BusinessAgentManager
from .definitions import BusinessArea, BusinessAgent, AgentCapability
from .capabilities import CapabilityRegistry, BaseCapability

__all__ = [
    "BusinessAgentManager",
    "BusinessArea",
    "BusinessAgent", 
    "AgentCapability",
    "CapabilityRegistry",
    "BaseCapability"
]
