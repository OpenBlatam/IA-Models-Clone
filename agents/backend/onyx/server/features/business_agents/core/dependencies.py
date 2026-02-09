"""
Dependencies
============

FastAPI dependency functions for service injection.
"""

from fastapi import Depends
from typing import Dict, Any

from .container import get_container
from ..business_agents import BusinessAgentManager
from ..services import (
    HealthService, SystemInfoService, MetricsService,
    AgentService, WorkflowService, DocumentService
)

def get_agent_manager() -> BusinessAgentManager:
    """Get the business agent manager instance."""
    container = get_container()
    return container.get("agent_manager")

def get_health_service() -> HealthService:
    """Get the health service instance."""
    container = get_container()
    return container.get("health_service")

def get_system_info_service() -> SystemInfoService:
    """Get the system info service instance."""
    container = get_container()
    return container.get("system_info_service")

def get_metrics_service() -> MetricsService:
    """Get the metrics service instance."""
    container = get_container()
    return container.get("metrics_service")

def get_agent_service() -> AgentService:
    """Get the agent service instance."""
    container = get_container()
    return container.get("agent_service")

def get_workflow_service() -> WorkflowService:
    """Get the workflow service instance."""
    container = get_container()
    return container.get("workflow_service")

def get_document_service() -> DocumentService:
    """Get the document service instance."""
    container = get_container()
    return container.get("document_service")

def get_services() -> Dict[str, Any]:
    """Get all services as a dictionary."""
    container = get_container()
    return {
        "agent_manager": container.get("agent_manager"),
        "health_service": container.get("health_service"),
        "system_info_service": container.get("system_info_service"),
        "metrics_service": container.get("metrics_service"),
        "agent_service": container.get("agent_service"),
        "workflow_service": container.get("workflow_service"),
        "document_service": container.get("document_service")
    }
