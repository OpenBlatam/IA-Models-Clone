"""
Custom Exceptions
=================

Custom exception classes for the Business Agents System.
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, status

class BusinessAgentsException(Exception):
    """Base exception for Business Agents System."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AgentNotFoundError(BusinessAgentsException):
    """Raised when an agent is not found."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent {agent_id} not found",
            {"agent_id": agent_id, "error_code": "AGENT_NOT_FOUND"}
        )

class AgentInactiveError(BusinessAgentsException):
    """Raised when trying to use an inactive agent."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent {agent_id} is not active",
            {"agent_id": agent_id, "error_code": "AGENT_INACTIVE"}
        )

class CapabilityNotFoundError(BusinessAgentsException):
    """Raised when an agent capability is not found."""
    
    def __init__(self, agent_id: str, capability_name: str):
        super().__init__(
            f"Capability {capability_name} not found for agent {agent_id}",
            {
                "agent_id": agent_id,
                "capability_name": capability_name,
                "error_code": "CAPABILITY_NOT_FOUND"
            }
        )

class WorkflowNotFoundError(BusinessAgentsException):
    """Raised when a workflow is not found."""
    
    def __init__(self, workflow_id: str):
        super().__init__(
            f"Workflow {workflow_id} not found",
            {"workflow_id": workflow_id, "error_code": "WORKFLOW_NOT_FOUND"}
        )

class WorkflowExecutionError(BusinessAgentsException):
    """Raised when workflow execution fails."""
    
    def __init__(self, workflow_id: str, reason: str):
        super().__init__(
            f"Workflow {workflow_id} execution failed: {reason}",
            {"workflow_id": workflow_id, "reason": reason, "error_code": "WORKFLOW_EXECUTION_ERROR"}
        )

class DocumentGenerationError(BusinessAgentsException):
    """Raised when document generation fails."""
    
    def __init__(self, document_type: str, reason: str):
        super().__init__(
            f"Document generation failed for {document_type}: {reason}",
            {"document_type": document_type, "reason": reason, "error_code": "DOCUMENT_GENERATION_ERROR"}
        )

class DocumentNotFoundError(BusinessAgentsException):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str):
        super().__init__(
            f"Document {document_id} not found",
            {"document_id": document_id, "error_code": "DOCUMENT_NOT_FOUND"}
        )

class ValidationError(BusinessAgentsException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(
            f"Validation error for {field}: {message}",
            {"field": field, "message": message, "value": value, "error_code": "VALIDATION_ERROR"}
        )

class ConfigurationError(BusinessAgentsException):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, message: str):
        super().__init__(
            f"Configuration error for {config_key}: {message}",
            {"config_key": config_key, "message": message, "error_code": "CONFIGURATION_ERROR"}
        )

class ServiceUnavailableError(BusinessAgentsException):
    """Raised when a service is unavailable."""
    
    def __init__(self, service_name: str, reason: str = "Service unavailable"):
        super().__init__(
            f"Service {service_name} is unavailable: {reason}",
            {"service_name": service_name, "reason": reason, "error_code": "SERVICE_UNAVAILABLE"}
        )

# HTTP Exception mappings
EXCEPTION_TO_HTTP = {
    AgentNotFoundError: status.HTTP_404_NOT_FOUND,
    AgentInactiveError: status.HTTP_400_BAD_REQUEST,
    CapabilityNotFoundError: status.HTTP_404_NOT_FOUND,
    WorkflowNotFoundError: status.HTTP_404_NOT_FOUND,
    WorkflowExecutionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    DocumentGenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    DocumentNotFoundError: status.HTTP_404_NOT_FOUND,
    ValidationError: status.HTTP_400_BAD_REQUEST,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}

def convert_to_http_exception(exc: BusinessAgentsException) -> HTTPException:
    """Convert BusinessAgentsException to HTTPException."""
    status_code = EXCEPTION_TO_HTTP.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return HTTPException(
        status_code=status_code,
        detail={
            "message": exc.message,
            "details": exc.details
        }
    )
