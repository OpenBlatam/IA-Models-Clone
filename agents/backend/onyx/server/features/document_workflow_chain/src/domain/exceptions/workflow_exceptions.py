"""
Workflow Domain Exceptions
=========================

Domain-specific exceptions for workflow operations.
"""


class WorkflowDomainException(Exception):
    """Base exception for workflow domain errors"""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "WORKFLOW_DOMAIN_ERROR"


class WorkflowNotFoundError(WorkflowDomainException):
    """Exception raised when workflow is not found"""
    
    def __init__(self, workflow_id: str):
        super().__init__(f"Workflow {workflow_id} not found", "WORKFLOW_NOT_FOUND")
        self.workflow_id = workflow_id


class WorkflowNameAlreadyExistsError(WorkflowDomainException):
    """Exception raised when workflow name already exists"""
    
    def __init__(self, name: str):
        super().__init__(f"Workflow name '{name}' already exists", "WORKFLOW_NAME_EXISTS")
        self.name = name


class InvalidWorkflowStatusError(WorkflowDomainException):
    """Exception raised when workflow status is invalid"""
    
    def __init__(self, status: str):
        super().__init__(f"Invalid workflow status: {status}", "INVALID_WORKFLOW_STATUS")
        self.status = status


class WorkflowStatusTransitionError(WorkflowDomainException):
    """Exception raised when workflow status transition is invalid"""
    
    def __init__(self, from_status: str, to_status: str):
        super().__init__(
            f"Invalid status transition from {from_status} to {to_status}",
            "INVALID_STATUS_TRANSITION"
        )
        self.from_status = from_status
        self.to_status = to_status


class WorkflowMaxNodesExceededError(WorkflowDomainException):
    """Exception raised when workflow exceeds maximum nodes"""
    
    def __init__(self, max_nodes: int):
        super().__init__(
            f"Workflow cannot have more than {max_nodes} nodes",
            "MAX_NODES_EXCEEDED"
        )
        self.max_nodes = max_nodes


class WorkflowMaxDepthExceededError(WorkflowDomainException):
    """Exception raised when workflow exceeds maximum depth"""
    
    def __init__(self, max_depth: int):
        super().__init__(
            f"Workflow cannot exceed depth of {max_depth}",
            "MAX_DEPTH_EXCEEDED"
        )
        self.max_depth = max_depth


class WorkflowCircularReferenceError(WorkflowDomainException):
    """Exception raised when workflow has circular reference"""
    
    def __init__(self, node_id: str):
        super().__init__(
            f"Circular reference detected involving node {node_id}",
            "CIRCULAR_REFERENCE"
        )
        self.node_id = node_id


class WorkflowValidationError(WorkflowDomainException):
    """Exception raised when workflow validation fails"""
    
    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error for field '{field}': {message}", "VALIDATION_ERROR")
        self.field = field


class WorkflowConcurrencyError(WorkflowDomainException):
    """Exception raised when workflow concurrency conflict occurs"""
    
    def __init__(self, workflow_id: str, expected_version: int, actual_version: int):
        super().__init__(
            f"Concurrency conflict for workflow {workflow_id}. Expected version {expected_version}, got {actual_version}",
            "CONCURRENCY_CONFLICT"
        )
        self.workflow_id = workflow_id
        self.expected_version = expected_version
        self.actual_version = actual_version




