"""
Node Domain Exceptions
=====================

Domain-specific exceptions for node operations.
"""


class NodeDomainException(Exception):
    """Base exception for node domain errors"""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "NODE_DOMAIN_ERROR"


class NodeNotFoundError(NodeDomainException):
    """Exception raised when node is not found"""
    
    def __init__(self, node_id: str):
        super().__init__(f"Node {node_id} not found", "NODE_NOT_FOUND")
        self.node_id = node_id


class NodeTitleAlreadyExistsError(NodeDomainException):
    """Exception raised when node title already exists in workflow"""
    
    def __init__(self, title: str):
        super().__init__(f"Node title '{title}' already exists in workflow", "NODE_TITLE_EXISTS")
        self.title = title


class InvalidNodeContentError(NodeDomainException):
    """Exception raised when node content is invalid"""
    
    def __init__(self, message: str):
        super().__init__(f"Invalid node content: {message}", "INVALID_NODE_CONTENT")


class InvalidNodePromptError(NodeDomainException):
    """Exception raised when node prompt is invalid"""
    
    def __init__(self, message: str):
        super().__init__(f"Invalid node prompt: {message}", "INVALID_NODE_PROMPT")


class NodeParentNotFoundError(NodeDomainException):
    """Exception raised when node parent is not found"""
    
    def __init__(self, parent_id: str):
        super().__init__(f"Parent node {parent_id} not found", "PARENT_NODE_NOT_FOUND")
        self.parent_id = parent_id


class NodeCircularReferenceError(NodeDomainException):
    """Exception raised when node has circular reference"""
    
    def __init__(self, node_id: str, parent_id: str):
        super().__init__(
            f"Circular reference detected: node {node_id} cannot have parent {parent_id}",
            "NODE_CIRCULAR_REFERENCE"
        )
        self.node_id = node_id
        self.parent_id = parent_id


class NodeMaxTagsExceededError(NodeDomainException):
    """Exception raised when node exceeds maximum tags"""
    
    def __init__(self, max_tags: int):
        super().__init__(
            f"Node cannot have more than {max_tags} tags",
            "MAX_TAGS_EXCEEDED"
        )
        self.max_tags = max_tags


class NodeValidationError(NodeDomainException):
    """Exception raised when node validation fails"""
    
    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error for field '{field}': {message}", "NODE_VALIDATION_ERROR")
        self.field = field


class NodeConcurrencyError(NodeDomainException):
    """Exception raised when node concurrency conflict occurs"""
    
    def __init__(self, node_id: str, expected_version: int, actual_version: int):
        super().__init__(
            f"Concurrency conflict for node {node_id}. Expected version {expected_version}, got {actual_version}",
            "NODE_CONCURRENCY_CONFLICT"
        )
        self.node_id = node_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class NodeContentTooLargeError(NodeDomainException):
    """Exception raised when node content is too large"""
    
    def __init__(self, max_size: int):
        super().__init__(
            f"Node content cannot exceed {max_size} characters",
            "NODE_CONTENT_TOO_LARGE"
        )
        self.max_size = max_size


class NodeTagInvalidError(NodeDomainException):
    """Exception raised when node tag is invalid"""
    
    def __init__(self, tag: str, message: str):
        super().__init__(f"Invalid tag '{tag}': {message}", "INVALID_NODE_TAG")
        self.tag = tag




