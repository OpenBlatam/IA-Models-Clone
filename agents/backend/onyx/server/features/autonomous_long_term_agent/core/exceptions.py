"""
Custom Exceptions for Autonomous Long-Term Agent
Provides specific exception types with context for better error handling
"""

from typing import Optional


class AgentError(Exception):
    """Base exception for agent-related errors"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        """
        Initialize agent error
        
        Args:
            message: Error message
            agent_id: Optional agent ID for context
            **kwargs: Additional context fields
        """
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.context = kwargs
    
    def __str__(self) -> str:
        """String representation with context"""
        if self.agent_id:
            return f"[Agent: {self.agent_id}] {self.message}"
        return self.message


class AgentNotFoundError(AgentError):
    """Raised when an agent with the given ID is not found"""
    
    def __init__(self, agent_id: str):
        super().__init__(f"Agent '{agent_id}' not found", agent_id=agent_id)


class AgentAlreadyRunningError(AgentError):
    """Raised when attempting to start an agent that is already running"""
    
    def __init__(self, agent_id: str):
        super().__init__(f"Agent '{agent_id}' is already running", agent_id=agent_id)


class AgentNotRunningError(AgentError):
    """Raised when attempting to perform an operation on a non-running agent"""
    
    def __init__(self, agent_id: str, operation: Optional[str] = None):
        message = f"Agent '{agent_id}' is not running"
        if operation:
            message += f" (cannot {operation})"
        super().__init__(message, agent_id=agent_id, operation=operation)


class TaskNotFoundError(AgentError):
    """Raised when a task with the given ID is not found"""
    
    def __init__(self, task_id: str, agent_id: Optional[str] = None):
        message = f"Task '{task_id}' not found"
        if agent_id:
            message += f" for agent '{agent_id}'"
        super().__init__(message, agent_id=agent_id, task_id=task_id)


class RateLimitExceededError(AgentError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, key: Optional[str] = None, remaining: int = 0, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if key:
            message += f" for '{key}'"
        if remaining >= 0:
            message += f". Remaining requests: {remaining}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, key=key, remaining=remaining, retry_after=retry_after)


class InvalidAgentStateError(AgentError):
    """Raised when agent is in an invalid state for the requested operation"""
    
    def __init__(self, agent_id: str, current_state: str, required_state: Optional[str] = None):
        message = f"Agent '{agent_id}' is in invalid state '{current_state}'"
        if required_state:
            message += f" (required: '{required_state}')"
        super().__init__(message, agent_id=agent_id, current_state=current_state, required_state=required_state)


class AgentServiceError(AgentError):
    """Raised when an agent service operation fails"""
    
    def __init__(self, message: str, operation: Optional[str] = None, agent_id: Optional[str] = None, **kwargs):
        if operation:
            message = f"{operation} failed: {message}"
        super().__init__(message, agent_id=agent_id, operation=operation, **kwargs)
