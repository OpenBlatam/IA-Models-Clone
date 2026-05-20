"""
Error Handler Middleware
Centralized error handling for API endpoints
"""

import functools
import logging
from typing import Callable, Dict, Type
from fastapi import HTTPException

from ....core.exceptions import (
    AgentError,
    AgentNotFoundError,
    AgentAlreadyRunningError,
    AgentNotRunningError,
    TaskNotFoundError,
    RateLimitExceededError,
    InvalidAgentStateError,
    AgentServiceError
)

logger = logging.getLogger(__name__)

# Exception to HTTP status code mapping
_EXCEPTION_STATUS_MAP: Dict[Type[AgentError], int] = {
    AgentNotFoundError: 404,
    TaskNotFoundError: 404,
    AgentAlreadyRunningError: 409,
    AgentNotRunningError: 400,
    InvalidAgentStateError: 400,
    RateLimitExceededError: 429,
    AgentServiceError: 500,
}


def handle_agent_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle agent-specific exceptions
    
    Converts custom exceptions to appropriate HTTP responses using a mapping
    for better maintainability and DRY principle.
    
    Usage:
        @handle_agent_exceptions
        async def my_endpoint(...):
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AgentError as e:
            # Get status code from mapping, default to 500
            status_code = _EXCEPTION_STATUS_MAP.get(type(e), 500)
            
            # Add rate limit headers if applicable
            headers = {}
            if isinstance(e, RateLimitExceededError) and e.remaining is not None:
                headers["X-RateLimit-Remaining"] = str(e.remaining)
            
            # Log service errors with full context
            if isinstance(e, AgentServiceError):
                logger.error(f"Agent service error: {e}", exc_info=True, extra={
                    "agent_id": e.agent_id,
                    "context": e.context
                })
            
            raise HTTPException(status_code=status_code, detail=str(e), headers=headers)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")
    
    return wrapper

