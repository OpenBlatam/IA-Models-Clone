"""
Service Decorators
Decorators for common service patterns (error handling, validation, etc.)
"""

import logging
from functools import wraps
from typing import Callable, TypeVar, Any, Awaitable

from .exceptions import (
    AgentNotFoundError,
    TaskNotFoundError,
    AgentServiceError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_service_errors(
    operation_name: str,
    re_raise_domain_exceptions: bool = True
):
    """
    Decorator for consistent service error handling
    
    Args:
        operation_name: Name of the operation for error messages
        re_raise_domain_exceptions: Whether to re-raise domain exceptions (AgentNotFoundError, etc.)
    
    Usage:
        @handle_service_errors("create agent")
        async def create_agent(...):
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except (AgentNotFoundError, TaskNotFoundError) as e:
                if re_raise_domain_exceptions:
                    raise
                logger.warning(f"Domain exception in {operation_name}: {e}")
                raise
            except Exception as e:
                logger.error(
                    f"Error in {operation_name}",
                    exc_info=True,
                    extra={"operation": operation_name, **kwargs}
                )
                raise AgentServiceError(
                    f"Failed to {operation_name}: {str(e)}",
                    operation=operation_name,
                    **kwargs
                )
        return wrapper
    return decorator


def validate_agent_exists(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to validate agent exists before operation
    
    Usage:
        @validate_agent_exists
        async def stop_agent(self, agent_id: str):
            ...
    """
    @wraps(func)
    async def wrapper(self, agent_id: str, *args, **kwargs) -> T:
        # Assume self has registry attribute
        agent = await self.registry.get(agent_id)
        if not agent:
            raise AgentNotFoundError(agent_id)
        return await func(self, agent_id, *args, **kwargs)
    return wrapper


def log_operation(operation_name: str):
    """
    Decorator for logging operations
    
    Args:
        operation_name: Name of the operation to log
    
    Usage:
        @log_operation("stop agent")
        async def stop_agent(...):
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            logger.info(f"Starting {operation_name}")
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Completed {operation_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation_name}: {e}")
                raise
        return wrapper
    return decorator

