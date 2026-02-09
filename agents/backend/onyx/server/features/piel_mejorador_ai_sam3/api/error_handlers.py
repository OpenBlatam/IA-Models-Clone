"""
Error handling utilities for Piel Mejorador AI SAM3 API.
"""

import logging
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def require_agent(func: Callable) -> Callable:
    """
    Decorator to ensure agent is initialized before endpoint execution.
    
    Returns:
        Decorated function with agent validation
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        from .piel_mejorador_api import _agent
        
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        return await func(*args, **kwargs)
    return wrapper


def handle_task_errors(func: Callable) -> Callable:
    """
    Decorator to handle task-related errors consistently.
    
    Returns:
        Decorated function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return wrapper




