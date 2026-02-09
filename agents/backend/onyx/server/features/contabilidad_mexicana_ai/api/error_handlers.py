"""
Error handling utilities for Contador API.

Refactored to consolidate error handling patterns into reusable decorators.
"""

import logging
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..core.validators import ValidationError

logger = logging.getLogger(__name__)


def handle_contador_errors(service_name: str):
    """
    Decorator to handle errors consistently across Contador API endpoints.
    
    Args:
        service_name: Name of the service for logging
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> JSONResponse:
            try:
                result = await func(*args, **kwargs)
                return JSONResponse(content=result)
            except ValidationError as e:
                logger.warning(f"Validation error in {service_name}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error in {service_name}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        return wrapper
    return decorator

