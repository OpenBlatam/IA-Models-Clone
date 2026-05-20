"""
Response Helpers for Contador API
=================================

Helper utilities for API endpoints.
Centralizes error handling and response formatting.

Single Responsibility: Provide reusable utilities for API endpoints.
"""

import logging
from typing import Dict, Any, Callable, Awaitable
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..core.validators import ValidationError

logger = logging.getLogger(__name__)


async def handle_service_call(
    service_method: Callable[..., Awaitable[Dict[str, Any]]],
    service_name: str,
    *args,
    **kwargs
) -> JSONResponse:
    """
    Handle a service method call with consistent error handling.
    
    Args:
        service_method: Async service method to call
        service_name: Name of the service (for error messages)
        *args: Positional arguments for the service method
        **kwargs: Keyword arguments for the service method
        
    Returns:
        JSONResponse with the result
        
    Raises:
        HTTPException: On validation or server errors
    """
    try:
        result = await service_method(*args, **kwargs)
        return JSONResponse(content=result)
    except ValidationError as e:
        logger.warning(f"Validation error in {service_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in {service_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

