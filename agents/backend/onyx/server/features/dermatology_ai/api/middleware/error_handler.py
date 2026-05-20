"""
Common error handling utilities for controllers
Extracted to reduce code duplication
"""

from fastapi import HTTPException
from typing import Callable, Any
import logging

from ...core.application.exceptions import ValidationError, ProcessingError, NotFoundError

logger = logging.getLogger(__name__)


async def handle_controller_errors(
    operation: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Common error handler for controller operations
    
    Args:
        operation: Async function to execute
        *args: Positional arguments for the operation
        **kwargs: Keyword arguments for the operation
    
    Returns:
        Result of the operation
    
    Raises:
        HTTPException: With appropriate status code and message
    """
    try:
        return await operation(*args, **kwargs)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundError as e:
        logger.warning(f"Not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

