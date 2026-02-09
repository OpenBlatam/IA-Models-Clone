from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from typing import Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from ..utils.response import create_error_response
from typing import Any, List, Dict, Optional
import asyncio
"""
Error Handling Middleware - Clean error responses
"""



logger = logging.getLogger(__name__)


def create_error_handler() -> Callable:
    """Create centralized error handler."""
    
    async def error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=create_error_response(
                    message=exc.detail,
                    status_code=exc.status_code,
                ).model_dump(),
            )
        
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                message="Internal server error",
                status_code=500,
            ).model_dump(),
        )
    
    return error_handler 