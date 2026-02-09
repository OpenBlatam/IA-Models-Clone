"""
Error handlers for consistent error responses
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_error_response(
    message: str,
    status_code: int = 500,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create a standardized error response"""
    error_data = {
        "success": False,
        "error": message
    }
    
    if error_code:
        error_data["error_code"] = error_code
    
    if details:
        error_data["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_data
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return create_error_response(
        message="Internal server error",
        status_code=500,
        error_code="INTERNAL_ERROR",
        details={"path": str(request.url.path)}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP exception handler"""
    return create_error_response(
        message=exc.detail,
        status_code=exc.status_code,
        error_code=f"HTTP_{exc.status_code}"
    )

