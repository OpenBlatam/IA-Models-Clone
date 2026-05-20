"""
Validation Exception Handler
===========================

Manejador de errores de validación.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )
    
    logger.warning(
        "Validation error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "errors": errors,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "request_id": request_id,
            "details": errors,
        },
    )

