"""
Error handling middleware for centralized exception handling.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from ..api.exceptions import MaintenanceAPIException
from ..utils.metrics import metrics_collector

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle exceptions and return standardized error responses.
    """
    
    def _record_error_metric(self, request: Request):
        """Record error in metrics collector."""
        metrics_collector.record_request(
            request.url.path,
            0,
            success=False
        )
    
    def _create_error_response(
        self,
        status_code: int,
        error: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            status_code: HTTP status code
            error: Error message
            error_code: Error code
            details: Optional error details
            headers: Optional response headers
        
        Returns:
            JSONResponse with error
        """
        content = {
            "success": False,
            "error": error,
            "error_code": error_code
        }
        if details:
            content["details"] = details
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers or {}
        )
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except MaintenanceAPIException as e:
            # Handle custom API exceptions
            logger.warning(f"API exception: {e.detail}", extra={"error_code": e.error_code})
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=e.status_code,
                error=e.detail,
                error_code=e.error_code,
                headers=e.headers
            )
        except RequestValidationError as e:
            # Handle Pydantic validation errors
            logger.warning(f"Validation error: {e.errors()}")
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error="Validation error",
                error_code="VALIDATION_ERROR",
                details=e.errors()
            )
        except StarletteHTTPException as e:
            # Handle FastAPI HTTP exceptions
            logger.warning(f"HTTP exception: {e.detail}")
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=e.status_code,
                error=e.detail,
                error_code="HTTP_ERROR",
                headers=e.headers
            )
        except ValueError as e:
            # Handle ValueError (often from validation)
            logger.warning(f"Value error: {str(e)}")
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error=str(e),
                error_code="VALIDATION_ERROR"
            )
        except TimeoutError as e:
            # Handle timeout errors
            logger.error(f"Timeout error: {str(e)}")
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                error="Request timeout",
                error_code="TIMEOUT"
            )
        except ConnectionError as e:
            # Handle connection errors
            logger.error(f"Connection error: {str(e)}")
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error="Service temporarily unavailable",
                error_code="SERVICE_UNAVAILABLE"
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            self._record_error_metric(request)
            return self._create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )

