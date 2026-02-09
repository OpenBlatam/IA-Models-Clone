"""
Error Handler Middleware

Centralized error handling for API v1.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from ....application.exceptions import (
    TrackNotFoundException,
    AnalysisException,
    RecommendationException,
    UseCaseException
)

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler middleware for API v1.
    
    Catches exceptions and returns appropriate HTTP responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except TrackNotFoundException as e:
        logger.warning(f"Track not found: {e}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": "Track not found",
                "detail": str(e),
                "code": "TRACK_NOT_FOUND"
            }
        )
    except AnalysisException as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Analysis failed",
                "detail": str(e),
                "code": "ANALYSIS_ERROR"
            }
        )
    except RecommendationException as e:
        logger.error(f"Recommendation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Recommendation failed",
                "detail": str(e),
                "code": "RECOMMENDATION_ERROR"
            }
        )
    except UseCaseException as e:
        logger.warning(f"Use case error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Invalid request",
                "detail": str(e),
                "code": "USE_CASE_ERROR"
            }
        )
    except RequestValidationError as e:
        logger.warning(f"Validation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation error",
                "detail": str(e),
                "code": "VALIDATION_ERROR"
            }
        )
    except StarletteHTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "error": e.detail,
                "code": f"HTTP_{e.status_code}"
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "code": "INTERNAL_ERROR"
            }
        )

