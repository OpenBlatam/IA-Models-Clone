"""
PDF Variantes API - Enhanced Validation Middleware
Request validation and transformation
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Optional

from ..utils.real_world import ErrorCode, format_error_response
from ..utils.validation import Validator
from ..utils.response_helpers import get_request_id


class ValidationMiddleware(BaseHTTPMiddleware):
    """Enhanced validation middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Validate request before processing"""
        
        # Validate request ID if present
        if request.headers.get("X-Request-ID"):
            request_id = request.headers.get("X-Request-ID")
            # Could validate format here if needed
            pass
        
        # Validate API version if present
        api_version = request.headers.get("API-Version")
        if api_version:
            valid_versions = ["v1", "v2"]
            if api_version.lower() not in valid_versions:
                request_id = get_request_id(request)
                error_response = format_error_response(
                    ErrorCode.VALIDATION_ERROR,
                    f"Invalid API version. Supported versions: {', '.join(valid_versions)}",
                    {"provided_version": api_version, "supported_versions": valid_versions},
                    request_id
                )
                raise HTTPException(status_code=400, detail=error_response)
        
        response = await call_next(request)
        return response


def validate_document_access(
    document_id: str,
    user_id: str,
    request: Request
) -> tuple[bool, Optional[str]]:
    """
    Validate user has access to document
    
    Returns:
        (has_access, error_message)
    """
    request_id = get_request_id(request)
    
    # Validate document ID format
    is_valid, error = Validator.validate_document_id(document_id)
    if not is_valid:
        return False, error
    
    # Validate user ID
    is_valid, error = Validator.validate_user_id(user_id)
    if not is_valid:
        return False, error
    
    # Additional access checks would go here
    # (check permissions, ownership, etc.)
    
    return True, None

