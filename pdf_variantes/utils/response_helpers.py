"""
PDF Variantes - Response Helpers
Consistent response formatting for API endpoints
"""

from typing import Optional, Dict, Any
from uuid import uuid4
from fastapi import Request


def get_request_id(request: Request) -> str:
    """Extract or generate request ID from request"""
    # Check if already set in middleware
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        # Check header
        request_id = request.headers.get("X-Request-ID")
    if not request_id:
        # Generate new one
        request_id = str(uuid4())
        request.state.request_id = request_id
    return request_id


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = 200,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized success response"""
    response = {
        "success": True,
        "data": data,
        "status_code": status_code
    }
    
    if message:
        response["message"] = message
    
    if request_id:
        response["request_id"] = request_id
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "success": False,
        "data": None,
        "error": {
            "message": message,
            "status_code": status_code
        },
        "status_code": status_code
    }
    
    if error_type:
        response["error"]["type"] = error_type
    
    if error_code:
        response["error"]["code"] = error_code
    elif error_type:
        response["error"]["code"] = error_type.upper()
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["request_id"] = request_id
        response["error"]["request_id"] = request_id
    
    if metadata:
        response["error"]["metadata"] = metadata
    
    return response


def create_not_found_response(
    resource: str,
    resource_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized not found response"""
    message = f"{resource} not found"
    if resource_id:
        message = f"{resource} with id '{resource_id}' not found"
    
    return create_error_response(
        message=message,
        status_code=404,
        error_type="NotFoundError",
        error_code="NOT_FOUND",
        request_id=request_id,
        metadata={
            "resource": resource,
            "resource_id": resource_id
        } if resource_id else {"resource": resource}
    )


def create_unauthorized_response(
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized unauthorized response"""
    return create_error_response(
        message=message or "Authentication required",
        status_code=401,
        error_type="UnauthorizedError",
        error_code="UNAUTHORIZED",
        request_id=request_id,
        metadata={"reason": reason} if reason else None
    )


def create_forbidden_response(
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    resource: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized forbidden response"""
    return create_error_response(
        message=message or "Access denied",
        status_code=403,
        error_type="ForbiddenError",
        error_code="FORBIDDEN",
        request_id=request_id,
        metadata={"resource": resource} if resource else None
    )


def create_validation_error_response(
    message: str,
    errors: Optional[list] = None,
    field: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized validation error response"""
    metadata = {}
    if field:
        metadata["field"] = field
    if errors:
        metadata["errors"] = errors
    
    return create_error_response(
        message=message,
        status_code=422,
        error_type="ValidationError",
        error_code="VALIDATION_ERROR",
        request_id=request_id,
        metadata=metadata if metadata else None
    )


def create_rate_limit_response(
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized rate limit response"""
    metadata = {}
    if retry_after:
        metadata["retry_after"] = retry_after
    
    return create_error_response(
        message="Rate limit exceeded",
        status_code=429,
        error_type="RateLimitError",
        error_code="RATE_LIMIT_EXCEEDED",
        request_id=request_id,
        metadata=metadata if metadata else None
    )
