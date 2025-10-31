"""
Error Codes - Standardized error codes for API responses
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes for API responses"""
    
    # Validation Errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_TYPE = "INVALID_TYPE"
    EMPTY_VALUE = "EMPTY_VALUE"
    CONTENT_EMPTY = "CONTENT_EMPTY"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"
    SIZE_EXCEEDED = "SIZE_EXCEEDED"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    INVALID_FORMAT = "INVALID_FORMAT"
    NOT_POSITIVE = "NOT_POSITIVE"
    EMPTY_LIST = "EMPTY_LIST"
    
    # Authentication & Authorization (401, 403)
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Not Found (404)
    NOT_FOUND = "NOT_FOUND"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    
    # Rate Limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    
    # Server Errors (500, 503)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    
    # Webhook Errors
    WEBHOOK_DELIVERY_FAILED = "WEBHOOK_DELIVERY_FAILED"
    WEBHOOK_ENDPOINT_INVALID = "WEBHOOK_ENDPOINT_INVALID"
    
    # AI/ML Errors
    AI_SERVICE_ERROR = "AI_SERVICE_ERROR"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"


def format_error_response(
    code: ErrorCode,
    message: str,
    details: dict = None,
    request_id: str = None
) -> dict:
    """
    Format standardized error response
    
    Args:
        code: Error code
        message: Human-readable error message
        details: Additional error details
        request_id: Request ID for tracing
        
    Returns:
        Formatted error response dictionary
    """
    import time
    
    error_response = {
        "success": False,
        "data": None,
        "error": {
            "code": code.value,
            "message": message,
            "type": code.name,
        },
        "timestamp": time.time()
    }
    
    if details:
        error_response["error"]["details"] = details
    
    if request_id:
        error_response["request_id"] = request_id
    
    return error_response


def get_status_code_for_error(code: ErrorCode) -> int:
    """Get HTTP status code for error code"""
    status_map = {
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.INVALID_TYPE: 400,
        ErrorCode.EMPTY_VALUE: 400,
        ErrorCode.CONTENT_EMPTY: 400,
        ErrorCode.CONTENT_TOO_SHORT: 400,
        ErrorCode.CONTENT_TOO_LONG: 400,
        ErrorCode.SIZE_EXCEEDED: 400,
        ErrorCode.OUT_OF_RANGE: 400,
        ErrorCode.INVALID_FORMAT: 400,
        ErrorCode.NOT_POSITIVE: 400,
        ErrorCode.EMPTY_LIST: 400,
        
        ErrorCode.UNAUTHORIZED: 401,
        ErrorCode.INVALID_TOKEN: 401,
        ErrorCode.TOKEN_EXPIRED: 401,
        ErrorCode.FORBIDDEN: 403,
        
        ErrorCode.NOT_FOUND: 404,
        ErrorCode.RESOURCE_NOT_FOUND: 404,
        ErrorCode.ENDPOINT_NOT_FOUND: 404,
        
        ErrorCode.RATE_LIMIT_EXCEEDED: 429,
        ErrorCode.TOO_MANY_REQUESTS: 429,
        
        ErrorCode.SERVICE_UNAVAILABLE: 503,
        ErrorCode.EXTERNAL_SERVICE_ERROR: 503,
        ErrorCode.AI_SERVICE_ERROR: 503,
        ErrorCode.MODEL_NOT_AVAILABLE: 503,
        ErrorCode.PROCESSING_TIMEOUT: 504,
        
        ErrorCode.INTERNAL_ERROR: 500,
        ErrorCode.PROCESSING_ERROR: 500,
        ErrorCode.DATABASE_ERROR: 500,
        ErrorCode.WEBHOOK_DELIVERY_FAILED: 500,
        ErrorCode.WEBHOOK_ENDPOINT_INVALID: 400,
    }
    
    return status_map.get(code, 500)
