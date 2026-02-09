"""
API Improvements - Microservices Patterns
Enhanced API with better organization and patterns
"""

from typing import Optional, Dict, Any
from fastapi import Request
import logging

logger = logging.getLogger(__name__)

# Import patterns from pdf_variantes if available
try:
    import sys
    import os
    
    pdf_variantes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "pdf_variantes"
    )
    
    if os.path.exists(pdf_variantes_path):
        sys.path.insert(0, os.path.abspath(pdf_variantes_path))
        
        from utils.response_helpers import (
            create_success_response,
            create_error_response,
            get_request_id as _get_request_id
        )
        from utils.structured_logging import (
            set_request_context,
            get_logger as get_structured_logger
        )
        from api.exceptions import (
            ValidationError,
            NotFoundError,
            ServiceUnavailableError
        )
        IMPROVEMENTS_AVAILABLE = True
    else:
        IMPROVEMENTS_AVAILABLE = False
except Exception as e:
    IMPROVEMENTS_AVAILABLE = False
    logger.warning(f"API improvements not available: {e}")


def enhance_request_context(request: Request):
    """Enhance request with context for logging and tracing"""
    if not IMPROVEMENTS_AVAILABLE:
        return
    
    try:
        request_id = _get_request_id(request)
        user_id = request.headers.get("X-User-Id") or request.headers.get("User-Id")
        correlation_id = request.headers.get("X-Correlation-ID")
        
        set_request_context(
            request_id=request_id,
            user_id=user_id,
            correlation_id=correlation_id
        )
    except Exception as e:
        logger.warning(f"Failed to enhance request context: {e}")


def create_enhanced_response(
    data: Any = None,
    message: Optional[str] = None,
    request: Optional[Request] = None,
    status_code: int = 200
):
    """Create enhanced response with request ID"""
    if not IMPROVEMENTS_AVAILABLE:
        # Fallback to simple dict
        return {
            "success": True,
            "data": data,
            "message": message or "Success"
        }
    
    request_id = None
    if request:
        request_id = _get_request_id(request)
    
    return create_success_response(
        data=data,
        message=message,
        request_id=request_id,
        status_code=status_code
    )


def create_enhanced_error(
    message: str,
    status_code: int = 500,
    error_type: Optional[str] = None,
    request: Optional[Request] = None
):
    """Create enhanced error response"""
    if not IMPROVEMENTS_AVAILABLE:
        return {
            "success": False,
            "error": {
                "message": message,
                "code": status_code
            }
        }
    
    request_id = None
    if request:
        request_id = _get_request_id(request)
    
    return create_error_response(
        message=message,
        status_code=status_code,
        error_type=error_type,
        request_id=request_id
    )






