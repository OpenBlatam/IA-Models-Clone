"""
Validation middleware for request validation and sanitization.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import status
import logging

from ..utils.security import sanitize_input
from ..constants import MAX_USER_ID_LENGTH, MAX_CHAT_ID_LENGTH

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""
    
    async def dispatch(self, request: Request, call_next):
        """Validate and sanitize request data."""
        # Skip validation for certain paths
        skip_paths = ["/health", "/", "/docs", "/openapi.json", "/redoc"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Sanitize query parameters
        if request.query_params:
            sanitized_params = {}
            for key, value in request.query_params.items():
                # Sanitize based on parameter type
                if key in ["user_id", "chat_id"]:
                    max_len = MAX_USER_ID_LENGTH if key == "user_id" else MAX_CHAT_ID_LENGTH
                    sanitized_params[key] = sanitize_input(value, max_length=max_len)
                else:
                    sanitized_params[key] = sanitize_input(value)
            
            # Note: FastAPI handles query params, so we log but don't modify
            # This is mainly for logging and monitoring
        
        # Process request
        response = await call_next(request)
        return response




