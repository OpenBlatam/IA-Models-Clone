"""
Context middleware for Multi-Model API
Automatically manages request context
"""

import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .context import create_request_context, clear_request_context, get_request_context
from .utils import generate_request_id

logger = logging.getLogger(__name__)


class ContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically create and manage request context
    
    Automatically:
    - Creates request context with unique ID
    - Extracts user ID and API key from headers
    - Tracks request timing
    - Cleans up context after request
    """
    
    def __init__(
        self,
        app,
        extract_user_id: Callable[[Request], str | None] = None,
        extract_api_key: Callable[[Request], str | None] = None,
        extract_metadata: Callable[[Request], dict] = None
    ):
        """
        Initialize context middleware
        
        Args:
            app: FastAPI application
            extract_user_id: Optional function to extract user ID from request
            extract_api_key: Optional function to extract API key from request
            extract_metadata: Optional function to extract metadata from request
        """
        super().__init__(app)
        self.extract_user_id = extract_user_id or self._default_extract_user_id
        self.extract_api_key = extract_api_key or self._default_extract_api_key
        self.extract_metadata = extract_metadata or self._default_extract_metadata
    
    @staticmethod
    def _default_extract_user_id(request: Request) -> str | None:
        """Default user ID extraction from headers"""
        return request.headers.get("X-User-ID") or request.headers.get("X-User-Id")
    
    @staticmethod
    def _default_extract_api_key(request: Request) -> str | None:
        """Default API key extraction from headers"""
        return request.headers.get("X-API-Key") or request.headers.get("Authorization")
    
    @staticmethod
    def _default_extract_metadata(request: Request) -> dict:
        """Default metadata extraction from request"""
        return {
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with context management
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Extract context information
        user_id = self.extract_user_id(request)
        api_key = self.extract_api_key(request)
        metadata = self.extract_metadata(request)
        
        # Add request ID to metadata
        request_id = generate_request_id()
        metadata["request_id"] = request_id
        
        # Create context
        context = create_request_context(
            request_id=request_id,
            user_id=user_id,
            api_key=api_key,
            metadata=metadata
        )
        
        # Add request ID to response headers
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            # Add timing information
            elapsed_ms = context.elapsed_ms
            response.headers["X-Request-Duration-MS"] = str(int(elapsed_ms))
            
            logger.debug(
                f"Request {request_id} completed in {elapsed_ms:.2f}ms",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms
                }
            )
            
            return response
        except Exception as e:
            logger.error(
                f"Request {request_id} failed: {e}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
        finally:
            # Always clear context
            clear_request_context()




