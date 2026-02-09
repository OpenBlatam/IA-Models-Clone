"""
Base router class for common API functionality.
Reduces code duplication across multiple API routers.
"""

from fastapi import APIRouter, Depends, Request
from typing import Optional, Dict, Any, Callable, TypeVar
from functools import wraps
import logging
import time

from .responses import success_response, paginated_response
from .dependencies import check_rate_limit
from ..core.database import MaintenanceDatabase
from ..core.conversation_manager import ConversationManager
from ..core.maintenance_tutor import RobotMaintenanceTutor
from ..api.auth_api import require_auth

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRouter:
    """
    Base router class with common functionality for all API routers.
    Provides standardized responses, error handling, and common dependencies.
    """
    
    def __init__(
        self,
        prefix: str,
        tags: list,
        require_authentication: bool = True,
        require_rate_limit: bool = False
    ):
        """
        Initialize base router.
        
        Args:
            prefix: API prefix for the router
            tags: OpenAPI tags for documentation
            require_authentication: Whether endpoints require authentication by default
            require_rate_limit: Whether endpoints require rate limiting by default
        """
        self.prefix = prefix
        self.tags = tags
        self.require_authentication = require_authentication
        self.require_rate_limit = require_rate_limit
        self.router = APIRouter(prefix=prefix, tags=tags)
        
        # Lazy-loaded dependencies
        self._database: Optional[MaintenanceDatabase] = None
        self._conversation_manager: Optional[ConversationManager] = None
    
    @property
    def database(self) -> MaintenanceDatabase:
        """Get database instance (lazy-loaded)."""
        if self._database is None:
            self._database = MaintenanceDatabase()
        return self._database
    
    @property
    def conversation_manager(self) -> ConversationManager:
        """Get conversation manager instance (lazy-loaded)."""
        if self._conversation_manager is None:
            self._conversation_manager = ConversationManager()
        return self._conversation_manager
    
    def get_auth_dependency(self) -> Callable:
        """
        Get authentication dependency.
        
        Returns:
            Dependency function for authentication
        """
        if self.require_authentication:
            return require_auth
        # Return a no-op dependency that doesn't require auth
        def no_auth():
            return None
        return no_auth
    
    def get_rate_limit_dependency(self) -> Optional[Callable]:
        """
        Get rate limit dependency.
        
        Returns:
            Dependency function for rate limiting or None
        """
        if self.require_rate_limit:
            return check_rate_limit
        return None
    
    def success(self, data: Any = None, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a success response.
        
        Args:
            data: Response data
            message: Optional success message
        
        Returns:
            Standardized success response
        """
        return success_response(data, message)
    
    def paginated(
        self,
        items: list,
        total: int,
        page: int = 1,
        page_size: int = 10,
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a paginated response.
        
        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number
            page_size: Number of items per page
            message: Optional message
        
        Returns:
            Standardized paginated response
        """
        return paginated_response(items, total, page, page_size, message)
    
    def error(
        self,
        error: str,
        error_code: Optional[str] = None,
        status_code: int = 400
    ) -> Dict[str, Any]:
        """
        Create an error response.
        
        Args:
            error: Error message
            error_code: Optional error code
            status_code: HTTP status code
        
        Returns:
            Standardized error response dictionary
        """
        return {
            "success": False,
            "error": error,
            "error_code": error_code
        }
    
    def timed_endpoint(self, endpoint_name: str):
        """
        Decorator to time endpoint execution and log metrics.
        
        Args:
            endpoint_name: Name of the endpoint for logging
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.debug(f"{endpoint_name} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"{endpoint_name} failed after {duration:.3f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    def log_request(self, endpoint: str, **kwargs):
        """
        Log API request.
        
        Args:
            endpoint: Endpoint name
            **kwargs: Additional log context
        """
        logger.info(f"API request: {endpoint}", extra=kwargs)
    
    def log_error(self, endpoint: str, error: Exception, **kwargs):
        """
        Log API error.
        
        Args:
            endpoint: Endpoint name
            error: Exception that occurred
            **kwargs: Additional log context
        """
        logger.error(f"API error in {endpoint}: {error}", exc_info=True, extra=kwargs)


def create_base_router(
    prefix: str,
    tags: list,
    require_authentication: bool = True,
    require_rate_limit: bool = False
) -> BaseRouter:
    """
    Factory function to create a base router.
    
    Args:
        prefix: API prefix
        tags: OpenAPI tags
        require_authentication: Whether to require authentication
        require_rate_limit: Whether to require rate limiting
    
    Returns:
        BaseRouter instance
    """
    return BaseRouter(prefix, tags, require_authentication, require_rate_limit)
