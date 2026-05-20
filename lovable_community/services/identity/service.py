"""
Identity Service

Service for handling identity and authentication operations.
Extracts and validates user identity from requests.
"""

from typing import Optional
from fastapi import Request, HTTPException, status

from ....config import settings
from ....exceptions import InvalidChatError
from ....utils.logging_config import StructuredLogger
from .validators import IdentityValidator

logger = StructuredLogger(__name__)


class IdentityService:
    """
    Service for identity and authentication operations.
    
    Handles extraction and validation of user identity from requests.
    Follows the same patterns as other services in the codebase.
    """
    
    def __init__(self, validator: Optional[IdentityValidator] = None):
        """
        Initialize IdentityService.
        
        Args:
            validator: IdentityValidator instance (created if not provided)
        """
        self.validator = validator or IdentityValidator()
    
    def get_user_id(self, request: Request) -> str:
        """
        Extract and validate user_id from request.
        
        Attempts to obtain user_id in the following order:
        1. Header `X-User-ID` (preferred)
        2. Query parameter `user_id`
        3. In debug mode: default value
        4. In production: raises HTTPException 401
        
        Args:
            request: FastAPI Request object
            
        Returns:
            str: Validated user ID
            
        Raises:
            HTTPException: If user_id cannot be obtained and in production mode
            InvalidChatError: If user_id is invalid
        """
        # Try header first (preferred method)
        user_id: Optional[str] = request.headers.get("X-User-ID")
        if user_id and user_id.strip():
            return self.validator.validate_user_id(user_id.strip())
        
        # Try query parameter
        user_id = request.query_params.get("user_id")
        if user_id and user_id.strip():
            return self.validator.validate_user_id(user_id.strip())
        
        # Production mode: require authentication
        if not settings.debug:
            logger.warning(
                "No user_id found in request",
                path=request.url.path,
                method=request.method,
                headers=dict(request.headers)
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide X-User-ID header or user_id query parameter."
            )
        
        # Debug mode: use default
        logger.warning(
            "No user_id found in request, using default (debug mode)",
            path=request.url.path,
            method=request.method
        )
        return "user_123"
    
    def get_optional_user_id(self, request: Request) -> Optional[str]:
        """
        Extract and validate optional user_id from request.
        
        Similar to `get_user_id()` but never raises exceptions.
        Returns None if user_id cannot be obtained.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            Optional[str]: Validated user ID or None if not available
        """
        try:
            return self.get_user_id(request)
        except HTTPException:
            # In production, if no user_id, return None instead of raising
            return None
        except InvalidChatError:
            # Invalid user_id format, return None
            logger.warning(
                "Invalid user_id format in request",
                path=request.url.path,
                method=request.method
            )
            return None
        except Exception as e:
            # Log unexpected errors but don't fail the request
            logger.warning(
                "Unexpected error getting optional user_id",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path
            )
            return None
    
    def validate_user_id(self, user_id: str) -> str:
        """
        Validate a user ID string.
        
        Args:
            user_id: User ID to validate
            
        Returns:
            str: Validated and sanitized user ID
            
        Raises:
            InvalidChatError: If user_id is invalid
        """
        return self.validator.validate_user_id(user_id)
    
    def is_authenticated(self, request: Request) -> bool:
        """
        Check if request has valid authentication.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            bool: True if request has valid user_id, False otherwise
        """
        try:
            user_id = self.get_optional_user_id(request)
            return user_id is not None and user_id.strip() != ""
        except Exception:
            return False

