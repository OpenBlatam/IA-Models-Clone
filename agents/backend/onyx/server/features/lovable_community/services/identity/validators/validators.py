"""
Identity Validators

Validation logic for user identity and authentication.
"""

from typing import Optional
from fastapi import HTTPException, status

from ....exceptions import InvalidChatError
from ....helpers import sanitize_text
from ....utils.logging_config import StructuredLogger

logger = StructuredLogger(__name__)


class IdentityValidator:
    """
    Validator for identity operations.
    
    Handles validation and sanitization of user IDs and authentication data.
    """
    
    def validate_user_id(self, user_id: str, raise_on_invalid: bool = True) -> Optional[str]:
        """
        Validate and sanitize a user ID.
        
        Args:
            user_id: User ID to validate
            raise_on_invalid: Whether to raise exception or return None
            
        Returns:
            Sanitized user ID or None
            
        Raises:
            InvalidChatError: If the ID is invalid and raise_on_invalid=True
        """
        if not user_id:
            if raise_on_invalid:
                raise InvalidChatError("User ID cannot be empty")
            return None
        
        try:
            user_id = sanitize_text(user_id)
            if not user_id or not user_id.strip():
                if raise_on_invalid:
                    raise InvalidChatError("User ID cannot be empty after sanitization")
                return None
            
            user_id = user_id.strip()
            
            # Basic validation: non-empty, reasonable length
            if len(user_id) > 255:
                if raise_on_invalid:
                    raise InvalidChatError("User ID is too long (max 255 characters)")
                return None
            
            return user_id
            
        except ValueError as e:
            if raise_on_invalid:
                raise InvalidChatError(str(e)) from e
            return None
        except Exception as e:
            logger.warning(
                "Unexpected error validating user_id",
                error=str(e),
                error_type=type(e).__name__
            )
            if raise_on_invalid:
                raise InvalidChatError(f"Validation error: {str(e)}") from e
            return None

