"""
Helpers for handling service results and validations
"""

from typing import Any, Optional, Dict
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def require_success(
    result: Any,
    error_message: str,
    status_code: int = 400
) -> None:
    """
    Require that a service result indicates success
    
    Args:
        result: Service result (bool, dict with 'error', or None)
        error_message: Error message if result is not successful
        status_code: HTTP status code for error
    
    Raises:
        HTTPException: If result is not successful
    """
    if result is None:
        raise HTTPException(status_code=status_code, detail=error_message)
    
    if isinstance(result, bool):
        if not result:
            raise HTTPException(status_code=status_code, detail=error_message)
    elif isinstance(result, dict):
        if "error" in result:
            raise HTTPException(status_code=status_code, detail=result["error"])
        if result.get("success") is False:
            raise HTTPException(status_code=status_code, detail=error_message)
    elif not result:
        raise HTTPException(status_code=status_code, detail=error_message)


def require_not_none(
    value: Any,
    error_message: str,
    status_code: int = 404
) -> None:
    """
    Require that a value is not None
    
    Args:
        value: Value to check
        error_message: Error message if value is None
        status_code: HTTP status code for error
    
    Raises:
        HTTPException: If value is None
    """
    if value is None:
        raise HTTPException(status_code=status_code, detail=error_message)


def extract_bearer_token(authorization: Optional[str]) -> str:
    """
    Extract Bearer token from authorization header
    
    Args:
        authorization: Authorization header value
    
    Returns:
        Extracted token
    
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Token de autorización requerido"
        )
    return authorization.replace("Bearer ", "")


def build_list_response_data(
    items: list,
    include_total: bool = True
) -> Dict[str, Any]:
    """
    Build standardized list response data
    
    Args:
        items: List of items
        include_total: Whether to include total count
    
    Returns:
        Dictionary with items and optional total
    """
    response = {"items": items, "count": len(items)}
    if include_total:
        response["total"] = len(items)
    return response


def check_service_error(result: Dict[str, Any]) -> Optional[str]:
    """
    Check if service result contains an error
    
    Args:
        result: Service result dictionary
    
    Returns:
        Error message if present, None otherwise
    """
    if isinstance(result, dict):
        if "error" in result:
            return result["error"]
        if result.get("success") is False:
            return result.get("message", "Operación fallida")
    return None


def validate_service_result(
    result: Any,
    error_message: str = "Service operation failed",
    status_code: int = 500,
    raise_on_error: bool = True
) -> bool:
    """
    Validate a service result and optionally raise an exception.
    
    Handles multiple result formats:
    - None -> Error
    - False (bool) -> Error
    - Dict with "error" key -> Error
    - Dict with "success": False -> Error
    - Empty list/string -> Error (optional)
    
    Args:
        result: Service result to validate
        error_message: Error message if validation fails
        status_code: HTTP status code for error
        raise_on_error: Whether to raise exception or return False
    
    Returns:
        True if result is valid, False if invalid (when raise_on_error=False)
    
    Raises:
        HTTPException: If result is invalid and raise_on_error=True
    """
    # Check for None
    if result is None:
        if raise_on_error:
            raise HTTPException(status_code=status_code, detail=error_message)
        return False
    
    # Check for boolean False
    if isinstance(result, bool):
        if not result:
            if raise_on_error:
                raise HTTPException(status_code=status_code, detail=error_message)
            return False
        return True
    
    # Check for dict with error
    if isinstance(result, dict):
        error_msg = result.get("error")
        if error_msg:
            if raise_on_error:
                raise HTTPException(
                    status_code=status_code,
                    detail=error_msg or error_message
                )
            return False
        
        if result.get("success") is False:
            msg = result.get("message", error_message)
            if raise_on_error:
                raise HTTPException(status_code=status_code, detail=msg)
            return False
    
    # Result is valid
    return True
