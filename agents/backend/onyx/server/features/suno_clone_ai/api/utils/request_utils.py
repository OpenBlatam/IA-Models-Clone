"""
Request utility functions for common API operations.

Consolidates common request parsing and extraction patterns.
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import Request, Query, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def get_query_param(
    request: Request,
    param_name: str,
    default: Any = None,
    param_type: type = str
) -> Any:
    """
    Extract query parameter from request.
    
    Args:
        request: FastAPI request
        param_name: Parameter name
        default: Default value if not found
        param_type: Type to convert to
    
    Returns:
        Parameter value or default
    """
    value = request.query_params.get(param_name, default)
    if value is None:
        return default
    
    try:
        if param_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        return param_type(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{param_name}' to {param_type}, using default")
        return default


def get_header(
    request: Request,
    header_name: str,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Extract header from request.
    
    Args:
        request: FastAPI request
        header_name: Header name (case-insensitive)
        default: Default value if not found
    
    Returns:
        Header value or default
    """
    return request.headers.get(header_name.lower(), default)


def get_path_param(
    request: Request,
    param_name: str,
    param_type: type = str
) -> Any:
    """
    Extract path parameter from request.
    
    Args:
        request: FastAPI request
        param_name: Parameter name
        param_type: Type to convert to
    
    Returns:
        Parameter value
    
    Raises:
        KeyError: If parameter not found
        ValueError: If conversion fails
    """
    value = request.path_params.get(param_name)
    if value is None:
        raise KeyError(f"Path parameter '{param_name}' not found")
    
    try:
        return param_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert '{param_name}' to {param_type}: {e}")


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Checks X-Forwarded-For, X-Real-IP headers, then falls back to direct connection.
    
    Args:
        request: FastAPI request
    
    Returns:
        Client IP address
    """
    # Check X-Forwarded-For (first IP in chain)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


def get_user_agent(request: Request) -> Optional[str]:
    """
    Extract user agent from request.
    
    Args:
        request: FastAPI request
    
    Returns:
        User agent string or None
    """
    return request.headers.get("user-agent")


def get_request_body_size(request: Request) -> int:
    """
    Get request body size from Content-Length header.
    
    Args:
        request: FastAPI request
    
    Returns:
        Body size in bytes, or 0 if unknown
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            pass
    return 0


def extract_query_params(
    request: Request,
    param_names: List[str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract multiple query parameters at once.
    
    Args:
        request: FastAPI request
        param_names: List of parameter names to extract
        defaults: Optional dict of default values
    
    Returns:
        Dictionary of parameter values
    """
    defaults = defaults or {}
    result = {}
    
    for param_name in param_names:
        result[param_name] = get_query_param(
            request,
            param_name,
            default=defaults.get(param_name)
        )
    
    return result


def is_json_request(request: Request) -> bool:
    """
    Check if request has JSON content type.
    
    Args:
        request: FastAPI request
    
    Returns:
        True if JSON content type
    """
    content_type = request.headers.get("content-type", "").lower()
    return "application/json" in content_type


def is_form_request(request: Request) -> bool:
    """
    Check if request has form content type.
    
    Args:
        request: FastAPI request
    
    Returns:
        True if form content type
    """
    content_type = request.headers.get("content-type", "").lower()
    return "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type

