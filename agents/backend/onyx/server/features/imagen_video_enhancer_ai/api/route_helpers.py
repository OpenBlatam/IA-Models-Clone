"""
Route Helpers
=============

Common helpers for API routes.
"""

import logging
import json
import shutil
from typing import Dict, Any, Optional, Callable
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..utils.validators import ValidationError
from ..utils.response_builder import ResponseBuilder
from ..utils.error_context import error_context

logger = logging.getLogger(__name__)


def handle_route_error(
    error: Exception,
    default_message: str = "An error occurred",
    status_code: int = 500
) -> HTTPException:
    """
    Handle route errors consistently.
    
    Args:
        error: Exception that occurred
        default_message: Default error message
        status_code: HTTP status code
        
    Returns:
        HTTPException
    """
    if isinstance(error, ValidationError):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, HTTPException):
        return error
    else:
        logger.error(f"Route error: {error}", exc_info=True)
        return HTTPException(
            status_code=status_code,
            detail=default_message if not str(error) else str(error)
        )


def parse_json_options(options: Optional[str]) -> Dict[str, Any]:
    """
    Parse JSON options string.
    
    Args:
        options: JSON string or None
        
    Returns:
        Parsed dictionary
    """
    if not options:
        return {}
    
    try:
        return json.loads(options)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in options: {e}")


async def save_uploaded_file(
    file: UploadFile,
    upload_dir: str,
    filename: Optional[str] = None
) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        file: Uploaded file
        upload_dir: Upload directory
        filename: Optional custom filename
        
    Returns:
        Path to saved file
    """
    from ..utils.file_helpers import ensure_unique_filename
    from pathlib import Path
    
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    file_path = ensure_unique_filename(
        upload_path,
        filename or file.filename or "upload"
    )
    
    # Save file content
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)


def create_success_response(
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    **kwargs
) -> JSONResponse:
    """
    Create success response.
    
    Args:
        data: Response data
        message: Success message
        **kwargs: Additional response fields
        
    Returns:
        JSONResponse
    """
    response_data = ResponseBuilder.success(
        data=data,
        message=message,
        metadata=kwargs
    )
    return JSONResponse(response_data)


def create_error_response(
    error: str | Exception,
    code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create error response.
    
    Args:
        error: Error message or exception
        code: Error code
        status_code: HTTP status code
        
    Returns:
        JSONResponse
    """
    response_data = ResponseBuilder.error(
        error=error,
        code=code
    )
    return JSONResponse(
        content=response_data,
        status_code=status_code
    )


def with_error_handling(
    error_message: str = "An error occurred",
    status_code: int = 500
):
    """
    Decorator for route error handling.
    
    Args:
        error_message: Default error message
        status_code: Default status code
        
    Usage:
        @with_error_handling()
        async def my_route():
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise handle_route_error(e, error_message, status_code)
        
        return wrapper
    
    return decorator


def validate_file_upload(
    file: UploadFile,
    allowed_types: list[str],
    max_size_mb: float
) -> None:
    """
    Validate file upload.
    
    Args:
        file: Uploaded file
        allowed_types: Allowed MIME types
        max_size_mb: Maximum file size in MB
        
    Raises:
        ValidationError: If validation fails
    """
    from ..utils.validators import ValidationError
    
    if not file.filename:
        raise ValidationError("Filename is required")
    
    # Check file extension
    file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if not any(file_ext in t.replace('.', '') for t in allowed_types):
        raise ValidationError(f"File type not allowed. Allowed types: {allowed_types}")
    
    # Note: File size validation should be done after saving
    # This is a basic validation

