"""
Response Utilities
==================

Utility functions for creating consistent API responses.
"""

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


def create_success_response(
    data: Dict[str, Any],
    message: Optional[str] = None,
    status_code: int = 200,
) -> JSONResponse:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
        
    Returns:
        JSONResponse with success format
    """
    response = {
        "success": True,
        "data": data,
    }
    
    if message:
        response["message"] = message
    
    return JSONResponse(content=response, status_code=status_code)


def create_error_response(
    error: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 500,
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        error: Error message
        details: Optional error details
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error format
    """
    response = {
        "success": False,
        "error": error,
    }
    
    if details:
        response["details"] = details
    
    return JSONResponse(content=response, status_code=status_code)


def image_to_base64_response(
    result: Dict[str, Any],
    image_path_key: str = "image_path",
) -> Dict[str, Any]:
    """
    Add base64 image data to result dictionary.
    
    Args:
        result: Result dictionary
        image_path_key: Key in result dict containing image path
        
    Returns:
        Updated result dictionary with image_base64 and image_url
    """
    if image_path_key not in result:
        return result
    
    from pathlib import Path
    import base64
    
    image_path = Path(result[image_path_key])
    if not image_path.exists():
        return result
    
    try:
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            result["image_base64"] = f"data:image/png;base64,{img_data}"
            result["image_url"] = f"/api/v1/image/{image_path.name}"
    except Exception as e:
        # Log error but don't fail the request
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to convert image to base64: {e}")
    
    return result

