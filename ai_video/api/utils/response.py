from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Optional
from ..schemas.video_schemas import APIResponse
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Response Utilities - RORO pattern helpers
"""



def create_success_response(
    data: Any = None,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> APIResponse:
    """
    Create standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        request_id: Optional request tracking ID
        processing_time_ms: Optional processing time
        
    Returns:
        APIResponse with success=True
    """
    return APIResponse(
        success=True,
        data=data,
        error=None,
        request_id=request_id,
        processing_time_ms=processing_time_ms,
    )


def create_error_response(
    message: str,
    status_code: int,
    request_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> APIResponse:
    """
    Create standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        request_id: Optional request tracking ID
        processing_time_ms: Optional processing time
        
    Returns:
        APIResponse with success=False
    """
    return APIResponse(
        success=False,
        data=None,
        error=message,
        request_id=request_id,
        processing_time_ms=processing_time_ms,
    ) 