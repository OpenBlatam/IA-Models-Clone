"""
Base router with common dependencies and utilities
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ...utils.logger import logger
from ...utils.rate_limiter import RateLimiter
from ...utils.endpoint_rate_limiter import EndpointRateLimiter
from ...utils.intelligent_cache import IntelligentCache
from ...utils.advanced_validator import AdvancedImageValidator
from ...utils.exceptions import (
    ImageProcessingError, VideoProcessingError,
    AnalysisError, ValidationError
)

# Shared instances (will be initialized by main router)
skin_analyzer = None
image_processor = None
video_processor = None
history_tracker = None
db_manager = None
alert_system = None
advanced_validator = AdvancedImageValidator()
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
endpoint_rate_limiter = EndpointRateLimiter()
intelligent_cache = IntelligentCache(default_ttl=3600, max_size=1000)


def initialize_base_services(
    analyzer,
    img_processor,
    vid_processor,
    tracker,
    database_manager,
    alerts
):
    """Initialize shared service instances"""
    global skin_analyzer, image_processor, video_processor
    global history_tracker, db_manager, alert_system
    
    skin_analyzer = analyzer
    image_processor = img_processor
    video_processor = vid_processor
    history_tracker = tracker
    db_manager = database_manager
    alert_system = alerts


def create_base_router(prefix: str = "", tags: Optional[list] = None) -> APIRouter:
    """Create a base router with common configuration"""
    return APIRouter(prefix=prefix, tags=tags or [])


def handle_api_error(e: Exception, endpoint: str, method: str = "POST") -> HTTPException:
    """Standardized error handling for API endpoints"""
    import time
    
    if isinstance(e, ValidationError):
        logger.warning(f"Validation error in {endpoint}: {str(e)}")
        return HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, (ImageProcessingError, VideoProcessingError, AnalysisError)):
        logger.error(f"Processing error in {endpoint}: {str(e)}")
        return HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, HTTPException):
        return e
    else:
        logger.error(f"Unexpected error in {endpoint}: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=500,
            detail=f"Error in {endpoint}: {str(e)}"
        )


def create_success_response(data: dict, message: str = "Success", processing_time: Optional[float] = None) -> JSONResponse:
    """Create a standardized success response"""
    response = {
        "success": True,
        "message": message,
        **data
    }
    if processing_time is not None:
        response["processing_time"] = round(processing_time, 2)
    return JSONResponse(content=response)




