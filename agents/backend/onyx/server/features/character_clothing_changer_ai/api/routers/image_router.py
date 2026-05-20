"""
Image Router
===========

API endpoints for image operations.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import logging

from ...api.utils.error_handler import APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["images"])


@router.get("/image/{image_name}")
async def get_image(image_name: str):
    """
    Get a generated image by name.
    
    Args:
        image_name: Image filename
        
    Returns:
        Image file
    """
    try:
        temp_dir = Path(tempfile.gettempdir())
        image_path = temp_dir / image_name
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=image_path,
            filename=image_name,
            media_type="image/png",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="get_image")


