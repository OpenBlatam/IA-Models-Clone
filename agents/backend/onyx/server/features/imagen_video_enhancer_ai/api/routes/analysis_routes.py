"""
Analysis Routes
===============

API routes for file analysis.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_agent
from ..models import AnalyzeRequest
from ...utils.validators import FileValidator, ParameterValidator, ValidationError
from ...utils.image_utils import get_image_info

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analyze")
async def analyze_file(request: AnalyzeRequest):
    """Analyze a file (image or video)."""
    agent = get_agent()
    
    try:
        ParameterValidator.validate_file_path(request.file_path)
        
        # Determine file type
        file_type = request.file_type or FileValidator.get_file_type(request.file_path)
        
        if file_type == "image":
            info = get_image_info(request.file_path)
            return JSONResponse({
                "file_type": "image",
                "info": info
            })
        elif file_type == "video":
            analysis = await agent.video_processor.analyze_video(request.file_path)
            return JSONResponse({
                "file_type": "video",
                "analysis": analysis
            })
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
            
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




