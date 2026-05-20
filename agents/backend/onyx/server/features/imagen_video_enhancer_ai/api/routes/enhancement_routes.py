"""
Enhancement Routes
=================

API routes for enhancement services.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ...utils.validators import FileValidator, ParameterValidator, ValidationError
from ...utils.image_utils import get_image_info, estimate_enhancement_time
from ..route_helpers import (
    save_uploaded_file,
    parse_json_options,
    handle_route_error,
    create_success_response
)
from ..dependencies import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["enhancement"])


@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    enhancement_type: str = Form("general"),
    priority: int = Form(0),
    options: Optional[str] = Form(None)
):
    """Upload and enhance an image."""
    agent = get_agent()
    
    try:
        # Validate parameters
        ParameterValidator.validate_enhancement_type(enhancement_type)
        ParameterValidator.validate_priority(priority)
        
        # Save uploaded file
        upload_dir = agent.output_dirs["uploads"]
        file_path = await save_uploaded_file(file, str(upload_dir))
        
        # Validate file
        FileValidator.validate_image_file(
            file_path,
            max_size_mb=agent.config.max_file_size_mb,
            allowed_extensions=agent.config.allowed_image_formats
        )
        
        # Get image info
        image_info = get_image_info(file_path)
        estimated_time = estimate_enhancement_time(
            image_info["file_size_mb"],
            enhancement_type,
            is_video=False
        )
        
        # Parse options
        options_dict = parse_json_options(options)
        
        # Submit task
        task_id = await agent.enhance_image(
            file_path=file_path,
            enhancement_type=enhancement_type,
            options=options_dict,
            priority=priority
        )
        
        return create_success_response(
            data={
                "task_id": task_id,
                "file_path": file_path,
                "image_info": image_info,
                "estimated_time_seconds": estimated_time
            },
            message="Image uploaded and enhancement task created"
        )
        
    except Exception as e:
        raise handle_route_error(e, "Error uploading image")


@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    enhancement_type: str = Form("general"),
    priority: int = Form(0),
    options: Optional[str] = Form(None)
):
    """Upload and enhance a video."""
    agent = get_agent()
    
    try:
        # Validate parameters
        ParameterValidator.validate_enhancement_type(enhancement_type)
        ParameterValidator.validate_priority(priority)
        
        # Save uploaded file
        upload_dir = agent.output_dirs["uploads"]
        file_path = await save_uploaded_file(file, str(upload_dir))
        
        # Validate file
        FileValidator.validate_video_file(
            file_path,
            max_size_mb=agent.config.max_file_size_mb,
            allowed_extensions=agent.config.allowed_video_formats
        )
        
        # Analyze video
        video_analysis = await agent.video_processor.analyze_video(file_path)
        estimated_time = estimate_enhancement_time(
            video_analysis.get("file_size_mb", 0),
            enhancement_type,
            is_video=True
        )
        
        # Parse options
        options_dict = parse_json_options(options)
        
        # Submit task
        task_id = await agent.enhance_video(
            file_path=file_path,
            enhancement_type=enhancement_type,
            options=options_dict,
            priority=priority
        )
        
        return create_success_response(
            data={
                "task_id": task_id,
                "file_path": file_path,
                "video_analysis": video_analysis,
                "estimated_time_seconds": estimated_time
            },
            message="Video uploaded and enhancement task created"
        )
        
    except Exception as e:
        raise handle_route_error(e, "Error uploading video")

