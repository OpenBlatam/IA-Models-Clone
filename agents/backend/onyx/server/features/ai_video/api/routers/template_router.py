from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from ..dependencies.auth import get_current_user, require_permissions
from ..dependencies.rate_limit import check_rate_limit
from ..schemas.template_schemas import (
from ..schemas.video_schemas import APIResponse
from ..services.template_service import (
from ..utils.response import create_error_response, create_success_response
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Template Router - AI Avatar and Template System
==============================================

Router for template selection, AI avatar creation, and image synchronization.
"""



    AvatarPreviewRequest,
    AvatarPreviewResponse,
    TemplateCategory,
    TemplateListResponse,
    TemplateVideoRequest,
    TemplateVideoResponse,
)
    create_avatar_preview,
    create_template_video,
    get_avatar_preview_status,
    get_template_by_id,
    get_templates_list,
    get_template_video_status,
)


router = APIRouter()


@router.get(
    "/templates",
    response_model=APIResponse,
    summary="Get available templates",
    description="Retrieve list of available video templates with filtering options",
)
async def list_templates(
    category: Optional[TemplateCategory] = Query(None, description="Filter by category"),
    premium_only: bool = Query(False, description="Show only premium templates"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search templates by name/description"),
    skip: int = Query(0, ge=0, description="Number of templates to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of templates to return"),
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["template:read"])),
) -> APIResponse:
    """
    Get list of available video templates.
    
    Supports filtering by category, premium status, tags, and search.
    """
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        templates_response = await get_templates_list(
            category=category,
            premium_only=premium_only,
            tags=tag_list,
            search=search,
            skip=skip,
            limit=limit,
            user_id=current_user["sub"],
        )
        
        return create_success_response(data=templates_response)
        
    except Exception as e:
        return create_error_response(
            message="Failed to retrieve templates",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/templates/{template_id}",
    response_model=APIResponse,
    summary="Get template details",
    description="Retrieve detailed information about a specific template",
)
async def get_template_details(
    template_id: str,
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["template:read"])),
) -> APIResponse:
    """
    Get detailed information about a specific template.
    
    Returns template configuration, features, and requirements.
    """
    try:
        template_info = await get_template_by_id(
            template_id=template_id,
            user_id=current_user["sub"],
        )
        
        if not template_info:
            return create_error_response(
                message="Template not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        return create_success_response(data=template_info)
        
    except Exception as e:
        return create_error_response(
            message="Failed to retrieve template details",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/avatar/preview",
    response_model=APIResponse,
    summary="Generate avatar preview",
    description="Create a preview of the AI avatar with given configuration",
)
async def create_avatar_preview_endpoint(
    request: AvatarPreviewRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
    _permissions: None = Depends(require_permissions(["avatar:create"])),
) -> APIResponse:
    """
    Generate a preview of the AI avatar.
    
    Creates a short video preview showing the avatar speaking sample text.
    """
    try:
        preview_response = await create_avatar_preview(
            request=request,
            user_id=current_user["sub"],
            background_tasks=background_tasks,
        )
        
        return create_success_response(
            data=preview_response,
            message="Avatar preview generation started",
        )
        
    except ValueError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    except Exception as e:
        return create_error_response(
            message="Failed to create avatar preview",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/avatar/preview/{preview_id}",
    response_model=APIResponse,
    summary="Get avatar preview status",
    description="Check the status of an avatar preview generation",
)
async def get_avatar_preview(
    preview_id: str,
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["avatar:read"])),
) -> APIResponse:
    """
    Get avatar preview generation status and result.
    
    Returns preview video URL when ready.
    """
    try:
        preview_status = await get_avatar_preview_status(
            preview_id=preview_id,
            user_id=current_user["sub"],
        )
        
        if not preview_status:
            return create_error_response(
                message="Avatar preview not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        return create_success_response(data=preview_status)
        
    except Exception as e:
        return create_error_response(
            message="Failed to retrieve avatar preview",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/videos/template",
    response_model=APIResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create template-based video",
    description="Generate a video using template, AI avatar, and image synchronization",
)
async def create_template_video_endpoint(
    request: TemplateVideoRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
    _permissions: None = Depends(require_permissions(["video:create", "template:use"])),
) -> APIResponse:
    """
    Create a video using a template with AI avatar and image synchronization.
    
    This is the main endpoint for the complete template-based video generation workflow:
    1. Validates template and user permissions
    2. Generates script based on configuration
    3. Creates AI avatar with voice synthesis
    4. Synchronizes images according to script timing
    5. Composes final video with all elements
    """
    try:
        # Validate template access
        template_info = await get_template_by_id(
            template_id=request.template_id,
            user_id=current_user["sub"],
        )
        
        if not template_info:
            return create_error_response(
                message="Template not found or access denied",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check premium access if needed
        if template_info.is_premium and not current_user.get("has_premium", False):
            return create_error_response(
                message="Premium subscription required for this template",
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        # Create template video
        video_response = await create_template_video(
            request=request,
            user_id=current_user["sub"],
            background_tasks=background_tasks,
        )
        
        return create_success_response(
            data=video_response,
            message="Template video generation started successfully",
        )
        
    except ValueError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Failed to create template video",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/videos/template/{request_id}",
    response_model=APIResponse,
    summary="Get template video status",
    description="Check the status of a template-based video generation",
)
async def get_template_video(
    request_id: str,
    current_user: dict = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["video:read"])),
) -> APIResponse:
    """
    Get template video generation status and results.
    
    Returns detailed status for each processing stage and final video URLs.
    """
    try:
        video_status = await get_template_video_status(
            request_id=request_id,
            user_id=current_user["sub"],
        )
        
        if not video_status:
            return create_error_response(
                message="Template video not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        return create_success_response(data=video_status)
        
    except PermissionError as e:
        return create_error_response(
            message=str(e),
            status_code=status.HTTP_403_FORBIDDEN,
        )
    
    except Exception as e:
        return create_error_response(
            message="Failed to retrieve template video status",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) 