"""
Clothing Change Router
======================

FastAPI router for clothing change operations.

Endpoints:
- POST /clothing/change - Execute clothing change workflow
- GET /clothing/status/{prompt_id} - Get workflow execution status
- GET /clothing/analytics - Get service analytics
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, status, Request
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl, validator, HttpUrl as PydanticHttpUrl

from services.clothing_service import ClothingChangeService, DEFAULT_GUIDANCE_SCALE, DEFAULT_NUM_STEPS
from services.batch_service import BatchProcessingService
from services.metrics_service import get_metrics_service
from services.cache_service import get_cache_service
from services.rate_limiter import get_rate_limiter
from services.webhook_service import get_webhook_service, WebhookConfig, WebhookEvent

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class ClothingChangeRequest(BaseModel):
    """Request model for clothing change operation"""
    
    image_url: str = Field(
        ...,
        description="URL or path to input image",
        example="https://example.com/character.png"
    )
    clothing_description: str = Field(
        ...,
        description="Description of desired clothing",
        min_length=1,
        max_length=500,
        example="a red elegant dress with floral patterns"
    )
    mask_url: Optional[str] = Field(
        None,
        description="URL or path to mask image (optional)",
        example="https://example.com/mask.png"
    )
    character_name: Optional[str] = Field(
        None,
        description="Name of character for context",
        max_length=100,
        example="MyCharacter"
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt to avoid certain elements",
        max_length=500,
        example="blurry, low quality, distorted"
    )
    guidance_scale: float = Field(
        DEFAULT_GUIDANCE_SCALE,
        description="Guidance scale for generation",
        ge=1.0,
        le=100.0,
        example=50.0
    )
    num_steps: int = Field(
        DEFAULT_NUM_STEPS,
        description="Number of inference steps",
        ge=1,
        le=100,
        example=12
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility",
        ge=0,
        example=12345
    )
    optimize_prompt: bool = Field(
        True,
        description="Whether to optimize prompt with OpenRouter"
    )
    
    @validator("clothing_description")
    def validate_clothing_description(cls, v):
        """Validate clothing description is not empty"""
        if not v or not v.strip():
            raise ValueError("clothing_description cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://example.com/character.png",
                "clothing_description": "a red elegant dress",
                "mask_url": "https://example.com/mask.png",
                "character_name": "MyCharacter",
                "negative_prompt": "blurry, low quality",
                "guidance_scale": 50.0,
                "num_steps": 12,
                "seed": 12345,
                "optimize_prompt": True
            }
        }


class ClothingChangeResponse(BaseModel):
    """Response model for successful clothing change"""
    
    success: bool = Field(..., description="Whether operation succeeded")
    prompt_id: Optional[str] = Field(None, description="ComfyUI prompt ID")
    original_prompt: str = Field(..., description="Original user prompt")
    optimized_prompt: str = Field(..., description="Final optimized prompt")
    openrouter_used: bool = Field(..., description="Whether OpenRouter was used")
    truthgpt_used: bool = Field(..., description="Whether TruthGPT was used")
    workflow_status: Optional[str] = Field(None, description="ComfyUI workflow status")
    settings: Dict[str, Any] = Field(..., description="Generation settings used")


class ErrorResponse(BaseModel):
    """Error response model"""
    
    success: bool = Field(False, description="Operation failed")
    error: str = Field(..., description="Error message")
    original_prompt: Optional[str] = Field(None, description="Original prompt if available")


class StatusResponse(BaseModel):
    """Response model for workflow status"""
    
    prompt_id: str = Field(..., description="ComfyUI prompt ID")
    queue_status: Dict[str, Any] = Field(..., description="ComfyUI queue status")
    status: Optional[str] = Field(None, description="Workflow status")
    error: Optional[str] = Field(None, description="Error message if failed")


class FaceSwapRequest(BaseModel):
    """Request model for face swap operation"""
    
    image_url: str = Field(
        ...,
        description="URL or path to input image (the image in inpainting)",
        example="https://example.com/image_in_painting.png"
    )
    face_url: str = Field(
        ...,
        description="URL or path to face image to swap in",
        example="https://example.com/new_face.png"
    )
    mask_url: Optional[str] = Field(
        None,
        description="URL or path to mask image (optional)",
        example="https://example.com/mask.png"
    )
    character_name: Optional[str] = Field(
        None,
        description="Name of character for context",
        max_length=100,
        example="MyCharacter"
    )
    prompt: Optional[str] = Field(
        None,
        description="Custom prompt (optional, uses default face swap prompt if not provided)",
        max_length=500,
        example="best quality, face swap, high quality portrait"
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt to avoid certain elements",
        max_length=500,
        example="blurry, low quality, distorted"
    )
    guidance_scale: float = Field(
        DEFAULT_GUIDANCE_SCALE,
        description="Guidance scale for generation",
        ge=1.0,
        le=100.0,
        example=50.0
    )
    num_steps: int = Field(
        DEFAULT_NUM_STEPS,
        description="Number of inference steps",
        ge=1,
        le=100,
        example=12
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility",
        ge=0,
        example=12345
    )
    optimize_prompt: bool = Field(
        True,
        description="Whether to optimize prompt with OpenRouter"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://example.com/image_in_painting.png",
                "face_url": "https://example.com/new_face.png",
                "mask_url": "https://example.com/mask.png",
                "character_name": "MyCharacter",
                "prompt": "best quality, face swap, high quality portrait",
                "negative_prompt": "blurry, low quality",
                "guidance_scale": 50.0,
                "num_steps": 12,
                "seed": 12345,
                "optimize_prompt": True
            }
        }


class FaceSwapResponse(BaseModel):
    """Response model for successful face swap"""
    
    success: bool = Field(..., description="Whether operation succeeded")
    prompt_id: Optional[str] = Field(None, description="ComfyUI prompt ID")
    original_prompt: Optional[str] = Field(None, description="Original user prompt")
    optimized_prompt: str = Field(..., description="Final optimized prompt")
    openrouter_used: bool = Field(..., description="Whether OpenRouter was used")
    truthgpt_used: bool = Field(..., description="Whether TruthGPT was used")
    workflow_status: Optional[str] = Field(None, description="ComfyUI workflow status")
    face_swap: bool = Field(True, description="Face swap operation flag")
    face_url: str = Field(..., description="URL of the face image used")
    settings: Dict[str, Any] = Field(..., description="Generation settings used")


# Dependency Injection
def get_clothing_service() -> ClothingChangeService:
    """
    Dependency to get ClothingChangeService instance.
    
    Returns:
        ClothingChangeService instance
    """
    return ClothingChangeService()


def get_batch_service() -> BatchProcessingService:
    """
    Dependency to get BatchProcessingService instance.
    
    Returns:
        BatchProcessingService instance
    """
    return BatchProcessingService()


# Endpoints
@router.post(
    "/clothing/change",
    response_model=ClothingChangeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Clothing change initiated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Change Character Clothing",
    description="""
    Change character clothing with AI assistance.
    
    This endpoint orchestrates the complete workflow:
    1. Validates input parameters
    2. Optimizes prompt with OpenRouter (if enabled)
    3. Enhances with TruthGPT (if enabled)
    4. Executes ComfyUI workflow
    
    Returns the prompt_id which can be used to check status.
    """
)
async def change_clothing(
    request: ClothingChangeRequest,
    service: ClothingChangeService = Depends(get_clothing_service)
) -> ClothingChangeResponse:
    """
    Change character clothing with AI assistance.
    
    Args:
        request: Clothing change request parameters
        service: ClothingChangeService instance (injected)
        
    Returns:
        ClothingChangeResponse with execution details
        
    Raises:
        HTTPException: If request is invalid or execution fails
    """
    try:
        logger.info(f"Received clothing change request for: {request.clothing_description[:50]}...")
        
        result = await service.change_clothing(
            image_url=request.image_url,
            clothing_description=request.clothing_description,
            mask_url=request.mask_url,
            character_name=request.character_name,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_steps=request.num_steps,
            seed=request.seed,
            optimize_prompt=request.optimize_prompt
        )
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Clothing change failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        logger.info(f"Clothing change successful, prompt_id: {result.get('prompt_id')}")
        return ClothingChangeResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in change_clothing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/clothing/status/{prompt_id}",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Status retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid prompt_id"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get Workflow Status",
    description="Get the status of a workflow execution by prompt_id"
)
async def get_status(
    prompt_id: str,
    service: ClothingChangeService = Depends(get_clothing_service)
) -> StatusResponse:
    """
    Get status of a workflow execution.
    
    Args:
        prompt_id: ComfyUI prompt ID to check
        service: ClothingChangeService instance (injected)
        
    Returns:
        StatusResponse with workflow status
        
    Raises:
        HTTPException: If prompt_id is invalid or status check fails
    """
    if not prompt_id or not prompt_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_id is required and cannot be empty"
        )
    
    try:
        logger.debug(f"Checking status for prompt_id: {prompt_id}")
        result = await service.get_status(prompt_id)
        
        if result.get("error"):
            logger.warning(f"Error getting status for {prompt_id}: {result.get('error')}")
        
        return StatusResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {prompt_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@router.post(
    "/face-swap",
    response_model=FaceSwapResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Face swap initiated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Swap Face in Inpainting Image",
    description="""
    Swap face in an image that's being processed with inpainting.
    
    This endpoint specifically handles face swap operations:
    1. Takes an image that's in inpainting
    2. Swaps the face with a new face from face_url
    3. Uses OpenRouter and TruthGPT for optimization
    4. Executes ComfyUI workflow with face swap
    
    The workflow uses the "Load New Face" node to load the replacement face
    and integrates it with the inpainting process.
    
    Returns the prompt_id which can be used to check status.
    """
)
async def face_swap(
    request: FaceSwapRequest,
    service: ClothingChangeService = Depends(get_clothing_service)
) -> FaceSwapResponse:
    """
    Swap face in an image that's being processed with inpainting.
    
    Args:
        request: Face swap request parameters
        service: ClothingChangeService instance (injected)
        
    Returns:
        FaceSwapResponse with execution details
        
    Raises:
        HTTPException: If request is invalid or execution fails
    """
    try:
        logger.info(f"Received face swap request: image={request.image_url[:50]}..., face={request.face_url[:50]}...")
        
        result = await service.face_swap(
            image_url=request.image_url,
            face_url=request.face_url,
            mask_url=request.mask_url,
            character_name=request.character_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_steps=request.num_steps,
            seed=request.seed,
            optimize_prompt=request.optimize_prompt
        )
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Face swap failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        logger.info(f"Face swap successful, prompt_id: {result.get('prompt_id')}")
        return FaceSwapResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in face_swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/clothing/cancel/{prompt_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel Workflow",
    description="Cancel a queued or running workflow by prompt_id"
)
async def cancel_workflow(
    prompt_id: str,
    service: ClothingChangeService = Depends(get_clothing_service)
) -> Dict[str, Any]:
    """
    Cancel a workflow execution.
    
    Args:
        prompt_id: Prompt ID to cancel
        service: ClothingChangeService instance (injected)
        
    Returns:
        Dictionary with cancellation result
    """
    if not prompt_id or not prompt_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_id is required"
        )
    
    try:
        result = await service.comfyui_service.cancel_prompt(prompt_id)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to cancel prompt")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}"
        )


@router.get(
    "/clothing/images/{prompt_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Output Images",
    description="Get output images for a completed workflow"
)
async def get_output_images(
    prompt_id: str,
    service: ClothingChangeService = Depends(get_clothing_service)
) -> Dict[str, Any]:
    """
    Get output images for a completed prompt.
    
    Args:
        prompt_id: Prompt ID to get images for
        service: ClothingChangeService instance (injected)
        
    Returns:
        Dictionary with list of output images
    """
    if not prompt_id or not prompt_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_id is required"
        )
    
    try:
        images = await service.comfyui_service.get_output_images(prompt_id)
        return {
            "prompt_id": prompt_id,
            "image_count": len(images),
            "images": images
        }
    except Exception as e:
        logger.error(f"Error getting output images: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get output images: {str(e)}"
        )


@router.get(
    "/clothing/workflow/info",
    status_code=status.HTTP_200_OK,
    summary="Get Workflow Info",
    description="Get information about the loaded workflow template"
)
async def get_workflow_info(
    service: ClothingChangeService = Depends(get_clothing_service)
) -> Dict[str, Any]:
    """
    Get information about the workflow template.
    
    Args:
        service: ClothingChangeService instance (injected)
        
    Returns:
        Dictionary with workflow information
    """
    try:
        return service.comfyui_service.get_workflow_info()
    except Exception as e:
        logger.error(f"Error getting workflow info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow info: {str(e)}"
        )


@router.post(
    "/clothing/batch",
    status_code=status.HTTP_200_OK,
    summary="Batch Clothing Change",
    description="Process multiple clothing changes in batch"
)
async def batch_clothing_change(
    items: List[ClothingChangeRequest],
    max_concurrent: Optional[int] = None,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    Process multiple clothing changes in batch.
    
    Args:
        items: List of clothing change requests
        max_concurrent: Maximum concurrent operations (default: 5)
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with batch results
    """
    try:
        # Convert Pydantic models to dicts
        items_dict = [item.dict() for item in items]
        
        result = await batch_service.batch_clothing_change(
            items=items_dict,
            max_concurrent=max_concurrent
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch clothing change: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.post(
    "/face-swap/batch",
    status_code=status.HTTP_200_OK,
    summary="Batch Face Swap",
    description="Process multiple face swaps in batch"
)
async def batch_face_swap(
    items: List[FaceSwapRequest],
    max_concurrent: Optional[int] = None,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    Process multiple face swaps in batch.
    
    Args:
        items: List of face swap requests
        max_concurrent: Maximum concurrent operations (default: 5)
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with batch results
    """
    try:
        # Convert Pydantic models to dicts
        items_dict = [item.dict() for item in items]
        
        result = await batch_service.batch_face_swap(
            items=items_dict,
            max_concurrent=max_concurrent
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch face swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get(
    "/batch/status/{batch_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Batch Status",
    description="Get status of a batch operation"
)
async def get_batch_status(
    batch_id: str,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    Get status of a batch operation.
    
    Args:
        batch_id: Batch operation ID
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with batch status
    """
    try:
        return await batch_service.get_batch_status(batch_id)
    except Exception as e:
        logger.error(f"Error getting batch status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch status: {str(e)}"
        )


@router.post(
    "/batch/cancel/{batch_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel Batch",
    description="Cancel a batch operation"
)
async def cancel_batch(
    batch_id: str,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    Cancel a batch operation.
    
    Args:
        batch_id: Batch operation ID
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with cancellation result
    """
    try:
        return await batch_service.cancel_batch(batch_id)
    except Exception as e:
        logger.error(f"Error cancelling batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel batch: {str(e)}"
        )


@router.get(
    "/batch/list",
    status_code=status.HTTP_200_OK,
    summary="List Batches",
    description="List all active batch operations with optional filtering"
)
async def list_batches(
    status: Optional[str] = None,
    operation_type: Optional[str] = None,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    List all active batch operations.
    
    Args:
        status: Optional status filter ("pending", "processing", "completed", "failed", "cancelled")
        operation_type: Optional operation type filter ("clothing_change" or "face_swap")
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with list of batches
    """
    try:
        from services.batch_service import BatchStatus
        
        status_filter = None
        if status:
            try:
                status_filter = BatchStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}. Must be one of: pending, processing, completed, failed, cancelled"
                )
        
        batches = await batch_service.list_batches(
            status_filter=status_filter,
            operation_type_filter=operation_type
        )
        
        return {
            "count": len(batches),
            "batches": batches
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing batches: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batches: {str(e)}"
        )


@router.post(
    "/batch/cleanup",
    status_code=status.HTTP_200_OK,
    summary="Cleanup Batches",
    description="Clean up completed or cancelled batches older than specified hours"
)
async def cleanup_batches(
    older_than_hours: float = 24.0,
    batch_service: BatchProcessingService = Depends(get_batch_service)
) -> Dict[str, Any]:
    """
    Clean up completed or cancelled batches.
    
    Args:
        older_than_hours: Remove batches older than this many hours (default: 24.0)
        batch_service: BatchProcessingService instance (injected)
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        if older_than_hours <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="older_than_hours must be positive"
            )
        
        return await batch_service.cleanup_completed_batches(older_than_hours=older_than_hours)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up batches: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup batches: {str(e)}"
        )


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get Metrics",
    description="Get service metrics and statistics"
)
async def get_metrics(
    time_range: Optional[str] = None,
    metrics_service = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """
    Get service metrics and statistics.
    
    Args:
        time_range: Optional time range ("hour", "day", "week", or None for all)
        metrics_service: MetricsService instance (injected)
        
    Returns:
        Dictionary with metrics
    """
    try:
        return metrics_service.get_metrics(time_range=time_range)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get(
    "/metrics/recent",
    status_code=status.HTTP_200_OK,
    summary="Get Recent Operations",
    description="Get recent operation metrics"
)
async def get_recent_operations(
    limit: int = 10,
    metrics_service = Depends(get_metrics_service)
) -> Dict[str, Any]:
    """
    Get recent operation metrics.
    
    Args:
        limit: Maximum number of operations to return (default: 10, max: 100)
        metrics_service: MetricsService instance (injected)
        
    Returns:
        Dictionary with recent operations
    """
    try:
        limit = min(max(1, limit), 100)  # Clamp between 1 and 100
        operations = metrics_service.get_recent_operations(limit=limit)
        return {
            "count": len(operations),
            "operations": operations
        }
    except Exception as e:
        logger.error(f"Error getting recent operations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent operations: {str(e)}"
        )


@router.get(
    "/cache/stats",
    status_code=status.HTTP_200_OK,
    summary="Get Cache Statistics",
    description="Get cache statistics and information"
)
async def get_cache_stats(
    cache_service = Depends(get_cache_service)
) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Args:
        cache_service: CacheService instance (injected)
        
    Returns:
        Dictionary with cache statistics
    """
    try:
        return cache_service.get_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear Cache",
    description="Clear all cache entries"
)
async def clear_cache(
    cache_service = Depends(get_cache_service)
) -> Dict[str, Any]:
    """
    Clear all cache entries.
    
    Args:
        cache_service: CacheService instance (injected)
        
    Returns:
        Dictionary with cleanup result
    """
    try:
        count = cache_service.clear()
        return {
            "success": True,
            "cleared_entries": count,
            "message": f"Cleared {count} cache entries"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get(
    "/rate-limit/info",
    status_code=status.HTTP_200_OK,
    summary="Get Rate Limit Info",
    description="Get rate limit information for current client"
)
async def get_rate_limit_info(
    client_id: Optional[str] = None,
    rate_limiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """
    Get rate limit information.
    
    Args:
        client_id: Optional client identifier
        rate_limiter: RateLimiter instance (injected)
        
    Returns:
        Dictionary with rate limit information
    """
    try:
        return rate_limiter.get_info(client_id=client_id)
    except Exception as e:
        logger.error(f"Error getting rate limit info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit info: {str(e)}"
        )


@router.get(
    "/rate-limit/stats",
    status_code=status.HTTP_200_OK,
    summary="Get Rate Limiter Statistics",
    description="Get rate limiter statistics for all clients"
)
async def get_rate_limit_stats(
    rate_limiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """
    Get rate limiter statistics.
    
    Args:
        rate_limiter: RateLimiter instance (injected)
        
    Returns:
        Dictionary with rate limiter statistics
    """
    try:
        return rate_limiter.get_stats()
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit stats: {str(e)}"
        )


@router.post(
    "/rate-limit/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset Rate Limit",
    description="Reset rate limit for a client"
)
async def reset_rate_limit(
    client_id: Optional[str] = None,
    rate_limiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """
    Reset rate limit for a client.
    
    Args:
        client_id: Optional client identifier
        rate_limiter: RateLimiter instance (injected)
        
    Returns:
        Dictionary with reset result
    """
    try:
        success = rate_limiter.reset_client(client_id=client_id)
        return {
            "success": success,
            "client_id": client_id or "default",
            "message": "Rate limit reset" if success else "Client not found"
        }
    except Exception as e:
        logger.error(f"Error resetting rate limit: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset rate limit: {str(e)}"
        )


@router.get(
    "/clothing/analytics",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Service Analytics",
    description="Get analytics and status from all integrated services"
)
async def get_analytics(
    service: ClothingChangeService = Depends(get_clothing_service)
) -> Dict[str, Any]:
    """
    Get analytics from all services.
    
    Args:
        service: ClothingChangeService instance (injected)
        
    Returns:
        Dictionary with analytics from all enabled services
        
    Raises:
        HTTPException: If analytics retrieval fails
    """
    try:
        logger.debug("Retrieving service analytics")
        analytics = await service.get_analytics()
        
        # Add metrics
        metrics_service = get_metrics_service()
        analytics["metrics"] = metrics_service.get_metrics()
        
        # Add cache stats
        cache_service = get_cache_service()
        analytics["cache"] = cache_service.get_stats()
        
        # Add rate limiter stats
        rate_limiter = get_rate_limiter()
        analytics["rate_limiter"] = rate_limiter.get_stats()
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.post(
    "/webhooks/register",
    status_code=status.HTTP_200_OK,
    summary="Register Webhook",
    description="Register a webhook for event notifications"
)
async def register_webhook(
    url: str = Field(..., description="Webhook URL"),
    secret: Optional[str] = Field(None, description="Optional secret for signature verification"),
    events: Optional[List[str]] = Field(None, description="List of events to subscribe to"),
    timeout: float = Field(30.0, description="Request timeout in seconds"),
    retries: int = Field(3, description="Number of retry attempts"),
    webhook_service = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Register a webhook for event notifications.
    
    Args:
        url: Webhook URL
        secret: Optional secret for signature verification
        events: List of events (default: all events)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        webhook_service: WebhookService instance (injected)
        
    Returns:
        Dictionary with registration result
    """
    try:
        config = WebhookConfig(
            url=url,
            secret=secret,
            timeout=timeout,
            retries=retries,
            events=events or ["workflow_completed", "workflow_failed", "batch_completed"]
        )
        
        webhook_service.register_webhook(config)
        
        return {
            "success": True,
            "url": url,
            "message": "Webhook registered successfully"
        }
    except Exception as e:
        logger.error(f"Error registering webhook: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register webhook: {str(e)}"
        )


@router.delete(
    "/webhooks/unregister",
    status_code=status.HTTP_200_OK,
    summary="Unregister Webhook",
    description="Unregister a webhook by URL"
)
async def unregister_webhook(
    url: str = Field(..., description="Webhook URL to unregister"),
    webhook_service = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Unregister a webhook.
    
    Args:
        url: Webhook URL to unregister
        webhook_service: WebhookService instance (injected)
        
    Returns:
        Dictionary with unregistration result
    """
    try:
        success = webhook_service.unregister_webhook(url)
        
        if success:
            return {
                "success": True,
                "url": url,
                "message": "Webhook unregistered successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook not found: {url}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister webhook: {str(e)}"
        )


@router.get(
    "/webhooks/list",
    status_code=status.HTTP_200_OK,
    summary="List Webhooks",
    description="List all registered webhooks"
)
async def list_webhooks(
    webhook_service = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    List all registered webhooks.
    
    Args:
        webhook_service: WebhookService instance (injected)
        
    Returns:
        Dictionary with list of webhooks
    """
    try:
        webhooks = webhook_service.get_webhooks()
        return {
            "count": len(webhooks),
            "webhooks": webhooks
        }
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list webhooks: {str(e)}"
        )

