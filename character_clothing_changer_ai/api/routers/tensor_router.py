"""
Tensor Router
=============

API endpoints for tensor operations.
"""

from fastapi import APIRouter, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Dict, Any
import logging

from ...core.clothing_changer_service import ClothingChangerService
from ...api.dependencies import get_service
from ...api.utils.error_handler import APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tensors"])


@router.post("/create-workflow", response_model=Dict[str, Any])
async def create_workflow(
    service: ClothingChangerService = Depends(get_service),
    tensor_path: str = Form(..., description="Path to safe tensor"),
    prompt: str = Form(..., description="Generation prompt"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt"),
    num_inference_steps: Optional[int] = Form(None, description="Inference steps"),
    guidance_scale: Optional[float] = Form(None, description="Guidance scale"),
):
    """
    Create ComfyUI workflow JSON from tensor.
    
    Returns:
        Workflow creation result
    """
    try:
        result = service.create_comfyui_workflow(
            tensor_path=tensor_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="create_workflow")


@router.get("/tensors", response_model=list[Dict[str, Any]])
async def list_tensors(service: ClothingChangerService = Depends(get_service)):
    """
    List all generated safe tensors.
    
    Returns:
        List of tensor information
    """
    try:
        tensors = service.list_tensors()
        return JSONResponse(content=tensors)
    
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="list_tensors")


@router.get("/tensor/{tensor_id}")
async def get_tensor(tensor_id: str, service: ClothingChangerService = Depends(get_service)):
    """
    Download a specific safe tensor.
    
    Args:
        tensor_id: Tensor filename or ID
        
    Returns:
        Safe tensor file
    """
    try:
        tensors = service.list_tensors()
        
        # Find tensor
        tensor = None
        for t in tensors:
            if t["filename"] == tensor_id or tensor_id in t["path"]:
                tensor = t
                break
        
        if tensor is None:
            raise HTTPException(status_code=404, detail="Tensor not found")
        
        return FileResponse(
            path=tensor["path"],
            filename=tensor["filename"],
            media_type="application/octet-stream",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="get_tensor")

