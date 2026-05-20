"""
Diffusion Service API Endpoints
FastAPI endpoints using modular architecture.
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from pydantic import BaseModel, Field
from typing import Optional
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from ..core.service_core import DiffusionServiceCore


class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(default=None)
    model_name: str = Field(default="runwayml/stable-diffusion-v1-5")
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    seed: Optional[int] = Field(default=None)


router = APIRouter(prefix="/api/v1", tags=["diffusion"])


def get_service_core() -> DiffusionServiceCore:
    """Get service core instance."""
    from ..config import get_config
    config = get_config()
    return DiffusionServiceCore(config=config)


@router.get("/health")
async def health_check(service: DiffusionServiceCore = Depends(get_service_core)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "diffusion_service",
        "device": service.device,
        "cached_pipelines": len(service._pipeline_cache),
    }


@router.post("/text-to-image")
async def text_to_image(
    request: TextToImageRequest,
    service: DiffusionServiceCore = Depends(get_service_core),
):
    """Generate image from text prompt."""
    try:
        image = await service.generate_image(
            prompt=request.prompt,
            model_name=request.model_name,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )
        
        if image is None:
            raise HTTPException(status_code=500, detail="Image generation failed")
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")



