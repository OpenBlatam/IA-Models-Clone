from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.diffusion_service import DiffusionService, GenerationParams
from onyx.server.features.ads.optimized_config import settings
from typing import Any, List, Dict, Optional
import logging
"""
API endpoints for diffusion models and image generation.
"""


logger = setup_logger()
router = APIRouter(prefix="/diffusion", tags=["diffusion"])

# Initialize diffusion service
diffusion_service = DiffusionService()

# Request Models
class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    style_preset: Optional[str] = Field(None, description="Style preset for generation")
    model_name: str = Field("runwayml/stable-diffusion-v1-5", description="Model to use for generation")

class ImageToImageRequest(BaseModel):
    """Request model for image-to-image generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Strength of transformation")
    model_name: str = Field("runwayml/stable-diffusion-v1-5", description="Model to use for generation")

class InpaintRequest(BaseModel):
    """Request model for image inpainting."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for inpainting")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    mask_type: str = Field("center", description="Type of mask to use (center, random, full)")
    model_name: str = Field("runwayml/stable-diffusion-inpainting", description="Model to use for inpainting")

class ControlNetRequest(BaseModel):
    """Request model for ControlNet generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    controlnet_type: str = Field("canny", description="Type of ControlNet (canny, depth)")

class LCMRequest(BaseModel):
    """Request model for LCM (Latent Consistency Model) generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(4, ge=1, le=8, description="Number of inference steps (LCM is fast)")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    model_name: str = Field("SimianLuo/LCM_Dreamshaper_v7", description="LCM model to use")

class TCDRequest(BaseModel):
    """Request model for TCD (Trajectory Consistency Distillation) generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(1, ge=1, le=4, description="Number of inference steps (TCD is very fast)")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    model_name: str = Field("h1t/TCD-SD15", description="TCD model to use")

class CustomSchedulerRequest(BaseModel):
    """Request model for custom scheduler generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    scheduler_type: str = Field("DPM++", description="Scheduler type (DDIM, PNDM, Euler, DPM++, Heun, etc.)")
    model_name: str = Field("runwayml/stable-diffusion-v1-5", description="Model to use")

class AdvancedGenerationRequest(BaseModel):
    """Request model for advanced generation with LoRA and Textual Inversion."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    negative_prompt: str = Field("", max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    model_name: str = Field("runwayml/stable-diffusion-v1-5", description="Model to use")
    scheduler_type: Optional[str] = Field(None, description="Custom scheduler type")
    use_lora: bool = Field(False, description="Whether to use LoRA")
    lora_path: Optional[str] = Field(None, description="Path to LoRA weights")
    use_textual_inversion: bool = Field(False, description="Whether to use Textual Inversion")
    textual_inversion_path: Optional[str] = Field(None, description="Path to Textual Inversion embeddings")

class DiffusionAnalysisRequest(BaseModel):
    """Request model for diffusion process analysis."""
    image_path: Optional[str] = Field(None, description="Path to image file")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    num_steps: int = Field(20, ge=5, le=100, description="Number of diffusion steps to analyze")
    save_visualization: bool = Field(False, description="Whether to save visualization")
    output_path: Optional[str] = Field(None, description="Path to save visualization")

class ForwardDiffusionRequest(BaseModel):
    """Request model for forward diffusion demonstration."""
    image_path: Optional[str] = Field(None, description="Path to image file")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    timesteps: Optional[List[int]] = Field(None, description="Specific timesteps to demonstrate")
    save_path: Optional[str] = Field(None, description="Path to save demonstration images")

class ReverseDiffusionRequest(BaseModel):
    """Request model for reverse diffusion demonstration."""
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_steps: int = Field(50, ge=10, le=200, description="Number of denoising steps")
    eta: float = Field(0.0, ge=0.0, le=1.0, description="Noise level for stochastic sampling")
    save_path: Optional[str] = Field(None, description="Path to save demonstration images")

class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""
    requests: List[TextToImageRequest] = Field(..., max_items=10, description="List of generation requests")
    priority: str = Field("normal", description="Priority level (low, normal, high)")

# Response Models
class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""
    id: str
    images: List[str]  # Base64 encoded images
    metadata: Dict[str, Any]
    generation_time: float
    cached: bool = False
    created_at: datetime

class BatchGenerationResponse(BaseModel):
    """Response model for batch generation."""
    batch_id: str
    results: List[ImageGenerationResponse]
    total_time: float
    created_at: datetime

class GenerationStatsResponse(BaseModel):
    """Response model for generation statistics."""
    total_cache_entries: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    loaded_models: int
    device: str
    timestamp: datetime

# Utility functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL image."""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

# API Endpoints
@router.post("/text-to-image", response_model=ImageGenerationResponse)
async def generate_text_to_image(request: TextToImageRequest):
    """Generate images from text prompt."""
    try:
        start_time = datetime.now()
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            style_preset=request.style_preset
        )
        
        # Generate images
        images = await diffusion_service.generate_text_to_image(
            params=params,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"text2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'model_name': request.model_name
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in text-to-image generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image-to-image", response_model=ImageGenerationResponse)
async def generate_image_to_image(
    request: ImageToImageRequest,
    init_image: UploadFile = File(...)
):
    """Generate images from initial image."""
    try:
        start_time = datetime.now()
        
        # Load initial image
        image_data = await init_image.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        init_image_pil = Image.open(BytesIO(image_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        if init_image_pil.mode != 'RGB':
            init_image_pil = init_image_pil.convert('RGB')
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_image_to_image(
            init_image=init_image_pil,
            params=params,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'strength': request.strength,
                'model_name': request.model_name
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in image-to-image generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inpaint", response_model=ImageGenerationResponse)
async def inpaint_image(
    request: InpaintRequest,
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None)
):
    """Inpaint image using mask."""
    try:
        start_time = datetime.now()
        
        # Load image
        image_data = await image.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        image_pil = Image.open(BytesIO(image_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Load or create mask
        if mask:
            mask_data = await mask.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            mask_pil = Image.open(BytesIO(mask_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if mask_pil.mode != 'L':
                mask_pil = mask_pil.convert('L')
        else:
            mask_pil = request.mask_type  # Use mask type string
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.inpaint_image(
            image=image_pil,
            mask=mask_pil,
            params=params,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'mask_type': request.mask_type,
                'model_name': request.model_name
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in image inpainting")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/controlnet", response_model=ImageGenerationResponse)
async def generate_with_controlnet(
    request: ControlNetRequest,
    control_image: UploadFile = File(...)
):
    """Generate images using ControlNet."""
    try:
        start_time = datetime.now()
        
        # Load control image
        image_data = await control_image.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        control_image_pil = Image.open(BytesIO(image_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        if control_image_pil.mode != 'RGB':
            control_image_pil = control_image_pil.convert('RGB')
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_with_controlnet(
            control_image=control_image_pil,
            params=params,
            controlnet_type=request.controlnet_type
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"controlnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'controlnet_type': request.controlnet_type
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in ControlNet generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lcm", response_model=ImageGenerationResponse)
async def generate_with_lcm(request: LCMRequest):
    """Generate images using LCM (Latent Consistency Model) for fast generation."""
    try:
        start_time = datetime.now()
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_with_lcm(
            params=params,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"lcm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'model_name': request.model_name,
                'method': 'lcm'
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in LCM generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tcd", response_model=ImageGenerationResponse)
async def generate_with_tcd(request: TCDRequest):
    """Generate images using TCD (Trajectory Consistency Distillation) for fast generation."""
    try:
        start_time = datetime.now()
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_with_tcd(
            params=params,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"tcd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'model_name': request.model_name,
                'method': 'tcd'
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in TCD generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom-scheduler", response_model=ImageGenerationResponse)
async def generate_with_custom_scheduler(request: CustomSchedulerRequest):
    """Generate images with custom scheduler."""
    try:
        start_time = datetime.now()
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_with_custom_scheduler(
            params=params,
            scheduler_type=request.scheduler_type,
            model_name=request.model_name
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"custom_scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'scheduler_type': request.scheduler_type,
                'model_name': request.model_name,
                'method': 'custom_scheduler'
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in custom scheduler generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/advanced", response_model=ImageGenerationResponse)
async def generate_with_advanced_options(request: AdvancedGenerationRequest):
    """Generate images with advanced Diffusers options."""
    try:
        start_time = datetime.now()
        
        # Create generation parameters
        params = GenerationParams(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed
        )
        
        # Generate images
        images = await diffusion_service.generate_with_advanced_options(
            params=params,
            model_name=request.model_name,
            scheduler_type=request.scheduler_type,
            use_lora=request.use_lora,
            lora_path=request.lora_path,
            use_textual_inversion=request.use_textual_inversion,
            textual_inversion_path=request.textual_inversion_path
        )
        
        # Encode images to base64
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ImageGenerationResponse(
            id=f"advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            images=encoded_images,
            metadata={
                'prompt': request.prompt,
                'negative_prompt': request.negative_prompt,
                'width': request.width,
                'height': request.height,
                'guidance_scale': request.guidance_scale,
                'num_inference_steps': request.num_inference_steps,
                'seed': request.seed,
                'model_name': request.model_name,
                'scheduler_type': request.scheduler_type,
                'use_lora': request.use_lora,
                'use_textual_inversion': request.use_textual_inversion,
                'method': 'advanced'
            },
            generation_time=generation_time,
            cached=False,
            created_at=datetime.now()
        )
        
            except Exception as e:
            logger.exception("Error in advanced generation")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-diffusion-process")
async def analyze_diffusion_process(request: DiffusionAnalysisRequest):
    """Analyze the forward diffusion process for a given image."""
    try:
        # Get image from request
        if request.image_path:
            image = request.image_path
        elif request.image_base64:
            image = decode_base64_to_image(request.image_base64)
        else:
            raise HTTPException(status_code=400, detail="Either image_path or image_base64 must be provided")
        
        # Analyze diffusion process
        analysis = await diffusion_service.analyze_diffusion_process(
            image=image,
            num_steps=request.num_steps,
            save_visualization=request.save_visualization,
            output_path=request.output_path
        )
        
        return {
            "analysis": analysis,
            "metadata": {
                "num_steps": request.num_steps,
                "save_visualization": request.save_visualization,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.exception("Error in diffusion process analysis")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demonstrate-forward-diffusion")
async def demonstrate_forward_diffusion(request: ForwardDiffusionRequest):
    """Demonstrate forward diffusion process at specific timesteps."""
    try:
        # Get image from request
        if request.image_path:
            image = request.image_path
        elif request.image_base64:
            image = decode_base64_to_image(request.image_base64)
        else:
            raise HTTPException(status_code=400, detail="Either image_path or image_base64 must be provided")
        
        # Demonstrate forward diffusion
        demonstration = await diffusion_service.demonstrate_forward_diffusion(
            image=image,
            timesteps=request.timesteps,
            save_path=request.save_path
        )
        
        # Convert images to base64 for response
        encoded_images = []
        for img in demonstration["images"]:
            encoded_images.append(encode_image_to_base64(img))
        
        return {
            "timesteps": demonstration["timesteps"],
            "results": demonstration["results"],
            "images": encoded_images,
            "metadata": demonstration["metadata"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error in forward diffusion demonstration")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demonstrate-reverse-diffusion")
async def demonstrate_reverse_diffusion(request: ReverseDiffusionRequest):
    """Demonstrate reverse diffusion process (denoising)."""
    try:
        # Create noise image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise_image = torch.randn(1, 3, request.height, request.width, device=device)
        
        # Demonstrate reverse diffusion
        demonstration = await diffusion_service.demonstrate_reverse_diffusion(
            noise_image=noise_image,
            num_steps=request.num_steps,
            eta=request.eta,
            save_path=request.save_path
        )
        
        # Convert images to base64 for response
        encoded_images = []
        for img in demonstration["images"]:
            encoded_images.append(encode_image_to_base64(img))
        
        return {
            "num_steps": demonstration["num_steps"],
            "eta": demonstration["eta"],
            "results": demonstration["results"],
            "images": encoded_images,
            "final_image": encode_image_to_base64(demonstration["final_image"]) if demonstration["final_image"] else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error in reverse diffusion demonstration")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diffusion-statistics")
async def get_diffusion_statistics():
    """Get comprehensive statistics about the diffusion process."""
    try:
        statistics = await diffusion_service.get_diffusion_statistics()
        
        return {
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error getting diffusion statistics")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-generate", response_model=BatchGenerationResponse)
async def batch_generate_images(request: BatchGenerationRequest):
    """Generate multiple images in batch."""
    try:
        start_time = datetime.now()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = []
        
        # Process each request in the batch
        for i, req in enumerate(request.requests):
            try:
                # Create generation parameters
                params = GenerationParams(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width,
                    height=req.height,
                    num_images=req.num_images,
                    guidance_scale=req.guidance_scale,
                    num_inference_steps=req.num_inference_steps,
                    seed=req.seed,
                    style_preset=req.style_preset
                )
                
                # Generate images
                images = await diffusion_service.generate_text_to_image(
                    params=params,
                    model_name=req.model_name
                )
                
                # Encode images to base64
                encoded_images = [encode_image_to_base64(img) for img in images]
                
                results.append(ImageGenerationResponse(
                    id=f"{batch_id}_item_{i}",
                    images=encoded_images,
                    metadata={
                        'prompt': req.prompt,
                        'negative_prompt': req.negative_prompt,
                        'width': req.width,
                        'height': req.height,
                        'guidance_scale': req.guidance_scale,
                        'num_inference_steps': req.num_inference_steps,
                        'seed': req.seed,
                        'model_name': req.model_name
                    },
                    generation_time=0.0,  # Will be calculated below
                    cached=False,
                    created_at=datetime.now()
                ))
                
            except Exception as e:
                logger.error(f"Error in batch item {i}: {e}")
                # Add error result
                results.append(ImageGenerationResponse(
                    id=f"{batch_id}_item_{i}_error",
                    images=[],
                    metadata={'error': str(e)},
                    generation_time=0.0,
                    cached=False,
                    created_at=datetime.now()
                ))
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return BatchGenerationResponse(
            batch_id=batch_id,
            results=results,
            total_time=total_time,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error in batch generation")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=GenerationStatsResponse)
async def get_generation_stats():
    """Get generation statistics."""
    try:
        stats = await diffusion_service.get_generation_stats()
        
        return GenerationStatsResponse(
            total_cache_entries=stats['total_cache_entries'],
            cache_hits=stats['cache_hits'],
            cache_misses=stats['cache_misses'],
            cache_hit_rate=stats['cache_hit_rate'],
            loaded_models=stats['loaded_models'],
            device=stats['device'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.exception("Error getting generation stats")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-cache")
async def cleanup_cache(max_age_hours: int = 24):
    """Clean up old cache entries."""
    try:
        result = await diffusion_service.cleanup_cache(max_age_hours)
        
        return {
            'message': f"Cleaned up {result['deleted_entries']} cache entries",
            'deleted_entries': result['deleted_entries'],
            'max_age_hours': max_age_hours,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error cleaning up cache")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models():
    """Get list of available models."""
    try:
        models = {
            'text_to_image': [
                'runwayml/stable-diffusion-v1-5',
                'stabilityai/stable-diffusion-2-1',
                'CompVis/stable-diffusion-v1-4',
                'stabilityai/stable-diffusion-xl-base-1.0',
                'stabilityai/stable-diffusion-xl-refiner-1.0'
            ],
            'image_to_image': [
                'runwayml/stable-diffusion-v1-5',
                'stabilityai/stable-diffusion-2-1',
                'stabilityai/stable-diffusion-xl-base-1.0'
            ],
            'inpainting': [
                'runwayml/stable-diffusion-inpainting',
                'stabilityai/stable-diffusion-2-inpainting'
            ],
            'controlnet': [
                'canny',
                'depth',
                'pose',
                'segmentation',
                'openpose',
                'scribble',
                'softedge'
            ],
            'fast_generation': {
                'lcm': [
                    'SimianLuo/LCM_Dreamshaper_v7',
                    'SimianLuo/LCM_SDXL',
                    'latent-consistency/lcm-sdxl'
                ],
                'tcd': [
                    'h1t/TCD-SD15',
                    'h1t/TCD-SDXL'
                ]
            },
            'schedulers': [
                'DDIM',
                'PNDM',
                'Euler',
                'DPM++',
                'DPM++_SDE',
                'Heun',
                'KDPM2',
                'KDPM2_Ancestral',
                'UniPC',
                'LCM',
                'TCD'
            ]
        }
        
        return {
            'models': models,
            'loaded_models': list(diffusion_service.model_manager._pipelines.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error getting available models")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for diffusion service."""
    try:
        # Test basic functionality
        stats = await diffusion_service.get_generation_stats()
        
        return {
            'status': 'healthy',
            'service': 'diffusion',
            'loaded_models': stats['loaded_models'],
            'device': stats['device'],
            'cache_entries': stats['total_cache_entries'],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Diffusion service health check failed")
        raise HTTPException(status_code=503, detail=str(e))

@router.on_event("shutdown")
async def shutdown_diffusion_service():
    """Cleanup diffusion service on shutdown."""
    try:
        await diffusion_service.close()
        logger.info("Diffusion service shutdown complete")
    except Exception as e:
        logger.exception("Error during diffusion service shutdown") 