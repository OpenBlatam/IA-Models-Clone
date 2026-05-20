"""
Diffusion Service endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.diffusion_service import DiffusionService, DiffusionConfig

router = APIRouter()
diffusion_service = DiffusionService()


@router.post("/load-pipeline")
async def load_pipeline(
    model_id: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Dict[str, Any]:
    """Cargar pipeline de difusión"""
    try:
        config = DiffusionConfig(
            model_id=model_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        success = diffusion_service.load_pipeline(config)
        return {
            "model_id": model_id,
            "loaded": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_image(
    prompt: str,
    model_id: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generar imagen con modelo de difusión"""
    try:
        image = diffusion_service.generate_image(prompt, model_id, negative_prompt, seed=seed)
        return {
            "prompt": image.prompt,
            "generation_time": image.generation_time,
            "seed": image.seed,
            "image_size": len(image.image_data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




