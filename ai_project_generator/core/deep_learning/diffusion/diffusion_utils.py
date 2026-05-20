"""
Diffusion Utilities - Advanced Diffusion Helpers
================================================

Utilities for working with Hugging Face Diffusers.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import diffusers
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DDIMScheduler,
        DDPMScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available")


def create_diffusion_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    use_xl: bool = False,
    scheduler_type: str = "ddim",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16
) -> Any:
    """
    Create diffusion pipeline with custom scheduler.
    
    Args:
        model_id: Hugging Face model ID
        use_xl: Use Stable Diffusion XL
        scheduler_type: Scheduler type ('ddim', 'ddpm', 'euler', 'dpm')
        device: Device to run on
        dtype: Data type
        
    Returns:
        Diffusion pipeline
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Diffusers library is required")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pipeline
    if use_xl:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
    
    # Set scheduler
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_type == "ddpm":
        scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    pipeline.scheduler = scheduler
    pipeline = pipeline.to(device)
    
    # Optimize for inference
    if hasattr(pipeline, 'enable_attention_slicing'):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, 'enable_model_cpu_offload'):
        pipeline.enable_model_cpu_offload()
    
    logger.info(f"Diffusion pipeline created: {model_id} with {scheduler_type} scheduler")
    return pipeline


class DiffusionPipelineWrapper:
    """
    Wrapper for diffusion pipelines with additional utilities.
    """
    
    def __init__(
        self,
        pipeline: Any,
        device: Optional[torch.device] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            pipeline: Diffusion pipeline
            device: Device to run on
        """
        self.pipeline = pipeline
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Generate images.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            seed: Random seed
            **kwargs: Additional pipeline arguments
            
        Returns:
            Generated images
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **kwargs
        ).images
        
        return images
    
    def change_scheduler(self, scheduler_type: str) -> None:
        """
        Change scheduler.
        
        Args:
            scheduler_type: New scheduler type
        """
        scheduler_map = {
            'ddim': DDIMScheduler,
            'ddpm': DDPMScheduler,
            'euler': EulerDiscreteScheduler,
            'dpm': DPMSolverMultistepScheduler
        }
        
        if scheduler_type not in scheduler_map:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        scheduler = scheduler_map[scheduler_type].from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = scheduler
        logger.info(f"Scheduler changed to: {scheduler_type}")



