"""
Diffusion Model - Diffusion Model Integration
==============================================

Integration with Hugging Face Diffusers library for:
- Stable Diffusion
- Stable Diffusion XL
- Custom diffusion pipelines
"""

import logging
from typing import Dict, Any, Optional, Union, List
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
        UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available. Install with: pip install diffusers")


class DiffusionModelWrapper:
    """
    Wrapper for diffusion models using Diffusers library.
    
    Provides a unified interface for text-to-image generation.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize diffusion model.
        
        Args:
            model_id: Hugging Face model ID
            use_xl: Use Stable Diffusion XL
            device: Device to run on
            dtype: Data type (float16 for efficiency)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required")
        
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Load pipeline
        if use_xl:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        
        self.pipeline = self.pipeline.to(self.device)
        
        # Optimize for inference
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, 'enable_model_cpu_offload'):
            self.pipeline.enable_model_cpu_offload()
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            seed: Random seed
            
        Returns:
            List of generated images
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
            generator=generator
        ).images
        
        return images
    
    def save_model(self, save_directory: Union[str, Path]) -> None:
        """
        Save model to directory.
        
        Args:
            save_directory: Directory to save model
        """
        self.pipeline.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")


def create_diffusion_model(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    use_xl: bool = False,
    device: Optional[torch.device] = None
) -> DiffusionModelWrapper:
    """
    Create a diffusion model wrapper.
    
    Args:
        model_id: Hugging Face model ID
        use_xl: Use Stable Diffusion XL
        device: Device to run on
        
    Returns:
        DiffusionModelWrapper instance
    """
    return DiffusionModelWrapper(
        model_id=model_id,
        use_xl=use_xl,
        device=device
    )



