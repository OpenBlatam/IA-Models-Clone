"""
Pipeline Orchestrator
=====================

Orchestrates pipeline operations for clothing change.
"""

import logging
from typing import Optional
from PIL import Image

try:
    from diffusers import FluxPipeline, FluxInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates pipeline operations."""
    
    def __init__(
        self,
        pipeline: FluxPipeline,
        use_inpainting: bool = True,
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            pipeline: Flux pipeline
            use_inpainting: Use inpainting mode
        """
        self.pipeline = pipeline
        self.use_inpainting = use_inpainting
    
    def generate(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Optional[Image.Image] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
    ) -> Image.Image:
        """
        Generate image using pipeline.
        
        Args:
            prompt: Generation prompt
            image: Input image
            mask_image: Optional mask for inpainting
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            
        Returns:
            Generated image
        """
        if self.use_inpainting and isinstance(self.pipeline, FluxInpaintPipeline):
            if mask_image is None:
                logger.warning("Inpainting pipeline requires mask, falling back to regular generation")
                return self._generate_regular(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            
            return self._generate_inpainting(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            )
        else:
            return self._generate_regular(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
    
    def _generate_inpainting(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
    ) -> Image.Image:
        """Generate using inpainting pipeline."""
        result = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]
        
        logger.debug("Image generated using inpainting pipeline")
        return result
    
    def _generate_regular(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        """Generate using regular pipeline."""
        result = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        logger.debug("Image generated using regular pipeline")
        return result


