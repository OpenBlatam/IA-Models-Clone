"""
Diffusion Pipeline Wrapper Module

Wrapper for diffusion model pipelines.
"""

from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AudioLDMPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DiffusionPipelineWrapper:
    """
    Wrapper for diffusion model pipelines.
    
    Args:
        pipeline_type: Type of pipeline ("stable_diffusion", "stable_diffusion_xl", "audio_ldm").
        model_id: Model identifier.
        device: Device to use.
        torch_dtype: Data type for model.
    """
    
    def __init__(
        self,
        pipeline_type: str,
        model_id: str,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library required")
        
        self.pipeline_type = pipeline_type.lower()
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16 if self.device == "cuda" else torch.float32
        
        self._load_pipeline()
        logger.info(f"Loaded {pipeline_type} pipeline: {model_id}")
    
    def _load_pipeline(self):
        """Load diffusion pipeline."""
        if self.pipeline_type == "stable_diffusion":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            )
        elif self.pipeline_type == "stable_diffusion_xl":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            )
        elif self.pipeline_type == "audio_ldm":
            self.pipeline = AudioLDMPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            )
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
        
        self.pipeline = self.pipeline.to(self.device)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Any:
        """
        Generate samples from the pipeline.
        
        Args:
            prompt: Text prompt(s).
            num_inference_steps: Number of inference steps.
            guidance_scale: Guidance scale.
            **kwargs: Additional pipeline arguments.
        
        Returns:
            Generated samples.
        """
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )



