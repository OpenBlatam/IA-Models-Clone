"""
Diffusion Pipeline Manager
===========================

Manages Stable Diffusion pipelines with proper GPU utilization.
"""

import logging
from typing import Any, Dict, Optional

import torch
from PIL import Image

# Third-party imports
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning(
        "Diffusers not available. Install with: pip install diffusers"
    )

from utils.device_manager import detect_device, get_torch_dtype, clear_cuda_cache

logger = logging.getLogger(__name__)


class DiffusionPipelineManager:
    """Manages Stable Diffusion pipelines with proper GPU utilization.
    
    Features:
    - Automatic device detection (CPU/GPU)
    - Mixed precision support
    - Memory-efficient attention
    - Scheduler configuration
    - Proper error handling
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize pipeline manager.
        
        Args:
            device: PyTorch device. Auto-detected if None.
        """
        self.device = detect_device(device)
        self.pipelines: Dict[str, Any] = {}
        self.torch_dtype = get_torch_dtype(self.device)
        self.logger = logging.getLogger(f"{__name__}.DiffusionPipelineManager")
        
    def load_pipeline(
        self,
        model_id: str,
        pipeline_type: str = "stable-diffusion-v1-5",
    ) -> None:
        """Load a diffusion pipeline.
        
        Args:
            model_id: HuggingFace model ID or local path
            pipeline_type: Type of pipeline (stable-diffusion-v1-5, stable-diffusion-xl)
        
        Raises:
            RuntimeError: If diffusers is not available or loading fails
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "Diffusers library not available. "
                "Install with: pip install diffusers"
            )
        
        try:
            self.logger.info(f"Loading pipeline: {model_id} on {self.device}")
            
            if pipeline_type == "stable-diffusion-xl":
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            
            # Move to device
            pipeline = pipeline.to(self.device)
            
            # Enable memory optimizations for better GPU utilization
            pipeline.enable_attention_slicing(slice_size="max")
            pipeline.enable_vae_slicing()
            
            # Enable CPU offload for memory-constrained systems
            if self.device.type == "cuda":
                try:
                    pipeline.enable_model_cpu_offload()
                    self.logger.info("CPU offload enabled for memory efficiency")
                except Exception as e:
                    self.logger.warning(f"CPU offload not available: {e}")
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    # Use mode="reduce-overhead" for better performance
                    pipeline.unet = torch.compile(
                        pipeline.unet,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    self.logger.info("Model compiled with torch.compile")
                except Exception as e:
                    self.logger.warning(f"torch.compile failed: {e}")
            
            self.pipelines[pipeline_type] = pipeline
            self.logger.info(f"Pipeline loaded successfully: {pipeline_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline {model_id}: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}") from e
    
    def get_pipeline(self, pipeline_type: str) -> Optional[Any]:
        """Get loaded pipeline by type."""
        return self.pipelines.get(pipeline_type)
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        config: Any = None,
        pipeline_type: str = "stable-diffusion-v1-5",
    ) -> Image.Image:
        """Generate image using diffusion pipeline.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            config: Generation configuration (must have get_inference_steps, guidance_scale, get_resolution_tuple, seed, scheduler)
            pipeline_type: Pipeline type to use
        
        Returns:
            Generated PIL Image
        
        Raises:
            RuntimeError: If pipeline not available or generation fails
        """
        if config is None:
            raise ValueError("Config is required for image generation")
        
        pipeline = self.get_pipeline(pipeline_type)
        if pipeline is None:
            raise RuntimeError(f"Pipeline {pipeline_type} not loaded")
        
        try:
            # Configure scheduler
            from .scheduler_factory import SchedulerFactory
            scheduler = SchedulerFactory.get_scheduler(config.scheduler)
            if scheduler:
                pipeline.scheduler = scheduler.from_config(
                    pipeline.scheduler.config
                )
            
            # Set up generator for reproducibility
            generator = None
            if hasattr(config, 'seed') and config.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(config.seed)
            
            # Generate image with proper error handling and mixed precision
            with torch.no_grad():
                # Use autocast for mixed precision inference on CUDA
                if self.device.type == "cuda" and self.torch_dtype == torch.float16:
                    with torch.cuda.amp.autocast():
                        result = pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=config.get_inference_steps(),
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                            height=config.get_resolution_tuple()[1],
                            width=config.get_resolution_tuple()[0],
                        )
                else:
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=config.get_inference_steps(),
                        guidance_scale=config.guidance_scale,
                        generator=generator,
                        height=config.get_resolution_tuple()[1],
                        width=config.get_resolution_tuple()[0],
                    )
            
            return result.images[0]
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                self.logger.error("GPU out of memory during generation")
                clear_cuda_cache()
                raise RuntimeError(
                    "GPU memory insufficient. Try reducing resolution or quality."
                ) from e
            raise
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("CUDA out of memory error")
            clear_cuda_cache()
            raise RuntimeError(
                "GPU memory insufficient. Try reducing resolution or quality."
            ) from e
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}") from e



