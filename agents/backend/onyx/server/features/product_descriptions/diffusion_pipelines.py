from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from typing_extensions import TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from diffusers import (
from transformers import (
from torchvision import transforms
import psutil
import gc
from typing import Any, List, Dict, Optional
"""
Advanced Diffusion Pipelines Implementation
==========================================

This module provides comprehensive implementations of various diffusion pipelines,
including:

1. StableDiffusionPipeline - Standard Stable Diffusion
2. StableDiffusionXLPipeline - Stable Diffusion XL
3. StableDiffusionImg2ImgPipeline - Image-to-Image
4. StableDiffusionInpaintPipeline - Inpainting
5. StableDiffusionControlNetPipeline - ControlNet
6. Custom pipeline implementations

Features:
- Production-ready pipeline implementations
- Comprehensive error handling
- Performance optimizations
- Memory management
- Security considerations
- Extensive configuration options

Author: AI Assistant
License: MIT
"""



# Diffusers imports
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel,
    AutoencoderKL, UNet2DConditionModel, UNet2DModel,
    DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
    DDPMScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler, UniPCMultistepScheduler
)

# Transformers imports
    CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor,
    T5EncoderModel, T5Tokenizer, AutoTokenizer
)

# Image processing

# Configure logging
logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Available pipeline types."""
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    CUSTOM = "custom"


class SchedulerType(Enum):
    """Available scheduler types."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    HEUN = "heun"
    LMS = "lms"
    KDPM2 = "kdpm2"
    KDPM2_ANCESTRAL = "kdpm2_ancestral"
    UNIPC = "unipc"


@dataclass
class PipelineConfig:
    """Configuration for diffusion pipelines."""
    pipeline_type: PipelineType = PipelineType.STABLE_DIFFUSION
    model_id: str = "runwayml/stable-diffusion-v1-5"
    scheduler_type: SchedulerType = SchedulerType.DDIM
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    
    # Advanced parameters
    use_safetensors: bool = True
    use_attention_slicing: bool = True
    use_vae_slicing: bool = True
    use_vae_tiling: bool = False
    use_memory_efficient_attention: bool = True
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    
    # Safety and security
    safety_checker: bool = True
    requires_safety_checking: bool = True
    safety_checker_guidance_scale: float = 0.5
    
    # Custom model paths
    custom_model_path: Optional[str] = None
    custom_vae_path: Optional[str] = None
    custom_text_encoder_path: Optional[str] = None
    custom_unet_path: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for generation parameters."""
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    
    # Advanced generation parameters
    latents: Optional[torch.Tensor] = None
    output_type: str = "pil"  # "pil", "latent", "np"
    return_dict: bool = True
    callback: Optional[Callable] = None
    callback_steps: int = 1
    
    # ControlNet specific
    controlnet_conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    
    # Inpainting specific
    mask_image: Optional[Union[Image.Image, torch.Tensor]] = None
    image: Optional[Union[Image.Image, torch.Tensor]] = None
    
    # Img2Img specific
    strength: float = 0.8
    num_images_per_prompt: int = 1


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    images: List[Image.Image]
    latents: Optional[torch.Tensor] = None
    nsfw_content_detected: Optional[List[bool]] = None
    processing_time: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDiffusionPipeline(ABC):
    """Abstract base class for diffusion pipelines."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._detect_device()
        self.pipeline = None
        self.scheduler = None
        
        logger.info(f"Initializing {config.pipeline_type.value} pipeline")
        logger.info(f"Model ID: {config.model_id}")
        logger.info(f"Device: {self.device}")
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    @abstractmethod
    def _create_pipeline(self) -> Any:
        """Create the specific pipeline instance."""
        pass
    
    @abstractmethod
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations and settings."""
        pass
    
    def load_pipeline(self) -> Any:
        """Load and configure the pipeline."""
        start_time = time.time()
        
        try:
            # Create pipeline
            self.pipeline = self._create_pipeline()
            
            # Configure pipeline
            self._configure_pipeline()
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Set scheduler
            self._set_scheduler()
            
            loading_time = time.time() - start_time
            logger.info(f"Pipeline loaded successfully in {loading_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _set_scheduler(self) -> Any:
        """Set the scheduler for the pipeline."""
        schedulers = {
            SchedulerType.DDPM: DDPMScheduler,
            SchedulerType.DDIM: DDIMScheduler,
            SchedulerType.DPM_SOLVER: DPMSolverMultistepScheduler,
            SchedulerType.EULER: EulerDiscreteScheduler,
            SchedulerType.HEUN: HeunDiscreteScheduler,
            SchedulerType.LMS: LMSDiscreteScheduler,
            SchedulerType.KDPM2: KDPM2DiscreteScheduler,
            SchedulerType.KDPM2_ANCESTRAL: KDPM2AncestralDiscreteScheduler,
            SchedulerType.UNIPC: UniPCMultistepScheduler,
        }
        
        scheduler_class = schedulers.get(self.config.scheduler_type)
        if scheduler_class is None:
            raise ValueError(f"Unsupported scheduler type: {self.config.scheduler_type}")
        
        # Create scheduler with pipeline's scheduler config
        self.scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = self.scheduler
        
        logger.info(f"Scheduler set to: {self.config.scheduler_type.value}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    @abstractmethod
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using the pipeline."""
        pass
    
    def cleanup(self) -> Any:
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Pipeline resources cleaned up")


class StableDiffusionPipelineWrapper(BaseDiffusionPipeline):
    """Wrapper for StableDiffusionPipeline."""
    
    def _create_pipeline(self) -> StableDiffusionPipeline:
        """Create StableDiffusionPipeline instance."""
        return StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
    
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.use_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.use_vae_tiling:
            self.pipeline.enable_vae_tiling()
        
        if self.config.use_memory_efficient_attention:
            self.pipeline.enable_model_cpu_offload()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("StableDiffusionPipeline configured with optimizations")
    
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using StableDiffusionPipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Generate images
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                height=config.height,
                width=config.width,
                num_images_per_prompt=config.num_images_per_prompt,
                latents=config.latents,
                output_type=config.output_type,
                return_dict=config.return_dict,
                callback=config.callback,
                callback_steps=config.callback_steps,
            )
            
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return PipelineResult(
                images=result.images,
                nsfw_content_detected=result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else None,
                processing_time=processing_time,
                memory_usage=memory_used,
                metadata={
                    "pipeline_type": self.config.pipeline_type.value,
                    "model_id": self.config.model_id,
                    "scheduler_type": self.config.scheduler_type.value,
                    "guidance_scale": config.guidance_scale,
                    "num_steps": config.num_inference_steps,
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class StableDiffusionXLPipelineWrapper(BaseDiffusionPipeline):
    """Wrapper for StableDiffusionXLPipeline."""
    
    def _create_pipeline(self) -> StableDiffusionXLPipeline:
        """Create StableDiffusionXLPipeline instance."""
        return StableDiffusionXLPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
    
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.use_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.use_vae_tiling:
            self.pipeline.enable_vae_tiling()
        
        if self.config.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        
        if self.config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("StableDiffusionXLPipeline configured with optimizations")
    
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using StableDiffusionXLPipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Generate images
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                height=config.height,
                width=config.width,
                num_images_per_prompt=config.num_images_per_prompt,
                latents=config.latents,
                output_type=config.output_type,
                return_dict=config.return_dict,
                callback=config.callback,
                callback_steps=config.callback_steps,
            )
            
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return PipelineResult(
                images=result.images,
                nsfw_content_detected=result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else None,
                processing_time=processing_time,
                memory_usage=memory_used,
                metadata={
                    "pipeline_type": self.config.pipeline_type.value,
                    "model_id": self.config.model_id,
                    "scheduler_type": self.config.scheduler_type.value,
                    "guidance_scale": config.guidance_scale,
                    "num_steps": config.num_inference_steps,
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class StableDiffusionImg2ImgPipelineWrapper(BaseDiffusionPipeline):
    """Wrapper for StableDiffusionImg2ImgPipeline."""
    
    def _create_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """Create StableDiffusionImg2ImgPipeline instance."""
        return StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
    
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.use_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("StableDiffusionImg2ImgPipeline configured with optimizations")
    
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using StableDiffusionImg2ImgPipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        if config.image is None:
            raise ValueError("Image is required for img2img generation")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Generate images
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=config.image,
                strength=config.strength,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                num_images_per_prompt=config.num_images_per_prompt,
                latents=config.latents,
                output_type=config.output_type,
                return_dict=config.return_dict,
                callback=config.callback,
                callback_steps=config.callback_steps,
            )
            
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return PipelineResult(
                images=result.images,
                nsfw_content_detected=result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else None,
                processing_time=processing_time,
                memory_usage=memory_used,
                metadata={
                    "pipeline_type": self.config.pipeline_type.value,
                    "model_id": self.config.model_id,
                    "scheduler_type": self.config.scheduler_type.value,
                    "guidance_scale": config.guidance_scale,
                    "num_steps": config.num_inference_steps,
                    "strength": config.strength,
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class StableDiffusionInpaintPipelineWrapper(BaseDiffusionPipeline):
    """Wrapper for StableDiffusionInpaintPipeline."""
    
    def _create_pipeline(self) -> StableDiffusionInpaintPipeline:
        """Create StableDiffusionInpaintPipeline instance."""
        return StableDiffusionInpaintPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
    
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.use_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("StableDiffusionInpaintPipeline configured with optimizations")
    
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using StableDiffusionInpaintPipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        if config.image is None:
            raise ValueError("Image is required for inpainting")
        
        if config.mask_image is None:
            raise ValueError("Mask image is required for inpainting")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Generate images
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=config.image,
                mask_image=config.mask_image,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                num_images_per_prompt=config.num_images_per_prompt,
                latents=config.latents,
                output_type=config.output_type,
                return_dict=config.return_dict,
                callback=config.callback,
                callback_steps=config.callback_steps,
            )
            
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return PipelineResult(
                images=result.images,
                nsfw_content_detected=result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else None,
                processing_time=processing_time,
                memory_usage=memory_used,
                metadata={
                    "pipeline_type": self.config.pipeline_type.value,
                    "model_id": self.config.model_id,
                    "scheduler_type": self.config.scheduler_type.value,
                    "guidance_scale": config.guidance_scale,
                    "num_steps": config.num_inference_steps,
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class StableDiffusionControlNetPipelineWrapper(BaseDiffusionPipeline):
    """Wrapper for StableDiffusionControlNetPipeline."""
    
    def __init__(self, config: PipelineConfig, controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"):
        
    """__init__ function."""
super().__init__(config)
        self.controlnet_model_id = controlnet_model_id
        self.controlnet = None
    
    def _create_pipeline(self) -> StableDiffusionControlNetPipeline:
        """Create StableDiffusionControlNetPipeline instance."""
        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=self.config.torch_dtype,
        )
        
        return StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=self.controlnet,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
    
    def _configure_pipeline(self) -> Any:
        """Configure the pipeline with optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.use_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("StableDiffusionControlNetPipeline configured with optimizations")
    
    def generate(self, config: GenerationConfig) -> PipelineResult:
        """Generate images using StableDiffusionControlNetPipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Generate images
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                height=config.height,
                width=config.width,
                num_images_per_prompt=config.num_images_per_prompt,
                latents=config.latents,
                output_type=config.output_type,
                return_dict=config.return_dict,
                callback=config.callback,
                callback_steps=config.callback_steps,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                control_guidance_start=config.control_guidance_start,
                control_guidance_end=config.control_guidance_end,
            )
            
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            return PipelineResult(
                images=result.images,
                nsfw_content_detected=result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else None,
                processing_time=processing_time,
                memory_usage=memory_used,
                metadata={
                    "pipeline_type": self.config.pipeline_type.value,
                    "model_id": self.config.model_id,
                    "controlnet_model_id": self.controlnet_model_id,
                    "scheduler_type": self.config.scheduler_type.value,
                    "guidance_scale": config.guidance_scale,
                    "num_steps": config.num_inference_steps,
                    "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class DiffusionPipelineFactory:
    """Factory for creating diffusion pipelines."""
    
    @staticmethod
    def create(config: PipelineConfig, **kwargs) -> BaseDiffusionPipeline:
        """Create a diffusion pipeline based on configuration."""
        pipelines = {
            PipelineType.STABLE_DIFFUSION: StableDiffusionPipelineWrapper,
            PipelineType.STABLE_DIFFUSION_XL: StableDiffusionXLPipelineWrapper,
            PipelineType.IMG2IMG: StableDiffusionImg2ImgPipelineWrapper,
            PipelineType.INPAINT: StableDiffusionInpaintPipelineWrapper,
            PipelineType.CONTROLNET: StableDiffusionControlNetPipelineWrapper,
        }
        
        pipeline_class = pipelines.get(config.pipeline_type)
        if pipeline_class is None:
            raise ValueError(f"Unsupported pipeline type: {config.pipeline_type}")
        
        return pipeline_class(config, **kwargs)


class AdvancedPipelineManager:
    """
    Advanced pipeline manager for handling multiple pipelines.
    
    This class provides a unified interface for managing different types
    of diffusion pipelines, with support for:
    - Multiple pipeline types
    - Performance monitoring
    - Memory management
    - Batch processing
    - Pipeline switching
    """
    
    def __init__(self) -> Any:
        self.pipelines: Dict[str, BaseDiffusionPipeline] = {}
        self.active_pipeline: Optional[str] = None
        
        logger.info("AdvancedPipelineManager initialized")
    
    def add_pipeline(self, name: str, config: PipelineConfig, **kwargs) -> str:
        """Add a new pipeline to the manager."""
        try:
            pipeline = DiffusionPipelineFactory.create(config, **kwargs)
            pipeline.load_pipeline()
            
            self.pipelines[name] = pipeline
            if self.active_pipeline is None:
                self.active_pipeline = name
            
            logger.info(f"Pipeline '{name}' added successfully")
            return name
            
        except Exception as e:
            logger.error(f"Failed to add pipeline '{name}': {e}")
            raise
    
    def remove_pipeline(self, name: str):
        """Remove a pipeline from the manager."""
        if name in self.pipelines:
            self.pipelines[name].cleanup()
            del self.pipelines[name]
            
            if self.active_pipeline == name:
                self.active_pipeline = list(self.pipelines.keys())[0] if self.pipelines else None
            
            logger.info(f"Pipeline '{name}' removed successfully")
        else:
            logger.warning(f"Pipeline '{name}' not found")
    
    def set_active_pipeline(self, name: str):
        """Set the active pipeline."""
        if name in self.pipelines:
            self.active_pipeline = name
            logger.info(f"Active pipeline set to '{name}'")
        else:
            raise ValueError(f"Pipeline '{name}' not found")
    
    def get_active_pipeline(self) -> Optional[BaseDiffusionPipeline]:
        """Get the active pipeline."""
        if self.active_pipeline and self.active_pipeline in self.pipelines:
            return self.pipelines[self.active_pipeline]
        return None
    
    def generate(self, config: GenerationConfig, pipeline_name: Optional[str] = None) -> PipelineResult:
        """Generate images using the specified or active pipeline."""
        pipeline_name = pipeline_name or self.active_pipeline
        
        if pipeline_name is None:
            raise RuntimeError("No active pipeline set")
        
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        pipeline = self.pipelines[pipeline_name]
        return pipeline.generate(config)
    
    def get_pipeline_info(self) -> Dict[str, Dict]:
        """Get information about all pipelines."""
        info = {}
        for name, pipeline in self.pipelines.items():
            info[name] = {
                "type": pipeline.config.pipeline_type.value,
                "model_id": pipeline.config.model_id,
                "scheduler_type": pipeline.config.scheduler_type.value,
                "device": str(pipeline.device),
                "is_active": name == self.active_pipeline,
            }
        return info
    
    def cleanup_all(self) -> Any:
        """Clean up all pipelines."""
        for name, pipeline in self.pipelines.items():
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup pipeline '{name}': {e}")
        
        self.pipelines.clear()
        self.active_pipeline = None
        logger.info("All pipelines cleaned up")


# Utility functions
def create_pipeline(pipeline_type: PipelineType, model_id: str, **kwargs) -> BaseDiffusionPipeline:
    """Create a pipeline with default parameters."""
    config = PipelineConfig(pipeline_type=pipeline_type, model_id=model_id, **kwargs)
    return DiffusionPipelineFactory.create(config)


def create_pipeline_manager() -> AdvancedPipelineManager:
    """Create an advanced pipeline manager."""
    return AdvancedPipelineManager()


def get_available_pipeline_types() -> List[PipelineType]:
    """Get list of available pipeline types."""
    return list(PipelineType)


def get_available_scheduler_types() -> List[SchedulerType]:
    """Get list of available scheduler types."""
    return list(SchedulerType) 