from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from typing_extensions import TypedDict
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from diffusers import (
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
from torchvision import transforms
import psutil
import gc
from typing import Any, List, Dict, Optional
"""
Diffusion Models for Cybersecurity Applications
==============================================

This module provides comprehensive diffusion model capabilities for cybersecurity
applications, including text-to-image generation, image analysis, security
visualization, and threat pattern recognition.

Features:
- Text-to-image generation for security reports
- Image-to-image transformation for threat analysis
- Inpainting for data reconstruction
- ControlNet integration for guided generation
- Security-focused prompt engineering
- Batch processing and optimization
- Memory-efficient inference
- Integration with existing transformers infrastructure

Author: AI Assistant
License: MIT
"""



# Diffusion models
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
    AutoencoderKL, UNet2DConditionModel
)

# Transformers for text processing

# Image processing

# Configure logging
logger = logging.getLogger(__name__)
diffusers_logging.set_verbosity_error()


class DiffusionTask(Enum):
    """Supported diffusion model tasks."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    CONTROLNET = "controlnet"
    IMAGE_ANALYSIS = "image_analysis"
    SECURITY_VISUALIZATION = "security_visualization"


class SchedulerType(Enum):
    """Available diffusion schedulers."""
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    HEUN = "heun"
    LMS = "lms"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model operations."""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    task: DiffusionTask = DiffusionTask.TEXT_TO_IMAGE
    scheduler: SchedulerType = SchedulerType.DPM_SOLVER
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    use_safety_checker: bool = True
    use_attention_slicing: bool = True
    use_memory_efficient_attention: bool = True
    enable_model_cpu_offload: bool = True
    enable_sequential_cpu_offload: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    revision: Optional[str] = None
    variant: Optional[str] = None
    low_cpu_mem_usage: bool = True


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    eta: float = 0.0
    output_type: str = "pil"
    return_dict: bool = True
    callback: Optional[Callable] = None
    callback_steps: int = 1


@dataclass
class ImageToImageConfig(GenerationConfig):
    """Configuration for image-to-image generation."""
    strength: float = 0.8
    image: Optional[Union[Image.Image, str]] = None


@dataclass
class InpaintingConfig(GenerationConfig):
    """Configuration for inpainting."""
    mask_image: Optional[Union[Image.Image, str]] = None
    mask_strength: float = 0.8


@dataclass
class ControlNetConfig(GenerationConfig):
    """Configuration for ControlNet."""
    control_image: Optional[Union[Image.Image, str]] = None
    controlnet_conditioning_scale: float = 1.0


@dataclass
class DiffusionResult:
    """Result of diffusion model operation."""
    images: List[Image.Image]
    nsfw_content_detected: List[bool] = field(default_factory=list)
    processing_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityVisualizationConfig:
    """Configuration for security visualization."""
    threat_type: str = "malware"
    severity: str = "high"
    visualization_style: str = "technical"
    include_metrics: bool = True
    include_timeline: bool = False
    color_scheme: str = "red_alert"


class DiffusionMetrics(TypedDict):
    """Metrics for diffusion model performance."""
    generation_time: float
    memory_usage: float
    throughput: float
    error_count: int
    success_count: int
    safety_violations: int


class SecurityPromptEngine:
    """Engine for generating security-focused prompts."""
    
    SECURITY_PROMPTS = {
        "malware_analysis": {
            "positive": "technical diagram of malware analysis, cybersecurity visualization, network security, digital threat detection, professional security report, clean technical illustration",
            "negative": "cartoon, anime, artistic, decorative, colorful, playful, child-like, fantasy"
        },
        "network_security": {
            "positive": "network security diagram, cybersecurity infrastructure, firewall visualization, intrusion detection system, professional technical diagram, clean network topology",
            "negative": "artistic, decorative, colorful, cartoon, anime, fantasy, child-like"
        },
        "threat_hunting": {
            "positive": "threat hunting visualization, cybersecurity investigation, digital forensics, security analysis, professional technical diagram, clean security workflow",
            "negative": "artistic, decorative, colorful, cartoon, anime, fantasy, child-like"
        },
        "incident_response": {
            "positive": "incident response workflow, cybersecurity incident, security operations center, professional technical diagram, clean security process",
            "negative": "artistic, decorative, colorful, cartoon, anime, fantasy, child-like"
        }
    }
    
    @classmethod
    def generate_security_prompt(
        cls,
        threat_type: str,
        severity: str = "medium",
        style: str = "technical"
    ) -> Tuple[str, str]:
        """Generate security-focused prompts."""
        base_prompts = cls.SECURITY_PROMPTS.get(threat_type, cls.SECURITY_PROMPTS["malware_analysis"])
        
        # Enhance based on severity
        severity_enhancements = {
            "low": "low priority, minor threat",
            "medium": "moderate threat level",
            "high": "high priority, critical threat, urgent",
            "critical": "critical security threat, emergency response, highest priority"
        }
        
        # Style enhancements
        style_enhancements = {
            "technical": "technical diagram, professional, clean, minimal",
            "detailed": "detailed technical diagram, comprehensive, thorough",
            "simple": "simple diagram, basic, clear, easy to understand"
        }
        
        positive_prompt = f"{base_prompts['positive']}, {severity_enhancements.get(severity, '')}, {style_enhancements.get(style, '')}"
        negative_prompt = base_prompts['negative']
        
        return positive_prompt, negative_prompt


class DiffusionModelsManager:
    """
    Comprehensive manager for diffusion models in cybersecurity applications.
    
    Provides text-to-image generation, image analysis, security visualization,
    and integration with existing transformers infrastructure.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Diffusion Models Manager.
        
        Args:
            cache_dir: Directory for caching models
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "diffusers")
        self._pipelines: Dict[str, Any] = {}
        self._configs: Dict[str, DiffusionConfig] = {}
        self._metrics: Dict[str, DiffusionMetrics] = {}
        self._device = self._detect_device()
        self._lock = asyncio.Lock()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"DiffusionModelsManager initialized with device: {self._device}")
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device for model execution."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def load_pipeline(
        self,
        config: DiffusionConfig,
        force_reload: bool = False
    ) -> Any:
        """
        Load a diffusion pipeline with optimization.
        
        Args:
            config: Diffusion configuration
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded pipeline
        """
        pipeline_key = f"{config.model_name}_{config.task.value}"
        
        async with self._lock:
            if pipeline_key in self._pipelines and not force_reload:
                logger.info(f"Pipeline {pipeline_key} already loaded, returning cached version")
                return self._pipelines[pipeline_key]
            
            start_time = time.time()
            
            try:
                logger.info(f"Loading diffusion pipeline: {config.model_name} ({config.task.value})")
                
                # Determine device
                device = self._get_device(config.device)
                
                # Load scheduler
                scheduler = self._get_scheduler(config.scheduler)
                
                # Pipeline loading parameters
                pipeline_kwargs = {
                    "torch_dtype": config.torch_dtype,
                    "cache_dir": config.cache_dir or self.cache_dir,
                    "local_files_only": config.local_files_only,
                    "low_cpu_mem_usage": config.low_cpu_mem_usage,
                }
                
                if config.revision:
                    pipeline_kwargs["revision"] = config.revision
                
                if config.variant:
                    pipeline_kwargs["variant"] = config.variant
                
                # Load pipeline based on task
                pipeline = await self._load_pipeline_by_task(config, scheduler, pipeline_kwargs)
                
                # Apply optimizations
                pipeline = await self._apply_optimizations(pipeline, config)
                
                # Move to device
                pipeline = pipeline.to(device)
                
                # Cache the pipeline
                self._pipelines[pipeline_key] = pipeline
                self._configs[pipeline_key] = config
                
                # Update metrics
                load_time = time.time() - start_time
                self._metrics[pipeline_key] = {
                    "generation_time": 0.0,
                    "memory_usage": self._get_memory_usage(),
                    "throughput": 0.0,
                    "error_count": 0,
                    "success_count": 0,
                    "safety_violations": 0
                }
                
                logger.info(f"Pipeline {pipeline_key} loaded successfully in {load_time:.2f}s")
                return pipeline
                
            except Exception as e:
                logger.error(f"Failed to load pipeline {config.model_name}: {str(e)}")
                raise
    
    async def _load_pipeline_by_task(
        self,
        config: DiffusionConfig,
        scheduler: Any,
        pipeline_kwargs: Dict[str, Any]
    ) -> Any:
        """Load pipeline based on task type."""
        loop = asyncio.get_event_loop()
        
        if config.task == DiffusionTask.TEXT_TO_IMAGE:
            return await loop.run_in_executor(
                None,
                lambda: StableDiffusionPipeline.from_pretrained(
                    config.model_name,
                    scheduler=scheduler,
                    **pipeline_kwargs
                )
            )
        
        elif config.task == DiffusionTask.IMAGE_TO_IMAGE:
            return await loop.run_in_executor(
                None,
                lambda: StableDiffusionImg2ImgPipeline.from_pretrained(
                    config.model_name,
                    scheduler=scheduler,
                    **pipeline_kwargs
                )
            )
        
        elif config.task == DiffusionTask.INPAINTING:
            return await loop.run_in_executor(
                None,
                lambda: StableDiffusionInpaintPipeline.from_pretrained(
                    config.model_name,
                    scheduler=scheduler,
                    **pipeline_kwargs
                )
            )
        
        elif config.task == DiffusionTask.CONTROLNET:
            # Load ControlNet model
            controlnet_model = await loop.run_in_executor(
                None,
                lambda: ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=config.torch_dtype
                )
            )
            
            return await loop.run_in_executor(
                None,
                lambda: StableDiffusionControlNetPipeline.from_pretrained(
                    config.model_name,
                    controlnet=controlnet_model,
                    scheduler=scheduler,
                    **pipeline_kwargs
                )
            )
        
        else:
            raise ValueError(f"Unsupported task: {config.task}")
    
    async def _apply_optimizations(self, pipeline: Any, config: DiffusionConfig) -> Any:
        """Apply performance optimizations to the pipeline."""
        try:
            if config.use_attention_slicing:
                pipeline.enable_attention_slicing()
            
            if config.use_memory_efficient_attention and hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
            
            if config.enable_model_cpu_offload:
                pipeline.enable_model_cpu_offload()
            
            if config.enable_sequential_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            
            if config.enable_xformers_memory_efficient_attention:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    logger.warning("xformers not available, skipping xformers optimization")
            
            return pipeline
            
        except Exception as e:
            logger.warning(f"Some optimizations failed: {str(e)}")
            return pipeline
    
    def _get_scheduler(self, scheduler_type: SchedulerType) -> Optional[Dict[str, Any]]:
        """Get the appropriate scheduler."""
        scheduler_map = {
            SchedulerType.DDIM: DDIMScheduler,
            SchedulerType.DPM_SOLVER: DPMSolverMultistepScheduler,
            SchedulerType.EULER: EulerDiscreteScheduler,
        }
        
        scheduler_class = scheduler_map.get(scheduler_type, DPMSolverMultistepScheduler)
        return scheduler_class.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    
    def _get_device(self, device_type: str) -> torch.device:
        """Get the appropriate device for model execution."""
        if device_type == "auto":
            return self._device
        elif device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def generate_image(
        self,
        pipeline_key: str,
        config: GenerationConfig
    ) -> DiffusionResult:
        """
        Generate images using the specified pipeline.
        
        Args:
            pipeline_key: Key of the loaded pipeline
            config: Generation configuration
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        if pipeline_key not in self._pipelines:
            raise ValueError(f"Pipeline {pipeline_key} not loaded")
        
        pipeline = self._pipelines[pipeline_key]
        diffusion_config = self._configs[pipeline_key]
        
        try:
            # Set seed if provided
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(config.seed)
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "width": config.width,
                "height": config.height,
                "num_images_per_prompt": config.num_images_per_prompt,
                "eta": config.eta,
                "output_type": config.output_type,
                "return_dict": config.return_dict,
                "callback": config.callback,
                "callback_steps": config.callback_steps,
            }
            
            # Run generation
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: pipeline(**generation_kwargs)
            )
            
            # Process outputs
            images = outputs.images if hasattr(outputs, 'images') else [outputs]
            nsfw_content_detected = outputs.nsfw_content_detected if hasattr(outputs, 'nsfw_content_detected') else [False] * len(images)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            if pipeline_key in self._metrics:
                self._metrics[pipeline_key]["generation_time"] += processing_time
                self._metrics[pipeline_key]["success_count"] += 1
                self._metrics[pipeline_key]["throughput"] = (
                    self._metrics[pipeline_key]["success_count"] / 
                    self._metrics[pipeline_key]["generation_time"]
                )
                self._metrics[pipeline_key]["safety_violations"] += sum(nsfw_content_detected)
            
            return DiffusionResult(
                images=images,
                nsfw_content_detected=nsfw_content_detected,
                processing_time=processing_time,
                memory_usage=self._get_memory_usage(),
                metadata={
                    "pipeline_key": pipeline_key,
                    "config": config.__dict__,
                    "diffusion_config": diffusion_config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation failed for {pipeline_key}: {str(e)}")
            
            # Update error metrics
            if pipeline_key in self._metrics:
                self._metrics[pipeline_key]["error_count"] += 1
            
            raise
    
    async def generate_image_to_image(
        self,
        pipeline_key: str,
        config: ImageToImageConfig
    ) -> DiffusionResult:
        """
        Generate images using image-to-image pipeline.
        
        Args:
            pipeline_key: Key of the loaded pipeline
            config: Image-to-image configuration
            
        Returns:
            Generation result
        """
        if pipeline_key not in self._pipelines:
            raise ValueError(f"Pipeline {pipeline_key} not loaded")
        
        pipeline = self._pipelines[pipeline_key]
        
        # Load image if provided as string
        if isinstance(config.image, str):
            if config.image.startswith(('http://', 'https://')):
                response = requests.get(config.image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            else:
                image = Image.open(config.image).convert("RGB")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        else:
            image = config.image
        
        if image is None:
            raise ValueError("Image is required for image-to-image generation")
        
        # Resize image to match generation dimensions
        image = image.resize((config.width, config.height))
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "image": image,
            "strength": config.strength,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "num_images_per_prompt": config.num_images_per_prompt,
            "eta": config.eta,
            "output_type": config.output_type,
            "return_dict": config.return_dict,
        }
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: pipeline(**generation_kwargs)
            )
            
            images = outputs.images if hasattr(outputs, 'images') else [outputs]
            nsfw_content_detected = outputs.nsfw_content_detected if hasattr(outputs, 'nsfw_content_detected') else [False] * len(images)
            
            processing_time = time.time() - start_time
            
            return DiffusionResult(
                images=images,
                nsfw_content_detected=nsfw_content_detected,
                processing_time=processing_time,
                memory_usage=self._get_memory_usage(),
                metadata={
                    "pipeline_key": pipeline_key,
                    "config": config.__dict__,
                    "task": "image_to_image"
                }
            )
            
        except Exception as e:
            logger.error(f"Image-to-image generation failed: {str(e)}")
            raise
    
    async def generate_security_visualization(
        self,
        threat_type: str,
        severity: str = "medium",
        style: str = "technical",
        config: Optional[GenerationConfig] = None
    ) -> DiffusionResult:
        """
        Generate security-focused visualizations.
        
        Args:
            threat_type: Type of security threat
            severity: Threat severity level
            style: Visualization style
            config: Optional generation configuration
            
        Returns:
            Generation result
        """
        # Generate security prompts
        positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
            threat_type, severity, style
        )
        
        # Use default config if not provided
        if config is None:
            config = GenerationConfig(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=8.0,
                width=768,
                height=512
            )
        else:
            config.prompt = positive_prompt
            config.negative_prompt = negative_prompt
        
        # Load text-to-image pipeline
        diffusion_config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        pipeline_key = f"{diffusion_config.model_name}_{diffusion_config.task.value}"
        
        # Ensure pipeline is loaded
        await self.load_pipeline(diffusion_config)
        
        # Generate image
        return await self.generate_image(pipeline_key, config)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    def get_metrics(self, pipeline_key: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for pipelines."""
        if pipeline_key:
            return self._metrics.get(pipeline_key, {})
        return self._metrics
    
    def clear_cache(self, pipeline_key: Optional[str] = None):
        """Clear pipeline cache to free memory."""
        if pipeline_key:
            if pipeline_key in self._pipelines:
                del self._pipelines[pipeline_key]
            if pipeline_key in self._configs:
                del self._configs[pipeline_key]
            if pipeline_key in self._metrics:
                del self._metrics[pipeline_key]
        else:
            self._pipelines.clear()
            self._configs.clear()
            self._metrics.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Pipeline cache cleared")
    
    @asynccontextmanager
    async def pipeline_context(self, config: DiffusionConfig):
        """Context manager for pipeline loading and cleanup."""
        pipeline = None
        try:
            pipeline = await self.load_pipeline(config)
            yield pipeline
        finally:
            if pipeline is not None:
                # Optionally clear cache after use
                pass
    
    def list_loaded_pipelines(self) -> List[str]:
        """List all currently loaded pipelines."""
        return list(self._pipelines.keys())


# Global instance for easy access
diffusion_manager = DiffusionModelsManager() 