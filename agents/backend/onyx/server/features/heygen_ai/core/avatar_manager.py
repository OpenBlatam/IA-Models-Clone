#!/usr/bin/env python3
"""
Avatar Manager for HeyGen AI
============================

Production-ready avatar generation system using Stable Diffusion.
Follows best practices for deep learning and diffusion models.

Key Features:
- Stable Diffusion v1.5 and XL pipeline support
- Mixed precision training/inference
- Proper GPU utilization
- Error handling and logging
- Modular architecture
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Third-party imports with proper error handling
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
    )
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning(
        "Diffusers not available. Install with: pip install diffusers"
    )

try:
    import mediapipe as mp
    FACE_LIBS_AVAILABLE = True
except ImportError:
    FACE_LIBS_AVAILABLE = False
    logging.warning(
        "MediaPipe not available. Install with: pip install mediapipe"
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Imports from shared module
# =============================================================================

from shared import (
    AvatarStyle,
    AvatarQuality,
    Resolution,
    AvatarGenerationConfig,
    AvatarModel,
)


# =============================================================================
# Imports from modular components
# =============================================================================

from core.diffusion import DiffusionPipelineManager
from core.face_processing import FaceProcessingService
from core.image_processing import ImageProcessor, PromptEnhancer

# =============================================================================
# Imports from utility helpers
# =============================================================================

from utils.gpu_error_handler import handle_gpu_errors

# =============================================================================
# Legacy Diffusion Pipeline Manager (deprecated - use core.diffusion)
# =============================================================================

class _LegacyDiffusionPipelineManager:
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
        self.device = device or self._detect_device()
        self.pipelines: Dict[str, Any] = {}
        self.torch_dtype = self._get_torch_dtype()
        self.logger = logging.getLogger(f"{__name__}.DiffusionPipelineManager")
        
    def _detect_device(self) -> torch.device:
        """Detect and return appropriate device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on device."""
        if self.device.type == "cuda":
            return torch.float16  # Use FP16 on CUDA
        elif self.device.type == "mps":
            return torch.float32  # MPS doesn't support FP16
        else:
            return torch.float32  # Use FP32 on CPU
    
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
        config: AvatarGenerationConfig = None,
        pipeline_type: str = "stable-diffusion-v1-5",
    ) -> Image.Image:
        """Generate image using diffusion pipeline.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            config: Generation configuration
            pipeline_type: Pipeline type to use
        
        Returns:
            Generated PIL Image
        
        Raises:
            RuntimeError: If pipeline not available or generation fails
        """
        if config is None:
            config = AvatarGenerationConfig()
        
        pipeline = self.get_pipeline(pipeline_type)
        if pipeline is None:
            raise RuntimeError(f"Pipeline {pipeline_type} not loaded")
        
        try:
            # Configure scheduler
            scheduler = self._get_scheduler(config.scheduler)
            if scheduler:
                pipeline.scheduler = scheduler.from_config(
                    pipeline.scheduler.config
                )
            
            # Set up generator for reproducibility
            generator = None
            if config.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(config.seed)
            
            # Generate image with proper error handling and mixed precision
            def _generate_with_pipeline():
                with torch.no_grad():
                    # Use autocast for mixed precision inference on CUDA
                    if self.device.type == "cuda" and self.torch_dtype == torch.float16:
                        with torch.cuda.amp.autocast():
                            return pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=config.get_inference_steps(),
                                guidance_scale=config.guidance_scale,
                                generator=generator,
                                height=config.get_resolution_tuple()[1],
                                width=config.get_resolution_tuple()[0],
                            )
                    else:
                        return pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=config.get_inference_steps(),
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                            height=config.get_resolution_tuple()[1],
                            width=config.get_resolution_tuple()[0],
                        )
            
            # Use helper function for GPU error handling
            result = handle_gpu_errors(
                _generate_with_pipeline,
                operation_name="Image generation"
            )
            
            return result.images[0]
    
    @staticmethod
    def _get_scheduler(scheduler_type: str):
        """Get scheduler class by type."""
        if not DIFFUSERS_AVAILABLE:
            return None
        
        scheduler_map = {
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
        }
        return scheduler_map.get(scheduler_type.lower())


# =============================================================================
# Legacy Face Processing Service (deprecated - use core.face_processing)
# =============================================================================

class _LegacyFaceProcessingService:
    """Service for face detection and enhancement using MediaPipe.
    
    Features:
    - Face detection
    - Face mesh for detailed landmarks
    - Face enhancement
    """
    
    def __init__(self):
        """Initialize face processing service."""
        self.face_detection = None
        self.face_mesh = None
        self.logger = logging.getLogger(f"{__name__}.FaceProcessingService")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MediaPipe components."""
        if not FACE_LIBS_AVAILABLE:
            self.logger.warning("MediaPipe not available")
            return
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            
            self.logger.info("Face processing service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face processing: {e}")
            raise
    
    def enhance_face(self, image: np.ndarray) -> np.ndarray:
        """Enhance facial features in image.
        
        Args:
            image: Input image as numpy array (RGB)
        
        Returns:
            Enhanced image
        """
        if not self.face_detection:
            return image
        
        try:
            # Convert RGB to BGR for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Enhance face region
                    face_region = image[y:y+height, x:x+width]
                    enhanced_face = self._enhance_face_region(face_region)
                    image[y:y+height, x:x+width] = enhanced_face
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Face enhancement failed: {e}")
            return image
    
    @staticmethod
    def _enhance_face_region(face_region: np.ndarray) -> np.ndarray:
        """Enhance a specific face region."""
        try:
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(face_region, -1, kernel)
            
            # Apply subtle color enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Face region enhancement failed: {e}")
            return face_region


# =============================================================================
# Legacy Image Processor (deprecated - use core.image_processing)
# =============================================================================

class _LegacyImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def enhance_prompt(prompt: str, style: AvatarStyle) -> str:
        """Enhance prompt based on style.
        
        Args:
            prompt: Base prompt
            style: Avatar style
        
        Returns:
            Enhanced prompt
        """
        style_enhancements = {
            AvatarStyle.REALISTIC: (
                "professional headshot, high quality, detailed, "
                "8k, photorealistic, sharp focus"
            ),
            AvatarStyle.CARTOON: (
                "cartoon style, vibrant colors, clean lines, "
                "professional illustration"
            ),
            AvatarStyle.ANIME: (
                "anime style, detailed, professional, high quality, "
                "manga art style"
            ),
            AvatarStyle.ARTISTIC: (
                "artistic portrait, creative, high quality, "
                "detailed, unique style"
            ),
        }
        
        enhancement = style_enhancements.get(style, style_enhancements[AvatarStyle.REALISTIC])
        return f"{prompt}, {enhancement}"
    
    @staticmethod
    def post_process_image(
        image: Image.Image,
        config: AvatarGenerationConfig,
        face_service: Optional[FaceProcessingService] = None,
    ) -> Image.Image:
        """Post-process generated image.
        
        Args:
            image: Generated PIL Image
            config: Generation configuration
            face_service: Optional face processing service
        
        Returns:
            Post-processed image
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Face enhancement if available
            if face_service and config.enable_expressions:
                img_array = face_service.enhance_face(img_array)
            
            # Quality enhancements
            if config.quality in [AvatarQuality.HIGH, AvatarQuality.ULTRA]:
                img_array = ImageProcessor._apply_quality_enhancements(img_array)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logging.warning(f"Post-processing failed: {e}")
            return image
    
    @staticmethod
    def _apply_quality_enhancements(img_array: np.ndarray) -> np.ndarray:
        """Apply quality enhancements to image."""
        try:
            # Apply noise reduction
            enhanced = cv2.fastNlMeansDenoisingColored(
                img_array, None, 10, 10, 7, 21
            )
            
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Quality enhancement failed: {e}")
            return img_array


# =============================================================================
# Avatar Manager
# =============================================================================

class AvatarManager:
    """Main avatar management system.
    
    Features:
    - Avatar generation using Stable Diffusion
    - Face processing and enhancement
    - Image post-processing
    - Proper error handling and logging
    - GPU utilization and mixed precision
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ):
        """Initialize avatar manager.
        
        Args:
            device: PyTorch device (auto-detected if None)
            model_id: HuggingFace model ID for Stable Diffusion
        """
        self.logger = logging.getLogger(f"{__name__}.AvatarManager")
        self.device = device
        
        # Initialize services from modular components
        self.pipeline_manager = DiffusionPipelineManager(device=device)
        self.face_service = FaceProcessingService()
        self.image_processor = ImageProcessor()
        self.prompt_enhancer = PromptEnhancer()
        
        # Load default pipeline
        try:
            self.pipeline_manager.load_pipeline(model_id)
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
        
        self.logger.info("Avatar Manager initialized successfully")
    
    async def generate_avatar(
        self,
        prompt: str,
        config: Optional[AvatarGenerationConfig] = None,
    ) -> str:
        """Generate avatar image.
        
        Args:
            prompt: Text description of avatar
            config: Generation configuration
        
        Returns:
            Path to generated avatar image
        
        Raises:
            RuntimeError: If generation fails
        """
        if config is None:
            config = AvatarGenerationConfig()
        
        try:
            self.logger.info(f"Generating avatar with prompt: {prompt}")
            
            # Enhance prompt using prompt enhancer
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                prompt, config.style
            )
            
            # Select pipeline based on style
            pipeline_type = (
                "stable-diffusion-xl"
                if config.style == AvatarStyle.REALISTIC
                else "stable-diffusion-v1-5"
            )
            
            # Generate image
            image = self.pipeline_manager.generate_image(
                prompt=enhanced_prompt,
                negative_prompt="blurry, low quality, distorted",
                config=config,
                pipeline_type=pipeline_type,
            )
            
            # Post-process image
            processed_image = self.image_processor.post_process_image(
                image, config, self.face_service
            )
            
            # Save image
            output_dir = Path("./generated_avatars")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"avatar_{uuid.uuid4().hex[:8]}.png"
            processed_image.save(output_path)
            
            self.logger.info(f"Avatar generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Avatar generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "device": str(self.pipeline_manager.device),
            "pipelines_loaded": len(self.pipeline_manager.pipelines),
            "face_service_available": self.face_service.face_detection is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "face_libs_available": FACE_LIBS_AVAILABLE,
        }
