"""
Flux2 Clothing Changer Model V2
================================

Model based on official Flux2 architecture adapted for clothing changes.
Uses the real Flux2 core implementation from:
https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py

Adapted for inpainting-based clothing replacement.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple, List, Callable
from pathlib import Path
import logging
from PIL import Image
import numpy as np
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from diffusers import (
        FluxInpaintPipeline,
        FluxPipeline,
        FluxTransformer2DModel,
    )
    from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available, some features may be limited")

from .flux2_core import Flux2Core, Flux2Params
from .constants import (
    DEFAULT_MODEL_ID,
    DEFAULT_CLIP_MODEL_ID,
    CHARACTER_EMBEDDING_DIM,
    CLOTHING_EMBEDDING_DIM,
    MAX_IMAGE_SIZE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
)

# Import modular components from v1
from .flux2_clothing_model import (
    ImagePreprocessor,
    FeaturePooler,
    MaskGenerator,
    CharacterEncoder,
    ClothingEncoder,
)

# Import new utilities
from .resolution_handler import ResolutionHandler
from .memory_optimizer import MemoryOptimizer
from .lora_adapter import LoRAAdapter
from .processing import ImageValidator, ImageEnhancer
from .auto_optimizer import AutoOptimizer
from .plugin_system import PluginManager, HookType, ProcessingContext
from .metrics.quality_metrics import ImageQualityMetrics, ProcessingMetrics
from .utils.retry import retry_on_failure
from .validators.quality_validator import ImageQualityValidator
from .core.clothing_change_executor import ClothingChangeExecutor

logger = logging.getLogger(__name__)


class Flux2ClothingChangerModelV2(nn.Module):
    """
    Flux2 Clothing Changer Model using official Flux2 architecture.
    
    This version uses the actual Flux2 core implementation adapted from:
    https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py
    
    The model is wrapped with inpainting capabilities for clothing replacement.
    """
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_optimizations: bool = True,
        use_inpainting: bool = True,
        use_core_architecture: bool = True,
        validate_images: bool = True,
        enhance_images: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize Flux2 Clothing Changer Model V2.
        
        Args:
            model_id: HuggingFace model ID for Flux2
            device: Device to run on (cuda/cpu/auto)
            dtype: Data type (float16/float32)
            enable_optimizations: Enable memory optimizations
            use_inpainting: Use inpainting pipeline for clothing replacement
            use_core_architecture: Use official Flux2 core architecture (experimental)
            validate_images: Validate image quality before processing
            enhance_images: Enhance images automatically before processing
            max_retries: Maximum retry attempts for failed operations
        """
        super().__init__()
        
        self.model_id = model_id
        self.use_inpainting = use_inpainting
        self.use_core_architecture = use_core_architecture
        self.validate_images = validate_images
        self.enhance_images = enhance_images
        self.max_retries = max_retries
        
        # Initialize utilities
        self.resolution_handler = ResolutionHandler(
            target_resolution=None,  # Auto-detect optimal
            maintain_aspect_ratio=True,
        )
        self.memory_optimizer = MemoryOptimizer(self.device)
        self.lora_adapter: Optional[LoRAAdapter] = None
        
        # Initialize image processing utilities
        if self.validate_images:
            self.image_validator = ImageValidator()
        else:
            self.image_validator = None
        
        if self.enhance_images:
            self.image_enhancer = ImageEnhancer()
        else:
            self.image_enhancer = None
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Setup dtype
        if dtype is None:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = dtype
        
        self.enable_optimizations = enable_optimizations
        
        # Auto optimizer
        self.auto_optimizer = AutoOptimizer(
            enable_auto_tuning=True,
            enable_adaptive_steps=True,
            enable_adaptive_guidance=True,
        )
        
        # Plugin system
        self.plugin_manager = PluginManager()
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "images_validated": 0,
            "images_enhanced": 0,
            "validation_failures": 0,
            "clothing_changes": 0,
            "successful_changes": 0,
            "failed_changes": 0,
            "total_time": 0.0,
        }
        
        # Initialize components
        self._build_model()
        
        logger.info(f"Flux2ClothingChangerModelV2 initialized on {self.device}")
        logger.info(f"Using core architecture: {use_core_architecture}")
        logger.info(f"Image validation: {validate_images}, Enhancement: {enhance_images}, Max retries: {max_retries}")
    
    def _build_model(self) -> None:
        """Build the model architecture."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library is required. Install with: pip install diffusers transformers"
            )
        
        try:
            # Load Flux2 pipeline (uses official implementation internally)
            if self.use_inpainting:
                self.pipeline = FluxInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                )
            else:
                self.pipeline = FluxPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                )
            
            # Load CLIP components
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(DEFAULT_CLIP_MODEL_ID)
            self.clip_text_model = CLIPTextModel.from_pretrained(
                DEFAULT_CLIP_MODEL_ID,
                torch_dtype=self.dtype,
            )
            
            self.clip_processor = CLIPImageProcessor.from_pretrained(DEFAULT_CLIP_MODEL_ID)
            self.clip_vision = CLIPVisionModel.from_pretrained(
                DEFAULT_CLIP_MODEL_ID,
                torch_dtype=self.dtype,
            )
            
            # Initialize modular components (same as v1)
            clip_hidden_size = self.clip_vision.config.hidden_size
            text_hidden_size = self.clip_text_model.config.hidden_size
            
            self.preprocessor = ImagePreprocessor(
                self.clip_processor,
                self.device,
                MAX_IMAGE_SIZE
            )
            
            self.pooler = FeaturePooler()
            
            self.character_encoder_module = CharacterEncoder(
                clip_hidden_size,
                CHARACTER_EMBEDDING_DIM
            )
            
            self.clothing_encoder_module = ClothingEncoder(
                text_hidden_size,
                CLOTHING_EMBEDDING_DIM
            )
            
            self.mask_generator = MaskGenerator()
            
            # Initialize clothing change executor
            self.clothing_change_executor = ClothingChangeExecutor(
                preprocessor=self.preprocessor,
                mask_generator=self.mask_generator,
                validate_images=self.validate_images,
                enhance_images=self.enhance_images,
            )
            
            # Fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(CHARACTER_EMBEDDING_DIM + CLOTHING_EMBEDDING_DIM, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, CHARACTER_EMBEDDING_DIM),
                nn.LayerNorm(CHARACTER_EMBEDDING_DIM),
            )
            
            # Optional: Initialize Flux2 core architecture for custom inference
            if self.use_core_architecture:
                try:
                    # Get transformer from pipeline to access core architecture
                    if hasattr(self.pipeline, 'transformer'):
                        self.transformer = self.pipeline.transformer
                        logger.info("Using Flux2 transformer from pipeline")
                    else:
                        logger.warning("Pipeline transformer not accessible, using pipeline directly")
                        self.transformer = None
                except Exception as e:
                    logger.warning(f"Could not access core architecture: {e}")
                    self.transformer = None
            
            # Initialize weights
            self._initialize_weights()
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            self.clip_text_model = self.clip_text_model.to(self.device)
            self.clip_vision = self.clip_vision.to(self.device)
            self.character_encoder_module = self.character_encoder_module.to(self.device)
            self.clothing_encoder_module = self.clothing_encoder_module.to(self.device)
            self.fusion_layer = self.fusion_layer.to(self.device)
            
            # Enable optimizations
            if self.enable_optimizations and self.device.type == "cuda":
                self._enable_optimizations()
                # Apply advanced memory optimizations
                if self.memory_optimizer:
                    self.memory_optimizer.apply_all_optimizations(
                        self.pipeline,
                        enable_attention_slicing=True,
                        enable_vae_slicing=True,
                        enable_xformers=True,
                    )
            
            # Set to eval mode
            self.eval()
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise RuntimeError(f"Failed to build Flux2 model: {e}")
    
    
    def encode_character(
        self, 
        image: Union[Image.Image, str, Path, np.ndarray],
        validate: Optional[bool] = None,
        enhance: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Encode character features from image with optional validation and enhancement.
        
        Args:
            image: Input character image
            validate: Override validate_images setting
            enhance: Override enhance_images setting
            
        Returns:
            Character embedding tensor [CHARACTER_EMBEDDING_DIM]
        """
        validate = validate if validate is not None else self.validate_images
        enhance = enhance if enhance is not None else self.enhance_images
        
        # Convert to PIL if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Validate image quality
        if validate and self.image_validator:
            is_valid, validation_info = self.image_validator.validate(pil_image)
            self.stats["images_validated"] += 1
            
            if not is_valid:
                self.stats["validation_failures"] += 1
                error_msg = f"Image validation failed: {', '.join(validation_info['errors'])}"
                logger.warning(error_msg)
                raise ValueError(error_msg)
            
            if validation_info["warnings"]:
                logger.debug(f"Image quality warnings: {', '.join(validation_info['warnings'])}")
        
        # Enhance image if requested
        if enhance and self.image_enhancer:
            pil_image, enhancement_info = self.image_enhancer.enhance(
                pil_image,
                metrics=validation_info.get("metrics") if validate and self.image_validator else None
            )
            self.stats["images_enhanced"] += 1
            logger.debug(f"Image enhanced: {enhancement_info['operations']}")
        
        with torch.no_grad():
            pixel_values = self.preprocessor.preprocess(pil_image)
            clip_outputs = self.clip_vision(pixel_values=pixel_values)
            image_features = clip_outputs.last_hidden_state
            pooled_features = self.pooler.pool_features(image_features)
            character_embedding = self.character_encoder_module(pooled_features)
            return character_embedding.squeeze(0)
    
    def encode_clothing_description(self, clothing_description: str) -> torch.Tensor:
        """
        Encode clothing description into embedding.
        
        Args:
            clothing_description: Text description of clothing
            
        Returns:
            Clothing embedding tensor [CLOTHING_EMBEDDING_DIM]
        """
        with torch.no_grad():
            inputs = self.clip_tokenizer(
                clothing_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_outputs = self.clip_text_model(**inputs)
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            clothing_embedding = self.clothing_encoder_module(text_features)
            return clothing_embedding.squeeze(0)
    
    @retry_on_failure(max_attempts=3, delay=0.5)
    def change_clothing(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        clothing_description: str,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        strength: float = DEFAULT_STRENGTH,
        return_metrics: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, ProcessingMetrics]]:
        """
        Change clothing in character image using Flux2 inpainting.
        
        This method uses the official Flux2 architecture through the diffusers pipeline,
        which internally uses the Flux2 core implementation.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            mask: Optional mask for clothing area
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength (0.0 to 1.0)
            return_metrics: If True, also return processing metrics
            
        Returns:
            Image with changed clothing or tuple (image, metrics)
        """
        start_time = time.time()
        metrics = ProcessingMetrics(
            processing_time=0.0,
            mask_quality=0.0,
            prompt_quality=0.0,
            success=False,
            errors=[]
        )
        
        try:
            with torch.no_grad():
                # Prepare image
                pil_image, metrics = self.clothing_change_executor.prepare_image(
                    image,
                    stats=self.stats
                )
                
                # Prepare mask
                mask_image, mask_quality = self.clothing_change_executor.prepare_mask(
                    pil_image,
                    mask,
                    stats=self.stats
                )
                metrics.mask_quality = mask_quality
                
                # Prepare prompts
                prompt, negative_prompt, prompt_quality = self.clothing_change_executor.prepare_prompt(
                    clothing_description,
                    prompt,
                    negative_prompt,
                    DEFAULT_NEGATIVE_PROMPT
                )
                metrics.prompt_quality = prompt_quality
                
                # Execute pipeline
                result = self.clothing_change_executor.execute_pipeline(
                    self.pipeline,
                    prompt,
                    pil_image,
                    mask_image,
                    negative_prompt,
                    num_inference_steps,
                    guidance_scale,
                    strength,
                    self.use_inpainting
                )
                
                # Update statistics
                processing_time = time.time() - start_time
                self.clothing_change_executor.update_stats(
                    self.stats,
                    metrics,
                    processing_time,
                    True
                )
                
                if return_metrics:
                    return result, metrics
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            metrics.errors.append(str(e))
            self.clothing_change_executor.update_stats(
                self.stats,
                metrics,
                processing_time,
                False
            )
            logger.error(f"Error changing clothing: {e}", exc_info=True)
            
            if return_metrics:
                return None, metrics
            raise
    
    def forward(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        clothing_description: str,
        **kwargs
    ) -> Image.Image:
        """
        Forward pass - change clothing.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            **kwargs: Additional arguments for change_clothing
            
        Returns:
            Image with changed clothing
        """
        return self.change_clothing(image, clothing_description, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "use_inpainting": self.use_inpainting,
            "use_core_architecture": self.use_core_architecture,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimizations_enabled": self.enable_optimizations,
            "architecture": "Flux2 Official (via diffusers)",
        }
        
        # Add pipeline info if available
        if hasattr(self.pipeline, 'transformer'):
            try:
                transformer_params = sum(p.numel() for p in self.pipeline.transformer.parameters())
                info["transformer_parameters"] = transformer_params
            except Exception:
                pass
        
        # Calculate averages
        avg_time = (
            self.stats["total_time"] / self.stats["clothing_changes"]
            if self.stats["clothing_changes"] > 0 else 0.0
        )
        success_rate = (
            self.stats["successful_changes"] / self.stats["clothing_changes"]
            if self.stats["clothing_changes"] > 0 else 0.0
        )
        
        info.update({
            "avg_processing_time": round(avg_time, 3),
            "success_rate": round(success_rate, 3),
            "stats": self.stats,
        })
        
        return info