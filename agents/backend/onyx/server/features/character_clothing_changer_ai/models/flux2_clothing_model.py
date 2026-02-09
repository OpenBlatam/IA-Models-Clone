"""
Flux2 Clothing Changer Model
============================

Model based on Flux2 architecture for changing character clothing.
Supports inpainting and control-based clothing replacement.
Refactored with modular architecture for better maintainability.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import logging
from PIL import Image
import numpy as np

try:
    from diffusers import (
        FluxPipeline,
        FluxTransformer2DModel,
        FluxInpaintPipeline,
        ControlNetModel,
    )
    from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available, some features may be limited")

# Import modular components
from .processing import ImagePreprocessor, FeaturePooler, MaskGenerator, MaskProcessor
from .encoding import CharacterEncoder, ClothingEncoder
from .core import (
    ModelBuilder,
    DeviceManager,
    PromptGenerator,
    PipelineOrchestrator,
    EncodingOrchestrator,
    PreprocessingOrchestrator,
    ClothingChangeOrchestrator,
)

from .constants import (
    DEFAULT_MODEL_ID,
    DEFAULT_CLIP_MODEL_ID,
    CHARACTER_EMBEDDING_DIM,
    CLOTHING_EMBEDDING_DIM,
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DROPOUT_RATE,
)

logger = logging.getLogger(__name__)


class Flux2ClothingChangerModel(nn.Module):
    """
    Flux2-based model for changing character clothing.
    
    Processes character images and generates new images with changed clothing
    while maintaining character consistency.
    
    Refactored with modular architecture:
    - ImagePreprocessor: Handles image preprocessing
    - FeaturePooler: Pools CLIP features
    - MaskGenerator: Generates clothing masks
    - CharacterEncoder: Encodes character features
    - ClothingEncoder: Encodes clothing descriptions
    """
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/flux2-dev",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_optimizations: bool = True,
        use_inpainting: bool = True,
        use_controlnet: bool = False,
    ):
        """
        Initialize Flux2 Clothing Changer Model.
        
        Args:
            model_id: HuggingFace model ID for Flux2
            device: Device to run on (cuda/cpu/auto)
            dtype: Data type (float16/float32)
            enable_optimizations: Enable memory optimizations
            use_inpainting: Use inpainting pipeline for clothing replacement
            use_controlnet: Use ControlNet for better control (experimental)
        """
        super().__init__()
        
        self.model_id = model_id
        self.use_inpainting = use_inpainting
        self.use_controlnet = use_controlnet
        
        # Setup device and dtype using DeviceManager
        self.device = DeviceManager.get_device(device)
        self.dtype = DeviceManager.get_dtype(self.device, dtype)
        self.enable_optimizations = enable_optimizations
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator()
        
        # Initialize components
        self._build_model()
        
        logger.info(f"Flux2ClothingChangerModel initialized on {self.device}")
    
    def _build_model(self) -> None:
        """Build the model architecture using ModelBuilder."""
        try:
            # Use ModelBuilder to construct all components
            builder = ModelBuilder(
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
                use_inpainting=self.use_inpainting,
            )
            
            components = builder.build_all(enable_optimizations=self.enable_optimizations)
            
            # Assign components
            self.pipeline = components["pipeline"]
            self.clip_tokenizer = components["clip_tokenizer"]
            self.clip_text_model = components["clip_text_model"]
            self.clip_processor = components["clip_processor"]
            self.clip_vision = components["clip_vision"]
            self.preprocessor = components["preprocessor"]
            self.pooler = components["pooler"]
            self.mask_generator = components["mask_generator"]
            self.character_encoder_module = components["character_encoder"]
            self.clothing_encoder_module = components["clothing_encoder"]
            self.fusion_layer = components["fusion_layer"]
            
            # Initialize orchestrators
            self.mask_processor = MaskProcessor()
            self.pipeline_orchestrator = PipelineOrchestrator(
                pipeline=self.pipeline,
                use_inpainting=self.use_inpainting,
            )
            self.encoding_orchestrator = EncodingOrchestrator(
                preprocessor=self.preprocessor,
                pooler=self.pooler,
                clip_vision=self.clip_vision,
                clip_text_model=self.clip_text_model,
                clip_tokenizer=self.clip_tokenizer,
                character_encoder=self.character_encoder_module,
                clothing_encoder=self.clothing_encoder_module,
                device=self.device,
            )
            self.preprocessing_orchestrator = PreprocessingOrchestrator(
                preprocessor=self.preprocessor,
            )
            self.clothing_change_orchestrator = ClothingChangeOrchestrator(
                preprocessing_orchestrator=self.preprocessing_orchestrator,
                pipeline_orchestrator=self.pipeline_orchestrator,
                mask_processor=self.mask_processor,
                prompt_generator=self.prompt_generator,
            )
            
            # Set to eval mode
            self.eval()
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise RuntimeError(f"Failed to build Flux2 model: {e}")
    
    
    def encode_character(self, image: Union[Image.Image, str, Path, np.ndarray]) -> torch.Tensor:
        """
        Encode character features from image with improved pooling.
        
        Args:
            image: Input character image
            
        Returns:
            Character embedding tensor [CHARACTER_EMBEDDING_DIM]
        """
        return self.encoding_orchestrator.encode_character(image)
    
    def encode_clothing_description(self, clothing_description: str) -> torch.Tensor:
        """
        Encode clothing description into embedding with improved encoding.
        
        Args:
            clothing_description: Text description of clothing
            
        Returns:
            Clothing embedding tensor [CLOTHING_EMBEDDING_DIM]
        """
        return self.encoding_orchestrator.encode_clothing_description(clothing_description)
    
    def change_clothing(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        clothing_description: str,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
    ) -> Image.Image:
        """
        Change clothing in character image.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            mask: Optional mask for clothing area (if None, will try to auto-detect)
            prompt: Optional full prompt (will be generated if not provided)
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength (0.0 to 1.0)
            
        Returns:
            Image with changed clothing
        """
        with torch.no_grad():
            return self.clothing_change_orchestrator.change_clothing(
                image=image,
                clothing_description=clothing_description,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            )
    
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
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "use_inpainting": self.use_inpainting,
            "use_controlnet": self.use_controlnet,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimizations_enabled": self.enable_optimizations,
        }


# Export for backward compatibility
__all__ = [
    "Flux2ClothingChangerModel",
    "ImagePreprocessor",
    "FeaturePooler",
    "MaskGenerator",
    "CharacterEncoder",
    "ClothingEncoder",
]
