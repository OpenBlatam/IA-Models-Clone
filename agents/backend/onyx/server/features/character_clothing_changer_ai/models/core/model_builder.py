"""
Model Builder
=============

Builds Flux2 clothing changer model with modular components.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

from .pipeline_manager import PipelineManager
from .clip_manager import CLIPManager
from .device_manager import DeviceManager
from ..processing import ImagePreprocessor, FeaturePooler, MaskGenerator
from ..encoding import CharacterEncoder, ClothingEncoder
from ..constants import (
    CHARACTER_EMBEDDING_DIM,
    CLOTHING_EMBEDDING_DIM,
    MAX_IMAGE_SIZE,
    DROPOUT_RATE,
)

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Builds Flux2 clothing changer model components."""
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/flux2-dev",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_inpainting: bool = True,
    ):
        """
        Initialize model builder.
        
        Args:
            model_id: Model ID
            device: Torch device
            dtype: Data type
            use_inpainting: Use inpainting pipeline
        """
        self.model_id = model_id
        self.device = device or DeviceManager.get_device()
        self.dtype = dtype or DeviceManager.get_dtype(self.device)
        self.use_inpainting = use_inpainting
        
        # Initialize managers
        self.pipeline_manager = PipelineManager(
            model_id=model_id,
            device=self.device,
            dtype=self.dtype,
            use_inpainting=use_inpainting,
        )
        
        self.clip_manager = CLIPManager(
            device=self.device,
            dtype=self.dtype,
        )
    
    def build_all(self, enable_optimizations: bool = True) -> dict:
        """
        Build all model components.
        
        Args:
            enable_optimizations: Enable optimizations
            
        Returns:
            Dictionary with all components
        """
        # Load pipeline
        pipeline = self.pipeline_manager.load_pipeline()
        if enable_optimizations:
            self.pipeline_manager.apply_optimizations()
        
        # Load CLIP components
        tokenizer, text_model, processor, vision_model = self.clip_manager.load_models()
        
        # Get hidden sizes
        text_hidden_size, vision_hidden_size = self.clip_manager.get_hidden_sizes()
        
        # Initialize processing components
        preprocessor = ImagePreprocessor(processor, self.device, MAX_IMAGE_SIZE)
        pooler = FeaturePooler()
        mask_generator = MaskGenerator()
        
        # Initialize encoding components
        character_encoder = CharacterEncoder(vision_hidden_size, CHARACTER_EMBEDDING_DIM)
        clothing_encoder = ClothingEncoder(text_hidden_size, CLOTHING_EMBEDDING_DIM)
        
        # Create fusion layer
        fusion_layer = nn.Sequential(
            nn.Linear(CHARACTER_EMBEDDING_DIM + CLOTHING_EMBEDDING_DIM, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, CHARACTER_EMBEDDING_DIM),
            nn.LayerNorm(CHARACTER_EMBEDDING_DIM),
        )
        
        # Move to device
        character_encoder = character_encoder.to(self.device)
        clothing_encoder = clothing_encoder.to(self.device)
        fusion_layer = fusion_layer.to(self.device)
        
        # Initialize weights
        self._initialize_weights(character_encoder)
        self._initialize_weights(clothing_encoder)
        self._initialize_weights(fusion_layer)
        
        return {
            "pipeline": pipeline,
            "clip_tokenizer": tokenizer,
            "clip_text_model": text_model,
            "clip_processor": processor,
            "clip_vision": vision_model,
            "preprocessor": preprocessor,
            "pooler": pooler,
            "mask_generator": mask_generator,
            "character_encoder": character_encoder,
            "clothing_encoder": clothing_encoder,
            "fusion_layer": fusion_layer,
        }
    
    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Initialize module weights."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


