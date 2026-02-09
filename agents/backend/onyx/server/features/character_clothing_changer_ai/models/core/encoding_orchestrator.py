"""
Encoding Orchestrator
=====================

Orchestrates encoding operations for character and clothing.
"""

import logging
from typing import Union
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from ..processing import ImagePreprocessor, FeaturePooler
from ..encoding import CharacterEncoder, ClothingEncoder

try:
    from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EncodingOrchestrator:
    """Orchestrates encoding operations."""
    
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        pooler: FeaturePooler,
        clip_vision: CLIPVisionModel,
        clip_text_model: CLIPTextModel,
        clip_tokenizer: CLIPTokenizer,
        character_encoder: CharacterEncoder,
        clothing_encoder: ClothingEncoder,
        device: torch.device,
    ):
        """
        Initialize encoding orchestrator.
        
        Args:
            preprocessor: Image preprocessor
            pooler: Feature pooler
            clip_vision: CLIP vision model
            clip_text_model: CLIP text model
            clip_tokenizer: CLIP tokenizer
            character_encoder: Character encoder
            clothing_encoder: Clothing encoder
            device: Torch device
        """
        self.preprocessor = preprocessor
        self.pooler = pooler
        self.clip_vision = clip_vision
        self.clip_text_model = clip_text_model
        self.clip_tokenizer = clip_tokenizer
        self.character_encoder = character_encoder
        self.clothing_encoder = clothing_encoder
        self.device = device
    
    def encode_character(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
    ) -> torch.Tensor:
        """
        Encode character features from image.
        
        Args:
            image: Input character image
            
        Returns:
            Character embedding tensor
        """
        with torch.no_grad():
            # Preprocess image
            pixel_values = self.preprocessor.preprocess(image)
            
            # Encode with CLIP vision
            clip_outputs = self.clip_vision(pixel_values=pixel_values)
            image_features = clip_outputs.last_hidden_state
            
            # Pool features
            pooled_features = self.pooler.pool_features(image_features)
            
            # Encode character
            character_embedding = self.character_encoder(pooled_features)
            
            return character_embedding.squeeze(0)
    
    def encode_clothing_description(
        self,
        clothing_description: str,
    ) -> torch.Tensor:
        """
        Encode clothing description into embedding.
        
        Args:
            clothing_description: Text description of clothing
            
        Returns:
            Clothing embedding tensor
        """
        with torch.no_grad():
            # Tokenize and encode text
            inputs = self.clip_tokenizer(
                clothing_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text embeddings
            text_outputs = self.clip_text_model(**inputs)
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            
            # Encode clothing
            clothing_embedding = self.clothing_encoder(text_features)
            
            return clothing_embedding.squeeze(0)


