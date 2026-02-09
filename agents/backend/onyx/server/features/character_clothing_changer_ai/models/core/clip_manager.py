"""
CLIP Manager
============

Manages CLIP model loading and initialization.
"""

import logging
from typing import Tuple, Optional
import torch

try:
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModel,
        CLIPTextModel,
        CLIPTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..constants import DEFAULT_CLIP_MODEL_ID

logger = logging.getLogger(__name__)


class CLIPManager:
    """Manages CLIP model components."""
    
    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL_ID,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize CLIP manager.
        
        Args:
            model_id: CLIP model ID
            device: Torch device
            dtype: Data type
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        self.clip_tokenizer: Optional[CLIPTokenizer] = None
        self.clip_text_model: Optional[CLIPTextModel] = None
        self.clip_processor: Optional[CLIPImageProcessor] = None
        self.clip_vision: Optional[CLIPVisionModel] = None
    
    def load_models(self) -> Tuple[CLIPTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModel]:
        """
        Load all CLIP components.
        
        Returns:
            Tuple of (tokenizer, text_model, processor, vision_model)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required. Install with: pip install transformers"
            )
        
        try:
            logger.info(f"Loading CLIP models from {self.model_id}")
            
            # Load tokenizer
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
            
            # Load text model
            self.clip_text_model = CLIPTextModel.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            
            # Load processor
            self.clip_processor = CLIPImageProcessor.from_pretrained(self.model_id)
            
            # Load vision model
            self.clip_vision = CLIPVisionModel.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            
            # Move to device
            if self.device is not None:
                self.clip_text_model = self.clip_text_model.to(self.device)
                self.clip_vision = self.clip_vision.to(self.device)
            
            logger.info("CLIP models loaded successfully")
            
            return (
                self.clip_tokenizer,
                self.clip_text_model,
                self.clip_processor,
                self.clip_vision,
            )
            
        except Exception as e:
            logger.error(f"Error loading CLIP models: {e}")
            raise RuntimeError(f"Failed to load CLIP models: {e}") from e
    
    def get_hidden_sizes(self) -> Tuple[int, int]:
        """
        Get hidden sizes for text and vision models.
        
        Returns:
            Tuple of (text_hidden_size, vision_hidden_size)
        """
        if self.clip_text_model is None or self.clip_vision is None:
            raise RuntimeError("CLIP models not loaded. Call load_models() first.")
        
        text_hidden_size = self.clip_text_model.config.hidden_size
        vision_hidden_size = self.clip_vision.config.hidden_size
        
        return text_hidden_size, vision_hidden_size


