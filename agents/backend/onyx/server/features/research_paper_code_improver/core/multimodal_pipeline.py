"""
Multi-Modal Pipeline - Pipeline para modelos multi-modales
===========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultiModalConfig:
    """Configuración multi-modal"""
    text_model_name: str = "bert-base-uncased"
    vision_model_name: str = "google/vit-base-patch16-224"
    fusion_method: str = "concat"  # "concat", "cross_attention", "bilinear"
    output_dim: int = 768
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultiModalEncoder(nn.Module):
    """Encoder multi-modal"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Text encoder
        try:
            from transformers import AutoModel, AutoTokenizer
            self.text_model = AutoModel.from_pretrained(config.text_model_name)
            self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        except ImportError:
            logger.warning("Transformers no disponible")
            self.text_model = None
            self.text_tokenizer = None
        
        # Vision encoder
        try:
            from transformers import AutoImageProcessor
            self.vision_model = AutoModel.from_pretrained(config.vision_model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        except ImportError:
            logger.warning("Vision model no disponible")
            self.vision_model = None
            self.image_processor = None
        
        # Fusion layer
        text_dim = self.text_model.config.hidden_size if self.text_model else 768
        vision_dim = self.vision_model.config.hidden_size if self.vision_model else 768
        
        if config.fusion_method == "concat":
            self.fusion = nn.Linear(text_dim + vision_dim, config.output_dim)
        elif config.fusion_method == "cross_attention":
            from .custom_attention import CrossAttention
            self.fusion = CrossAttention(config.output_dim, num_heads=8)
        else:
            self.fusion = nn.Linear(text_dim + vision_dim, config.output_dim)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Codifica texto"""
        if not self.text_model or not self.text_tokenizer:
            raise ValueError("Text model no disponible")
        
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Codifica imagen"""
        if not self.vision_model or not self.image_processor:
            raise ValueError("Vision model no disponible")
        
        inputs = self.image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        return outputs.last_hidden_state
    
    def forward(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None
    ) -> torch.Tensor:
        """Forward pass multi-modal"""
        text_embedding = None
        image_embedding = None
        
        if text:
            text_embedding = self.encode_text(text)
        
        if image:
            image_embedding = self.encode_image(image)
        
        # Fusion
        if text_embedding is not None and image_embedding is not None:
            if self.config.fusion_method == "concat":
                # Pool embeddings
                text_pooled = text_embedding.mean(dim=1)
                image_pooled = image_embedding.mean(dim=1)
                fused = torch.cat([text_pooled, image_pooled], dim=-1)
                return self.fusion(fused)
            elif self.config.fusion_method == "cross_attention":
                fused, _ = self.fusion(text_embedding, image_embedding, image_embedding)
                return fused.mean(dim=1)
        elif text_embedding is not None:
            return text_embedding.mean(dim=1)
        elif image_embedding is not None:
            return image_embedding.mean(dim=1)
        else:
            raise ValueError("Al menos texto o imagen debe ser proporcionado")


class MultiModalPipeline:
    """Pipeline multi-modal"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.encoder = MultiModalEncoder(config)
        self.device = torch.device(config.device)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
    
    def encode(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None
    ) -> torch.Tensor:
        """Codifica entrada multi-modal"""
        with torch.no_grad():
            embedding = self.encoder(text=text, image=image)
        return embedding
    
    def similarity(
        self,
        text1: Optional[str] = None,
        image1: Optional[Image.Image] = None,
        text2: Optional[str] = None,
        image2: Optional[Image.Image] = None
    ) -> float:
        """Calcula similitud entre dos entradas multi-modales"""
        emb1 = self.encode(text=text1, image=image1)
        emb2 = self.encode(text=text2, image=image2)
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        return cos_sim.item()




