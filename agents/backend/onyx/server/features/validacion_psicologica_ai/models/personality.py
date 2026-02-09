"""
Personality Classifier
======================
Big Five personality classifier using transformers
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import structlog

from .base import BaseModel
from ..config_loader import config_loader

logger = structlog.get_logger()


class PersonalityClassifier(BaseModel):
    """
    Big Five personality classifier using transformers
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_traits: int = 5,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize personality classifier
        
        Args:
            model_name: Model name (uses config if None)
            num_traits: Number of traits (Big Five = 5)
            hidden_dim: Hidden dimension (uses config if None)
            device: Device (auto-detect if None)
        """
        super().__init__(device)
        
        self.num_traits = num_traits
        
        # Load config
        model_config = config_loader.get_model_config("personality")
        model_name = model_name or model_config.get("name", "distilbert-base-uncased")
        hidden_dim = hidden_dim or model_config.get("hidden_dim", 768)
        dropout = model_config.get("dropout", 0.1)
        self.max_length = model_config.get("max_length", 512)
        
        # Load model
        self.tokenizer = None
        self.backbone = None
        self._load_model(model_name, hidden_dim)
        
        # Trait heads
        self.trait_heads = nn.ModuleDict({
            "openness": nn.Linear(hidden_dim, 1),
            "conscientiousness": nn.Linear(hidden_dim, 1),
            "extraversion": nn.Linear(hidden_dim, 1),
            "agreeableness": nn.Linear(hidden_dim, 1),
            "neuroticism": nn.Linear(hidden_dim, 1)
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def _load_model(self, model_name: str, hidden_dim: int) -> None:
        """Load model with error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            self.backbone = self.backbone.to(self.device)
            logger.info("Personality model loaded", model_name=model_name)
        except Exception as e:
            logger.error("Error loading personality model", error=str(e), model_name=model_name)
            self.tokenizer = None
            self.backbone = None
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary of trait predictions
        """
        if not texts or self.backbone is None:
            # Return zeros if model not loaded
            batch_size = len(texts) if texts else 1
            return {
                trait: torch.zeros(batch_size, 1, device=self.device)
                for trait in self.trait_heads.keys()
            }
        
        try:
            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.backbone(**encoded)
                # Use [CLS] token
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Apply dropout
            embeddings = self.dropout(embeddings)
            
            # Get predictions for each trait
            predictions = {}
            for trait, head in self.trait_heads.items():
                predictions[trait] = torch.sigmoid(head(embeddings))
            
            return predictions
        except Exception as e:
            logger.error("Error in personality forward", error=str(e))
            batch_size = len(texts)
            return {
                trait: torch.zeros(batch_size, 1, device=self.device)
                for trait in self.trait_heads.keys()
            }




