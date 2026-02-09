"""
Embedding Model
===============
Semantic embedding model for psychological analysis
"""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import structlog

from .base import BaseModel
from ..config_loader import config_loader

logger = structlog.get_logger()


class PsychologicalEmbeddingModel(BaseModel):
    """
    Semantic embedding model for psychological analysis
    Uses transformers for semantic embeddings
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: Model name (uses config if None)
            embedding_dim: Embedding dimension (uses config if None)
            dropout: Dropout rate (uses config if None)
            device: Device (auto-detect if None)
        """
        super().__init__(device)
        
        # Load config
        model_config = config_loader.get_model_config("embedding")
        self.model_name = model_name or model_config.get("name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = embedding_dim or model_config.get("embedding_dim", 384)
        dropout = dropout or model_config.get("dropout", 0.1)
        self.max_length = model_config.get("max_length", 512)
        
        # Load model
        self.tokenizer = None
        self.backbone = None
        self._load_model()
        
        # Layers
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _load_model(self) -> None:
        """Load model with error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name)
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()
            logger.info("Embedding model loaded", model_name=self.model_name)
        except Exception as e:
            logger.error("Error loading embedding model", error=str(e), model_name=self.model_name)
            raise
    
    def _initialize_weights(self) -> None:
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            texts: List of texts
            
        Returns:
            Embeddings tensor [batch_size, embedding_dim]
        """
        if not texts:
            return torch.zeros(0, self.embedding_dim, device=self.device)
        
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
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Project and apply dropout
            embeddings = self.projection(embeddings)
            embeddings = self.dropout(embeddings)
            
            return embeddings
        except Exception as e:
            logger.error("Error in embedding forward", error=str(e))
            return torch.zeros(len(texts), self.embedding_dim, device=self.device)




