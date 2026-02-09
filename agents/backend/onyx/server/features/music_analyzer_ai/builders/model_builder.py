"""
Model Builder

Builder pattern for constructing complex models step by step.
Follows PyTorch best practices for model construction.
"""

from typing import Dict, Any, Optional, List
import logging
import torch
import torch.nn as nn

from ..interfaces.base import IModel
from ..models.architectures.attention import MultiHeadAttention
from ..models.architectures.feedforward import ResidualFeedForward
from ..models.architectures.normalization import LayerNorm
from ..models.architectures.positional_encoding import LearnedPositionalEncoding

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Builder for constructing transformer models.
    
    Allows step-by-step construction of complex models
    with proper configuration and validation.
    """
    
    def __init__(self):
        """Initialize model builder."""
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self.config: Dict[str, Any] = {}
        self.layers: List[nn.Module] = []
        return self
    
    def set_embed_dim(self, embed_dim: int):
        """Set embedding dimension."""
        self.config["embed_dim"] = embed_dim
        return self
    
    def set_num_heads(self, num_heads: int):
        """Set number of attention heads."""
        self.config["num_heads"] = num_heads
        return self
    
    def set_num_layers(self, num_layers: int):
        """Set number of layers."""
        self.config["num_layers"] = num_layers
        return self
    
    def set_ff_dim(self, ff_dim: int):
        """Set feedforward dimension."""
        self.config["ff_dim"] = ff_dim
        return self
    
    def set_dropout(self, dropout: float):
        """Set dropout rate."""
        self.config["dropout"] = dropout
        return self
    
    def set_max_seq_len(self, max_seq_len: int):
        """Set maximum sequence length."""
        self.config["max_seq_len"] = max_seq_len
        return self
    
    def add_embedding_layer(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None
    ):
        """Add embedding layer."""
        embed_dim = embed_dim or self.config.get("embed_dim", 512)
        # Embedding layer would be added here
        return self
    
    def add_positional_encoding(self):
        """Add positional encoding."""
        embed_dim = self.config.get("embed_dim", 512)
        max_seq_len = self.config.get("max_seq_len", 512)
        
        pos_enc = LearnedPositionalEncoding(embed_dim, max_seq_len)
        self.layers.append(("positional_encoding", pos_enc))
        return self
    
    def add_attention_layer(
        self,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None
    ):
        """Add attention layer."""
        embed_dim = embed_dim or self.config.get("embed_dim", 512)
        num_heads = num_heads or self.config.get("num_heads", 8)
        dropout = dropout or self.config.get("dropout", 0.1)
        
        attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.layers.append(("attention", attention))
        return self
    
    def add_feedforward_layer(
        self,
        embed_dim: Optional[int] = None,
        ff_dim: Optional[int] = None,
        dropout: Optional[float] = None
    ):
        """Add feedforward layer."""
        embed_dim = embed_dim or self.config.get("embed_dim", 512)
        ff_dim = ff_dim or self.config.get("ff_dim", 2048)
        dropout = dropout or self.config.get("dropout", 0.1)
        
        ff = ResidualFeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.layers.append(("feedforward", ff))
        return self
    
    def build(self) -> nn.Module:
        """
        Build the model from configured layers.
        
        Returns:
            Constructed model
        """
        if not self.layers:
            raise ValueError("No layers added. Use builder methods to add layers.")
        
        # Create sequential model from layers
        modules = [layer for _, layer in self.layers]
        model = nn.Sequential(*modules)
        
        logger.info(f"Built model with {len(self.layers)} layers")
        return model


__all__ = [
    "ModelBuilder",
]
