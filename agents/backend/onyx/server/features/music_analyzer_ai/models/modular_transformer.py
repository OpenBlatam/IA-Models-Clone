"""
Modular Transformer Model using Architecture Components
Refactored to use modular components for better maintainability
"""

from typing import Optional, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .architectures import (
    MultiHeadAttention,
    ResidualFeedForward,
    LearnedPositionalEncoding,
    MusicFeatureEmbedding
)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer using modular components
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        # Use modular attention component
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Use modular feedforward component
        self.feedforward = ResidualFeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            pre_norm=pre_norm
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        residual = x
        if self.pre_norm:
            x = self.attention_norm(x)
            x = self.attention(x, x, x, mask)
            x = self.dropout(x)
            x = x + residual
        else:
            x = self.attention(x, x, x, mask)
            x = self.dropout(x)
            x = x + residual
            x = self.attention_norm(x)
        
        # Feedforward with residual (handled by ResidualFeedForward)
        x = self.feedforward(x)
        
        return x


class ModularTransformerEncoder(nn.Module):
    """
    Modular Transformer Encoder using architecture components
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        pre_norm: bool = True
    ):
        super().__init__()
        
        # Use modular embedding component
        self.embedding = MusicFeatureEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Use modular positional encoding
        self.pos_encoding = LearnedPositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout,
            max_len=max_seq_len
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask [batch, seq_len]
        
        Returns:
            Encoded features [batch, embed_dim]
        """
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Embed features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, mask)
            
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf in layer {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Global pooling (with mask support)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            masked_sum = (x * mask_expanded).sum(dim=1)
            mask_count = mask_expanded.sum(dim=1)
            x = masked_sum / (mask_count + 1e-8)
        else:
            x = x.mean(dim=1)
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in final output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x


class ModularMusicClassifier(nn.Module):
    """
    Modular music classifier using transformer encoder
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        num_genres: int = 10,
        num_emotions: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transformer encoder
        self.encoder = ModularTransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Classification heads
        self.genre_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_genres)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_emotions)
        )
        
        # Initialize classification heads
        self._init_classification_heads()
    
    def _init_classification_heads(self):
        """Initialize classification head weights"""
        for module in [self.genre_classifier, self.emotion_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input features [batch, seq_len, input_dim]
            mask: Optional attention mask
        
        Returns:
            Dictionary with genre_logits and emotion_logits
        """
        # Encode features
        encoded = self.encoder(x, mask)  # [batch, embed_dim]
        
        # Classifications
        genre_logits = self.genre_classifier(encoded)
        emotion_logits = self.emotion_classifier(encoded)
        
        return {
            "genre_logits": genre_logits,
            "emotion_logits": emotion_logits,
            "embeddings": encoded
        }



