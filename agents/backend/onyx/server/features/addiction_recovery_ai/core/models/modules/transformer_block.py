"""
Transformer Blocks
Reusable transformer building blocks
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention import SelfAttention


class TransformerBlock(nn.Module):
    """
    Standard transformer block
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize transformer block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension (default: 4 * embed_dim)
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        ffn_dim = ffn_dim or (4 * embed_dim)
        
        # Self-attention
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual
        x = self.attention(x, mask)
        
        # Feed-forward with residual
        x = self.norm(x + self.ffn(x))
        
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block for transformer
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize encoder block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of heads
            ffn_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.block = TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        return self.block(x, mask)


class DecoderBlock(nn.Module):
    """
    Decoder block for transformer
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize decoder block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of heads
            ffn_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        from .attention import SelfAttention, CrossAttention
        
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        
        ffn_dim = ffn_dim or (4 * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Decoder input
            encoder_output: Encoder output
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention
        x = self.norm1(x + self.self_attn(x, mask))
        
        # Cross-attention
        x = self.norm2(x + self.cross_attn(x, encoder_output, encoder_output, mask))
        
        # Feed-forward
        x = self.norm3(x + self.ffn(x))
        
        return x













