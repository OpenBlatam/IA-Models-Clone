"""
Transformer Music Encoder Module

Implements transformer encoder for music feature encoding.
"""

from typing import Optional, List
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


class TransformerMusicEncoder(nn.Module):
    """
    Transformer encoder for music feature encoding.
    
    Args:
        input_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        ff_dim: Feedforward dimension.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = 1000
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.empty(1, self.max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "attn_norm": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "ff_norm": nn.LayerNorm(embed_dim),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, embed_dim),
                    nn.Dropout(dropout)
                )
            })
            self.layers.append(layer)
        
        # Final layer norm and output projection
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self._initialize_weights()
        logger.debug(f"Initialized TransformerMusicEncoder with embed_dim={embed_dim}, num_layers={num_layers}")
    
    def _initialize_weights(self):
        """Initialize transformer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            Encoded features [batch_size, embed_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds max {self.max_seq_len}, truncating")
            seq_len = self.max_seq_len
            x = x[:, :seq_len, :]
        
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Attention with residual (pre-norm)
            residual = x
            x = layer["attn_norm"](x)
            attn_out, _ = layer["attn"](x, x, x, key_padding_mask=mask)
            x = attn_out + residual
            
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf in attention layer {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Feed-forward with residual (pre-norm)
            residual = x
            x = layer["ff_norm"](x)
            x = layer["ff"](x)
            x = x + residual
            
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf in FF layer {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Global average pooling (with mask support if needed)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
            masked_sum = (x * mask_expanded).sum(dim=1)  # [batch, embed_dim]
            mask_count = mask_expanded.sum(dim=1)  # [batch, 1]
            x = masked_sum / (mask_count + 1e-8)
        else:
            x = x.mean(dim=1)  # [batch, embed_dim]
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in final output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x

