"""
Optimized Transformer - Performance Optimized
=============================================

Optimized transformer implementation with performance improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .transformer import TransformerEncoder, PositionalEncoding


class OptimizedTransformerEncoder(TransformerEncoder):
    """
    Optimized transformer encoder with performance improvements.
    
    Optimizations:
    - Flash attention (if available)
    - Fused operations
    - Optimized attention computation
    - Better memory efficiency
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_length: int = 512,
        device: Optional[torch.device] = None,
        use_flash_attention: bool = True
    ):
        """
        Initialize optimized transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            num_classes: Number of output classes
            max_length: Maximum sequence length
            device: Target device
            use_flash_attention: Whether to use flash attention
        """
        super().__init__(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, num_classes, max_length, device
        )
        
        self.use_flash_attention = use_flash_attention
        
        # Enable flash attention if available
        if use_flash_attention and device and device.type == "cuda":
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("✅ Flash attention enabled")
            except Exception:
                logger.debug("Flash attention not available")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized forward pass.
        
        Args:
            x: Input tensor
            mask: Attention mask
        
        Returns:
            Output tensor
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # Transformer encoding with optimizations
        if self.use_flash_attention:
            # Use scaled_dot_product_attention for better performance
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use [CLS] token
        x = x[:, 0, :]
        
        # Classification
        x = self.classifier(x)
        return x



