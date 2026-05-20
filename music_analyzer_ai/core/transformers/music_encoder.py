"""
Music Transformer Encoder Module

Advanced transformer encoder for music features.
"""

from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("PyTorch not available")


class MusicTransformerEncoder:
    """
    Advanced transformer encoder for music features.
    
    Args:
        input_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        ff_dim: Feedforward dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required")
        
        from ..models.transformer_encoder import TransformerMusicEncoder
        
        self.encoder = TransformerMusicEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.max_seq_len = max_seq_len
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode music features using transformer.
        
        Args:
            features: Input features array.
        
        Returns:
            Encoded features array.
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.FloatTensor(features)
        else:
            features_tensor = features
        
        # Add sequence dimension if needed
        if len(features_tensor.shape) == 2:
            features_tensor = features_tensor.unsqueeze(0)
        
        # Encode
        with torch.no_grad():
            encoded = self.encoder(features_tensor)
        
        return encoded.cpu().numpy() if isinstance(encoded, torch.Tensor) else encoded
    
    def get_attention_patterns(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get attention patterns from encoding.
        
        Args:
            features: Input features.
        
        Returns:
            Dictionary with attention pattern information.
        """
        # This would require modifying the encoder to return attention
        # For now, return placeholder
        return {
            "attention_available": False,
            "message": "Attention patterns require model modification"
        }



