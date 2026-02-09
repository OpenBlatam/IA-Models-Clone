"""
Enhanced Music Generation Model Architecture

Implements:
- Custom nn.Module with proper architecture
- Weight initialization best practices
- Attention mechanisms
- Positional encodings
- Proper normalization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
import math

from .layers import PositionalEncoding, TransformerBlock
from .initialization import initialize_weights

logger = logging.getLogger(__name__)


class EnhancedMusicModel(nn.Module):
    """
    Enhanced music generation model with proper architecture.
    
    Features:
    - Custom nn.Module implementation
    - Proper weight initialization
    - Transformer-based architecture
    - Layer normalization
    - Dropout for regularization
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True
    ):
        """
        Initialize enhanced music model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = None
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using best practices.
        
        Uses the centralized initialization function from .initialization module.
        """
        self.apply(initialize_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary with logits and optionally attention weights
        """
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        else:
            x = self.dropout(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1)),
                device=input_ids.device,
                dtype=torch.bool
            )
        
        # Expand mask for multi-head attention
        # Shape: (batch_size, 1, 1, seq_len)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply transformer blocks
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            if return_attention_weights:
                # Store attention weights if requested
                _, attn_weights = transformer_block.attention(
                    x, x, x, extended_mask
                )
                attention_weights_list.append(attn_weights)
            x = transformer_block(x, extended_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        result = {'logits': logits}
        
        if return_attention_weights:
            result['attention_weights'] = attention_weights_list
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Initial input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for padding token (stop condition)
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return generated


def create_enhanced_music_model(
    vocab_size: int = 32000,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    **kwargs
) -> EnhancedMusicModel:
    """
    Factory function to create enhanced music model.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        **kwargs: Additional arguments
        
    Returns:
        Enhanced music model instance
    """
    return EnhancedMusicModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        **kwargs
    )

