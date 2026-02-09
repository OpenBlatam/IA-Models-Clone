"""
Attention Mechanisms and Positional Encodings System
Comprehensive implementation of attention mechanisms and positional encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms and positional encodings."""
    
    # Model dimensions
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_position_embeddings: int = 512
    vocab_size: int = 50257
    
    # Attention parameters
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    scale_attention_weights: bool = True
    
    # Positional encoding parameters
    positional_encoding_type: str = "sinusoidal"  # sinusoidal, learned, relative
    positional_encoding_dropout: float = 0.1
    
    # Training parameters
    dropout: float = 0.1
    activation_function: str = "gelu"  # gelu, relu, swish


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""
    
    def __init__(self, hidden_size: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate division terms for different frequencies
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(math.log(10000.0) / hidden_size))
        
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, hidden_size)
        
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, hidden_size: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add positional encoding
        x = x + position_embeddings
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer models."""
    
    def __init__(self, hidden_size: int, max_relative_position: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(dropout)
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, hidden_size
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize relative positional embedding weights."""
        nn.init.normal_(self.relative_position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add relative positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Tensor with relative positional encoding added
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Create relative position matrix
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        relative_position_embeddings = self.relative_position_embeddings(final_mat)
        
        # Add to input (simplified version - in practice, this would be integrated into attention)
        x = x + relative_position_embeddings.mean(dim=0).unsqueeze(0)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism implementation."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Ensure hidden size is divisible by number of heads
        assert self.head_dim * config.num_heads == config.hidden_size, \
            f"Hidden size {config.hidden_size} must be divisible by number of heads {config.num_heads}"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5 if config.scale_attention_weights else 1.0
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        # Initialize linear layers
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        
        # Initialize biases
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_length, hidden_size = x.size()
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, seq_length)
            head_mask: Optional head mask for attention heads
            output_attentions: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Linear projections and reshape for multi-head attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match attention scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Apply attention weights to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_size ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_length, hidden_size)
            key: Key tensor of shape (batch_size, seq_length, hidden_size)
            value: Value tensor of shape (batch_size, seq_length, hidden_size)
            mask: Optional attention mask
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through self-attention with residual connection.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        attention_outputs = self.multi_head_attention(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = attention_outputs[0]
        
        # Residual connection and layer normalization
        attention_output = self.layer_norm(hidden_states + attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        return outputs


class CrossAttention(nn.Module):
    """Cross-attention mechanism for encoder-decoder architectures."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through cross-attention.
        
        Args:
            hidden_states: Decoder hidden states
            encoder_hidden_states: Encoder hidden states
            attention_mask: Optional attention mask for decoder
            encoder_attention_mask: Optional attention mask for encoder
            head_mask: Optional head mask
            output_attentions: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Cross-attention: query from decoder, key/value from encoder
        attention_outputs = self.multi_head_attention(
            hidden_states, encoder_attention_mask, head_mask, output_attentions
        )
        attention_output = attention_outputs[0]
        
        # Residual connection and layer normalization
        attention_output = self.layer_norm(hidden_states + attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        return outputs


class AttentionVisualizer:
    """Visualize attention weights and patterns."""
    
    def __init__(self):
        self.attention_weights = []
        self.tokens = []
    
    def capture_attention(self, attention_weights: torch.Tensor, tokens: List[str]):
        """Capture attention weights for visualization."""
        self.attention_weights.append(attention_weights.detach().cpu())
        self.tokens.append(tokens)
    
    def visualize_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Visualize attention weights as a heatmap."""
        # Get attention weights for specific layer and head
        if attention_weights.dim() == 4:  # (batch_size, num_heads, seq_len, seq_len)
            attn_weights = attention_weights[0, head_idx].numpy()
        else:  # (seq_len, seq_len)
            attn_weights = attention_weights.numpy()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            attn_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            annot=False,
            cbar=True
        )
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def visualize_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """Visualize attention weights for all heads in a layer."""
        num_heads = attention_weights.size(1)
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()
        
        for head_idx in range(min(num_heads, 12)):
            attn_weights = attention_weights[0, head_idx].numpy()
            
            sns.heatmap(
                attn_weights,
                xticklabels=tokens if head_idx % 4 == 0 else [],
                yticklabels=tokens if head_idx < 4 else [],
                cmap='Blues',
                ax=axes[head_idx],
                cbar=False
            )
            axes[head_idx].set_title(f'Head {head_idx}')
        
        plt.suptitle(f'Multi-Head Attention - Layer {layer_idx}')
        plt.tight_layout()
        plt.show()
    
    def visualize_positional_encoding(self, pos_encoding: torch.Tensor, figsize: Tuple[int, int] = (12, 8)):
        """Visualize positional encoding patterns."""
        pos_encoding_np = pos_encoding.squeeze().numpy()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            pos_encoding_np,
            cmap='RdBu_r',
            center=0,
            cbar=True
        )
        plt.title('Positional Encoding Patterns')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Position')
        plt.tight_layout()
        plt.show()


class AttentionAnalyzer:
    """Analyze attention patterns and behaviors."""
    
    def __init__(self):
        self.attention_patterns = {}
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str]
    ) -> Dict[str, Any]:
        """Analyze attention patterns in the weights."""
        # Convert to numpy for analysis
        attn_weights = attention_weights.detach().cpu().numpy()
        
        # Calculate attention statistics
        mean_attention = np.mean(attn_weights, axis=(0, 1))  # Average across heads
        max_attention = np.max(attn_weights, axis=(0, 1))
        attention_entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-8), axis=-1)
        mean_entropy = np.mean(attention_entropy, axis=0)
        
        # Find most attended positions
        most_attended = np.argmax(mean_attention, axis=-1)
        
        # Calculate attention diversity (how spread out attention is)
        attention_diversity = 1.0 - np.max(attn_weights, axis=-1)
        mean_diversity = np.mean(attention_diversity, axis=0)
        
        return {
            'mean_attention': mean_attention,
            'max_attention': max_attention,
            'attention_entropy': mean_entropy,
            'most_attended_positions': most_attended,
            'attention_diversity': mean_diversity,
            'tokens': tokens
        }
    
    def analyze_positional_bias(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Analyze positional bias in attention patterns."""
        attn_weights = attention_weights.detach().cpu().numpy()
        seq_len = attn_weights.shape[-1]
        
        # Calculate distance-based attention
        distances = np.abs(np.arange(seq_len).reshape(-1, 1) - np.arange(seq_len))
        
        # Average attention for each distance
        distance_attention = {}
        for dist in range(seq_len):
            mask = (distances == dist)
            if mask.any():
                distance_attention[dist] = np.mean(attn_weights[mask])
        
        # Calculate local vs global attention ratio
        local_mask = distances <= 3  # Local attention (distance <= 3)
        global_mask = distances > 3   # Global attention (distance > 3)
        
        local_attention = np.mean(attn_weights[local_mask]) if local_mask.any() else 0.0
        global_attention = np.mean(attn_weights[global_mask]) if global_mask.any() else 0.0
        
        return {
            'distance_attention': distance_attention,
            'local_attention_ratio': local_attention / (local_attention + global_attention + 1e-8),
            'global_attention_ratio': global_attention / (local_attention + global_attention + 1e-8)
        }
    
    def print_attention_analysis(self, analysis_results: Dict[str, Any]):
        """Print attention analysis results."""
        print("\n=== Attention Analysis Results ===")
        
        # Print attention statistics
        print(f"Mean attention entropy: {np.mean(analysis_results['attention_entropy']):.4f}")
        print(f"Mean attention diversity: {np.mean(analysis_results['attention_diversity']):.4f}")
        
        # Print most attended positions
        print("\nMost attended positions:")
        for i, pos in enumerate(analysis_results['most_attended_positions'][:10]):
            token = analysis_results['tokens'][pos] if pos < len(analysis_results['tokens']) else f"pos_{pos}"
            print(f"  Position {i}: {token} (pos {pos})")


class TransformerBlock(nn.Module):
    """Complete transformer block with attention and feed-forward."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = SelfAttention(config)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU() if config.activation_function == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through transformer block."""
        # Self-attention
        attention_outputs = self.self_attention(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = attention_outputs[0]
        
        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(attention_output)
        output = self.final_layer_norm(attention_output + feed_forward_output)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        return outputs


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Attention Mechanisms and Positional Encodings Demonstration ===\n")
    
    # Configuration
    config = AttentionConfig(
        hidden_size=768,
        num_heads=12,
        max_position_embeddings=512,
        positional_encoding_type="sinusoidal"
    )
    
    # 1. Test different positional encodings
    print("1. Testing Positional Encodings...")
    
    # Sinusoidal positional encoding
    sinusoidal_pe = SinusoidalPositionalEncoding(768, 512)
    test_input = torch.randn(2, 10, 768)  # (batch_size, seq_len, hidden_size)
    sinusoidal_output = sinusoidal_pe(test_input.transpose(0, 1)).transpose(0, 1)
    print(f"Sinusoidal PE output shape: {sinusoidal_output.shape}")
    
    # Learned positional encoding
    learned_pe = LearnedPositionalEncoding(768, 512)
    learned_output = learned_pe(test_input)
    print(f"Learned PE output shape: {learned_output.shape}")
    
    # Relative positional encoding
    relative_pe = RelativePositionalEncoding(768, 32)
    relative_output = relative_pe(test_input)
    print(f"Relative PE output shape: {relative_output.shape}")
    
    # 2. Test attention mechanisms
    print("\n2. Testing Attention Mechanisms...")
    
    # Multi-head attention
    multi_head_attn = MultiHeadAttention(config)
    attention_output, attention_weights = multi_head_attn(
        test_input, output_attentions=True
    )
    print(f"Multi-head attention output shape: {attention_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Scaled dot-product attention
    scaled_attn = ScaledDotProductAttention(768)
    query = torch.randn(2, 10, 768)
    key = torch.randn(2, 10, 768)
    value = torch.randn(2, 10, 768)
    scaled_output, scaled_weights = scaled_attn(query, key, value)
    print(f"Scaled dot-product attention output shape: {scaled_output.shape}")
    
    # 3. Test complete transformer block
    print("\n3. Testing Transformer Block...")
    
    transformer_block = TransformerBlock(config)
    block_output, block_attention = transformer_block(
        test_input, output_attentions=True
    )
    print(f"Transformer block output shape: {block_output.shape}")
    
    # 4. Visualize attention patterns
    print("\n4. Visualizing Attention Patterns...")
    
    # Sample text for visualization
    sample_text = "The transformer model uses attention mechanisms effectively"
    tokens = sample_text.split()
    
    # Create attention visualizer
    visualizer = AttentionVisualizer()
    
    # Visualize attention weights
    if attention_weights is not None:
        visualizer.visualize_attention_heatmap(
            attention_weights, tokens, layer_idx=0, head_idx=0
        )
        
        # Visualize multi-head attention
        visualizer.visualize_multi_head_attention(
            attention_weights, tokens, layer_idx=0
        )
    
    # 5. Analyze attention patterns
    print("\n5. Analyzing Attention Patterns...")
    
    analyzer = AttentionAnalyzer()
    
    if attention_weights is not None:
        # Analyze attention patterns
        analysis_results = analyzer.analyze_attention_patterns(attention_weights, tokens)
        analyzer.print_attention_analysis(analysis_results)
        
        # Analyze positional bias
        positional_bias = analyzer.analyze_positional_bias(attention_weights)
        print(f"\nPositional bias analysis:")
        print(f"Local attention ratio: {positional_bias['local_attention_ratio']:.4f}")
        print(f"Global attention ratio: {positional_bias['global_attention_ratio']:.4f}")
    
    # 6. Visualize positional encoding
    print("\n6. Visualizing Positional Encoding...")
    
    # Create sample positional encoding for visualization
    pos_encoding = sinusoidal_pe.pe[:50, :]  # First 50 positions
    visualizer.visualize_positional_encoding(pos_encoding)
    
    print("\n=== Demonstration Completed Successfully! ===")





