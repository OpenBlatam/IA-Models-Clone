from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
"""
Transformers and LLMs for HeyGen AI.

Advanced transformer architectures and Large Language Models (LLMs) implementation
following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_attention_heads: int,
        dropout_probability: float = 0.1,
        bias: bool = True,
        use_relative_position: bool = False,
        max_relative_position: int = 512
    ):
        """Initialize multi-head attention.

        Args:
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
            use_relative_position: Whether to use relative position encoding.
            max_relative_position: Maximum relative position.
        """
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_attention_heads = num_attention_heads
        self.dropout_probability = dropout_probability
        self.use_relative_position = use_relative_position
        
        assert embedding_dimensions % num_attention_heads == 0, (
            f"embedding_dimensions ({embedding_dimensions}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
        
        self.head_dimensions = embedding_dimensions // num_attention_heads
        self.scaling_factor = self.head_dimensions ** -0.5
        
        # Linear projections
        self.query_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.key_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.value_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        self.output_projection = nn.Linear(embedding_dimensions, embedding_dimensions, bias=bias)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout_probability)
        self.output_dropout = nn.Dropout(dropout_probability)
        
        # Relative position encoding
        if use_relative_position:
            self.relative_position_embeddings = nn.Parameter(
                torch.randn(2 * max_relative_position + 1, self.head_dimensions)
            )
            self.relative_position_bias = nn.Parameter(
                torch.randn(2 * max_relative_position + 1, num_attention_heads)
            )
            self.max_relative_position = max_relative_position
        else:
            self.relative_position_embeddings = None
            self.relative_position_bias = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embedding_dim).
            key: Key tensor of shape (batch_size, seq_len, embedding_dim).
            value: Value tensor of shape (batch_size, seq_len, embedding_dim).
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len).
            key_padding_mask: Key padding mask of shape (batch_size, seq_len).
            causal_mask: Whether to apply causal masking.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size, query_seq_len, embedding_dim = query.shape
        key_seq_len = key.shape[1]
        
        # Linear projections and reshape
        query = self.query_projection(query).view(
            batch_size, query_seq_len, self.num_attention_heads, self.head_dimensions
        ).transpose(1, 2)
        
        key = self.key_projection(key).view(
            batch_size, key_seq_len, self.num_attention_heads, self.head_dimensions
        ).transpose(1, 2)
        
        value = self.value_projection(value).view(
            batch_size, key_seq_len, self.num_attention_heads, self.head_dimensions
        ).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling_factor
        
        # Add relative position bias if enabled
        if self.use_relative_position:
            relative_position_bias = self._compute_relative_position_bias(
                query_seq_len, key_seq_len
            )
            attention_scores = attention_scores + relative_position_bias
        
        # Apply attention masks
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))
        
        if causal_mask:
            causal_mask = self._create_causal_mask(query_seq_len, key_seq_len, query.device)
            attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_seq_len, embedding_dim
        )
        output = self.output_projection(output)
        output = self.output_dropout(output)
        
        return output, attention_weights

    def _compute_relative_position_bias(
        self, query_seq_len: int, key_seq_len: int
    ) -> torch.Tensor:
        """Compute relative position bias.

        Args:
            query_seq_len: Query sequence length.
            key_seq_len: Key sequence length.

        Returns:
            torch.Tensor: Relative position bias.
        """
        # Compute relative positions
        query_positions = torch.arange(query_seq_len, device=self.relative_position_embeddings.device)
        key_positions = torch.arange(key_seq_len, device=self.relative_position_embeddings.device)
        
        relative_positions = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        relative_positions = relative_positions.clamp(-self.max_relative_position, self.max_relative_position)
        relative_positions = relative_positions + self.max_relative_position
        
        # Get relative position embeddings and bias
        relative_position_embeddings = self.relative_position_embeddings[relative_positions]
        relative_position_bias = self.relative_position_bias[relative_positions]
        
        return relative_position_bias.unsqueeze(0)

    def _create_causal_mask(
        self, query_seq_len: int, key_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal mask for autoregressive attention.

        Args:
            query_seq_len: Query sequence length.
            key_seq_len: Key sequence length.
            device: Device to create mask on.

        Returns:
            torch.Tensor: Causal mask.
        """
        mask = torch.triu(torch.ones(query_seq_len, key_seq_len, device=device), diagonal=1)
        return mask.bool()


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_attention_heads: int,
        feed_forward_dimensions: int,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        use_cross_attention: bool = False,
        activation_function: str = "gelu"
    ):
        """Initialize transformer block.

        Args:
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            feed_forward_dimensions: Dimension of feed-forward network.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            use_cross_attention: Whether to use cross-attention.
            activation_function: Activation function for feed-forward network.
        """
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_attention_heads = num_attention_heads
        self.feed_forward_dimensions = feed_forward_dimensions
        self.dropout_probability = dropout_probability
        self.use_cross_attention = use_cross_attention
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            dropout_probability=dropout_probability,
            use_relative_position=use_relative_position
        )
        
        # Cross-attention (if enabled)
        if use_cross_attention:
            self.cross_attention = MultiHeadAttention(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                use_relative_position=use_relative_position
            )
            self.cross_attention_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # Layer normalization
        self.self_attention_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimensions, feed_forward_dimensions),
            self._get_activation_function(activation_function),
            nn.Dropout(dropout_probability),
            nn.Linear(feed_forward_dimensions, embedding_dimensions),
            nn.Dropout(dropout_probability)
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cross_attention_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, embedding_dim).
            cross_attention_input: Cross-attention input tensor.
            attention_mask: Attention mask for self-attention.
            cross_attention_mask: Attention mask for cross-attention.
            causal_mask: Whether to apply causal masking.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Self-attention
        residual = input_tensor
        input_tensor = self.self_attention_layer_norm(input_tensor)
        attention_output, _ = self.self_attention(
            query=input_tensor,
            key=input_tensor,
            value=input_tensor,
            attention_mask=attention_mask,
            causal_mask=causal_mask
        )
        input_tensor = residual + attention_output
        
        # Cross-attention (if enabled)
        if self.use_cross_attention and cross_attention_input is not None:
            residual = input_tensor
            input_tensor = self.cross_attention_layer_norm(input_tensor)
            cross_attention_output, _ = self.cross_attention(
                query=input_tensor,
                key=cross_attention_input,
                value=cross_attention_input,
                attention_mask=cross_attention_mask
            )
            input_tensor = residual + cross_attention_output
        
        # Feed-forward network
        residual = input_tensor
        input_tensor = self.feed_forward_layer_norm(input_tensor)
        feed_forward_output = self.feed_forward(input_tensor)
        input_tensor = residual + feed_forward_output
        
        return input_tensor

    def _get_activation_function(self, activation_function: str) -> nn.Module:
        """Get activation function.

        Args:
            activation_function: Name of activation function.

        Returns:
            nn.Module: Activation function.
        """
        if activation_function.lower() == "gelu":
            return nn.GELU()
        elif activation_function.lower() == "relu":
            return nn.ReLU()
        elif activation_function.lower() == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(
        self,
        embedding_dimensions: int,
        max_sequence_length: int = 5000,
        dropout_probability: float = 0.1
    ):
        """Initialize positional encoding.

        Args:
            embedding_dimensions: Dimension of embeddings.
            max_sequence_length: Maximum sequence length.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Create positional encoding matrix
        positional_encoding = torch.zeros(max_sequence_length, embedding_dimensions)
        position = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embedding_dimensions, 2).float() *
            -(math.log(10000.0) / embedding_dimensions)
        )
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output tensor with positional encoding.
        """
        return self.dropout(
            input_tensor + self.positional_encoding[:, :input_tensor.size(1)]
        )


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_layers: int,
        feed_forward_dimensions: int,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        activation_function: str = "gelu"
    ):
        """Initialize transformer encoder.

        Args:
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feed_forward_dimensions: Dimension of feed-forward network.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            activation_function: Activation function for feed-forward network.
        """
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_layers = num_layers
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, embedding_dim).
            attention_mask: Attention mask.
            causal_mask: Whether to apply causal masking.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            input_tensor = layer(
                input_tensor=input_tensor,
                attention_mask=attention_mask,
                causal_mask=causal_mask
            )
        
        return self.final_layer_norm(input_tensor)


class TransformerDecoder(nn.Module):
    """Transformer decoder stack."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_layers: int,
        feed_forward_dimensions: int,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        activation_function: str = "gelu"
    ):
        """Initialize transformer decoder.

        Args:
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feed_forward_dimensions: Dimension of feed-forward network.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            activation_function: Activation function for feed-forward network.
        """
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_layers = num_layers
        
        # Transformer blocks with cross-attention
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                use_cross_attention=True,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)

    def forward(
        self,
        input_tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, embedding_dim).
            encoder_output: Encoder output tensor.
            attention_mask: Attention mask for self-attention.
            cross_attention_mask: Attention mask for cross-attention.
            causal_mask: Whether to apply causal masking.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            input_tensor = layer(
                input_tensor=input_tensor,
                cross_attention_input=encoder_output,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                causal_mask=causal_mask
            )
        
        return self.final_layer_norm(input_tensor)


class TransformerModel(nn.Module):
    """Complete transformer model."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        feed_forward_dimensions: int,
        max_sequence_length: int = 5000,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        activation_function: str = "gelu",
        pad_token_id: int = 0
    ):
        """Initialize transformer model.

        Args:
            vocabulary_size: Size of vocabulary.
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            feed_forward_dimensions: Dimension of feed-forward network.
            max_sequence_length: Maximum sequence length.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            activation_function: Activation function for feed-forward network.
            pad_token_id: Padding token ID.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimensions, padding_idx=pad_token_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimensions=embedding_dimensions,
            max_sequence_length=max_sequence_length,
            dropout_probability=dropout_probability
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_encoder_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            dropout_probability=dropout_probability,
            layer_norm_eps=layer_norm_eps,
            use_relative_position=use_relative_position,
            activation_function=activation_function
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_decoder_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            dropout_probability=dropout_probability,
            layer_norm_eps=layer_norm_eps,
            use_relative_position=use_relative_position,
            activation_function=activation_function
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dimensions, vocabulary_size, bias=False)
        
        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            target_ids: Target token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask for input.
            target_attention_mask: Attention mask for target.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocabulary_size).
        """
        # Create attention masks if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        if target_attention_mask is None:
            target_attention_mask = (target_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Create cross-attention mask
        cross_attention_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Encoder
        input_embeddings = self.token_embeddings(input_ids)
        input_embeddings = self.positional_encoding(input_embeddings)
        encoder_output = self.encoder(
            input_tensor=input_embeddings,
            attention_mask=attention_mask
        )
        
        # Decoder
        target_embeddings = self.token_embeddings(target_ids)
        target_embeddings = self.positional_encoding(target_embeddings)
        decoder_output = self.decoder(
            input_tensor=target_embeddings,
            encoder_output=encoder_output,
            attention_mask=target_attention_mask,
            cross_attention_mask=cross_attention_mask,
            causal_mask=True
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits

    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class GPTModel(nn.Module):
    """GPT-style autoregressive language model."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_layers: int,
        feed_forward_dimensions: int,
        max_sequence_length: int = 2048,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        activation_function: str = "gelu",
        pad_token_id: int = 0
    ):
        """Initialize GPT model.

        Args:
            vocabulary_size: Size of vocabulary.
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feed_forward_dimensions: Dimension of feed-forward network.
            max_sequence_length: Maximum sequence length.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            activation_function: Activation function for feed-forward network.
            pad_token_id: Padding token ID.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimensions, padding_idx=pad_token_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimensions=embedding_dimensions,
            max_sequence_length=max_sequence_length,
            dropout_probability=dropout_probability
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dimensions, vocabulary_size, bias=False)
        
        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocabulary_size).
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Token embeddings
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.positional_encoding(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(
                input_tensor=embeddings,
                attention_mask=attention_mask,
                causal_mask=True
            )
        
        # Final layer normalization
        embeddings = self.final_layer_norm(embeddings)
        
        # Output projection
        logits = self.output_projection(embeddings)
        
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Input token IDs.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            do_sample: Whether to use sampling.
            pad_token_id: Padding token ID.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # Initialize output with input
        output = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get logits for next token
                logits = self.forward(output)[:, -1, :]
                
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append next token
                output = torch.cat([output, next_token], dim=-1)
                
                # Check if all sequences are complete
                if (next_token == pad_token_id).all():
                    break
        
        return output

    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class BERTModel(nn.Module):
    """BERT-style bidirectional language model."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_layers: int,
        feed_forward_dimensions: int,
        max_sequence_length: int = 512,
        dropout_probability: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_relative_position: bool = False,
        activation_function: str = "gelu",
        pad_token_id: int = 0,
        mask_token_id: int = 103,
        type_vocabulary_size: int = 2
    ):
        """Initialize BERT model.

        Args:
            vocabulary_size: Size of vocabulary.
            embedding_dimensions: Dimension of embeddings.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            feed_forward_dimensions: Dimension of feed-forward network.
            max_sequence_length: Maximum sequence length.
            dropout_probability: Dropout probability.
            layer_norm_eps: Layer normalization epsilon.
            use_relative_position: Whether to use relative position encoding.
            activation_function: Activation function for feed-forward network.
            pad_token_id: Padding token ID.
            mask_token_id: Mask token ID.
            type_vocabulary_size: Size of token type vocabulary.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimensions, padding_idx=pad_token_id)
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(type_vocabulary_size, embedding_dimensions)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimensions=embedding_dimensions,
            max_sequence_length=max_sequence_length,
            dropout_probability=dropout_probability
        )
        
        # Layer normalization for embeddings
        self.embedding_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(dropout_probability)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dimensions=embedding_dimensions,
                num_attention_heads=num_attention_heads,
                feed_forward_dimensions=feed_forward_dimensions,
                dropout_probability=dropout_probability,
                layer_norm_eps=layer_norm_eps,
                use_relative_position=use_relative_position,
                activation_function=activation_function
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(embedding_dimensions, embedding_dimensions),
            nn.GELU(),
            nn.LayerNorm(embedding_dimensions, eps=layer_norm_eps),
            nn.Linear(embedding_dimensions, vocabulary_size)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            token_type_ids: Token type IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask.
            masked_lm_labels: Labels for masked language modeling.

        Returns:
            Dict[str, torch.Tensor]: Model outputs.
        """
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + token_type_embeddings
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(
                input_tensor=embeddings,
                attention_mask=attention_mask,
                causal_mask=False
            )
        
        # Final layer normalization
        sequence_output = self.final_layer_norm(embeddings)
        
        # MLM predictions
        mlm_logits = self.mlm_head(sequence_output)
        
        outputs = {
            "sequence_output": sequence_output,
            "mlm_logits": mlm_logits
        }
        
        # Compute MLM loss if labels are provided
        if masked_lm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.vocabulary_size),
                masked_lm_labels.view(-1),
                ignore_index=-100
            )
            outputs["mlm_loss"] = mlm_loss
        
        return outputs

    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def create_transformer_model(
    model_type: str,
    vocabulary_size: int,
    embedding_dimensions: int = 768,
    num_attention_heads: int = 12,
    num_layers: int = 12,
    feed_forward_dimensions: int = 3072,
    **kwargs
) -> nn.Module:
    """Create transformer model.

    Args:
        model_type: Type of model ("transformer", "gpt", "bert").
        vocabulary_size: Size of vocabulary.
        embedding_dimensions: Dimension of embeddings.
        num_attention_heads: Number of attention heads.
        num_layers: Number of layers.
        feed_forward_dimensions: Dimension of feed-forward network.
        **kwargs: Additional arguments.

    Returns:
        nn.Module: Created model.

    Raises:
        ValueError: If model type is not supported.
    """
    if model_type.lower() == "transformer":
        return TransformerModel(
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            **kwargs
        )
    elif model_type.lower() == "gpt":
        return GPTModel(
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            **kwargs
        )
    elif model_type.lower() == "bert":
        return BERTModel(
            vocabulary_size=vocabulary_size,
            embedding_dimensions=embedding_dimensions,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            feed_forward_dimensions=feed_forward_dimensions,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 