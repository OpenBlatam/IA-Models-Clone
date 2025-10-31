from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
"""
Attention Mechanisms and Positional Encodings for HeyGen AI.

Implementation of various attention mechanisms and positional encodings
following PEP 8 style guidelines and best practices.
"""


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformers."""

    def __init__(self, embedding_dimension: int, max_sequence_length: int = 5000, dropout_probability: float = 0.1):
        """Initialize positional encoding.

        Args:
            embedding_dimension: Dimension of the embeddings.
            max_sequence_length: Maximum sequence length for positional encoding.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # Create positional encoding matrix
        positional_encoding_matrix = torch.zeros(max_sequence_length, embedding_dimension)
        position_indices = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        
        # Calculate division terms for sinusoidal encoding
        division_terms = torch.exp(torch.arange(0, embedding_dimension, 2).float() * 
                                 -(math.log(10000.0) / embedding_dimension))
        
        # Apply sinusoidal encoding
        positional_encoding_matrix[:, 0::2] = torch.sin(position_indices * division_terms)
        positional_encoding_matrix[:, 1::2] = torch.cos(position_indices * division_terms)
        
        # Register as buffer (not a parameter)
        self.register_buffer('positional_encoding', positional_encoding_matrix.unsqueeze(0))

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input embeddings.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Embeddings with positional encoding added.
        """
        sequence_length = input_embeddings.size(1)
        positional_encodings = self.positional_encoding[:, :sequence_length, :]
        encoded_embeddings = input_embeddings + positional_encodings
        return self.dropout(encoded_embeddings)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for transformers."""

    def __init__(self, embedding_dimension: int, max_sequence_length: int = 5000, dropout_probability: float = 0.1):
        """Initialize learned positional encoding.

        Args:
            embedding_dimension: Dimension of the embeddings.
            max_sequence_length: Maximum sequence length for positional encoding.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, max_sequence_length, embedding_dimension))

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply learned positional encoding to input embeddings.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Embeddings with learned positional encoding added.
        """
        sequence_length = input_embeddings.size(1)
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds maximum {self.max_sequence_length}")
        
        positional_encodings = self.positional_embeddings[:, :sequence_length, :]
        encoded_embeddings = input_embeddings + positional_encodings
        return self.dropout(encoded_embeddings)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformers."""

    def __init__(self, embedding_dimension: int, max_relative_position: int = 32, dropout_probability: float = 0.1):
        """Initialize relative positional encoding.

        Args:
            embedding_dimension: Dimension of the embeddings.
            max_relative_position: Maximum relative position distance.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, embedding_dimension)
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply relative positional encoding to input embeddings.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Embeddings with relative positional encoding added.
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.size()
        
        # Create relative position indices
        position_indices = torch.arange(sequence_length, device=input_embeddings.device)
        relative_positions = position_indices.unsqueeze(1) - position_indices.unsqueeze(0)
        
        # Clip relative positions to valid range
        relative_positions = torch.clamp(relative_positions, -self.max_relative_position, self.max_relative_position)
        relative_positions += self.max_relative_position
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings[relative_positions]
        
        # Add to input embeddings
        encoded_embeddings = input_embeddings + relative_embeddings
        return self.dropout(encoded_embeddings)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        dropout_probability: float = 0.1,
        bias: bool = True,
        attention_type: str = "scaled_dot_product"
    ):
        """Initialize multi-head attention.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            dropout_probability: Dropout probability.
            bias: Whether to use bias in linear layers.
            attention_type: Type of attention mechanism.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.attention_type = attention_type
        
        # Ensure embedding dimension is divisible by number of heads
        if embedding_dimension % num_attention_heads != 0:
            raise ValueError(f"Embedding dimension {embedding_dimension} must be divisible by number of heads {num_attention_heads}")
        
        self.head_dimension = embedding_dimension // num_attention_heads
        self.scaling_factor = math.sqrt(self.head_dimension)
        
        # Linear projections for query, key, value, and output
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.output_dropout = nn.Dropout(p=dropout_probability)

    def _reshape_for_attention(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention.

        Args:
            input_tensor: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_heads, sequence_length, head_dim).
        """
        batch_size, sequence_length, embedding_dim = input_tensor.size()
        return input_tensor.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)

    def _compute_attention_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute attention scores between query and key.

        Args:
            query: Query tensor of shape (batch_size, num_heads, query_length, head_dim).
            key: Key tensor of shape (batch_size, num_heads, key_length, head_dim).

        Returns:
            torch.Tensor: Attention scores of shape (batch_size, num_heads, query_length, key_length).
        """
        if self.attention_type == "scaled_dot_product":
            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor
        elif self.attention_type == "additive":
            # Additive attention
            attention_scores = torch.tanh(query.unsqueeze(-2) + key.unsqueeze(-3))
            attention_scores = attention_scores.sum(dim=-1)
        elif self.attention_type == "multiplicative":
            # Multiplicative attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1))
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        return attention_scores

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, query_length, embedding_dim).
            key: Key tensor of shape (batch_size, key_length, embedding_dim).
            value: Value tensor of shape (batch_size, value_length, embedding_dim).
            attention_mask: Attention mask of shape (batch_size, query_length, key_length).
            key_padding_mask: Key padding mask of shape (batch_size, key_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size, query_length, embedding_dim = query.size()
        key_length = key.size(1)
        value_length = value.size(1)
        
        # Linear projections
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        
        # Reshape for multi-head attention
        query_reshaped = self._reshape_for_attention(query_projected)
        key_reshaped = self._reshape_for_attention(key_projected)
        value_reshaped = self._reshape_for_attention(value_projected)
        
        # Compute attention scores
        attention_scores = self._compute_attention_scores(query_reshaped, key_reshaped)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        context_vectors = torch.matmul(attention_weights, value_reshaped)
        
        # Reshape back to original dimensions
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, query_length, embedding_dim
        )
        
        # Final linear projection
        output = self.output_projection(context_vectors)
        output = self.output_dropout(output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """Self-attention mechanism."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        dropout_probability: float = 0.1,
        bias: bool = True
    ):
        """Initialize self-attention.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            dropout_probability: Dropout probability.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension,
            num_attention_heads=num_attention_heads,
            dropout_probability=dropout_probability,
            bias=bias
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of self-attention.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).
            attention_mask: Attention mask of shape (batch_size, sequence_length, sequence_length).
            padding_mask: Padding mask of shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        return self.multi_head_attention(
            query=input_embeddings,
            key=input_embeddings,
            value=input_embeddings,
            attention_mask=attention_mask,
            key_padding_mask=padding_mask
        )


class CrossAttention(nn.Module):
    """Cross-attention mechanism."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        dropout_probability: float = 0.1,
        bias: bool = True
    ):
        """Initialize cross-attention.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            dropout_probability: Dropout probability.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension,
            num_attention_heads=num_attention_heads,
            dropout_probability=dropout_probability,
            bias=bias
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of cross-attention.

        Args:
            query: Query tensor of shape (batch_size, query_length, embedding_dim).
            key: Key tensor of shape (batch_size, key_length, embedding_dim).
            value: Value tensor of shape (batch_size, value_length, embedding_dim).
            attention_mask: Attention mask of shape (batch_size, query_length, key_length).
            key_padding_mask: Key padding mask of shape (batch_size, key_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        return self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )


class LocalAttention(nn.Module):
    """Local attention mechanism with sliding window."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        window_size: int = 128,
        dropout_probability: float = 0.1,
        bias: bool = True
    ):
        """Initialize local attention.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            window_size: Size of the local attention window.
            dropout_probability: Dropout probability.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.head_dimension = embedding_dimension // num_attention_heads
        self.scaling_factor = math.sqrt(self.head_dimension)
        
        # Linear projections
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.output_dropout = nn.Dropout(p=dropout_probability)

    def _create_local_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask.

        Args:
            sequence_length: Length of the sequence.
            device: Device to create mask on.

        Returns:
            torch.Tensor: Local attention mask.
        """
        mask = torch.zeros(sequence_length, sequence_length, device=device)
        for i in range(sequence_length):
            start = max(0, i - self.window_size // 2)
            end = min(sequence_length, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of local attention.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).
            attention_mask: Additional attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.size()
        
        # Linear projections
        query = self.query_projection(input_embeddings)
        key = self.key_projection(input_embeddings)
        value = self.value_projection(input_embeddings)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor
        
        # Create local attention mask
        local_mask = self._create_local_mask(sequence_length, input_embeddings.device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply local mask
        attention_scores = attention_scores.masked_fill(local_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        context_vectors = torch.matmul(attention_weights, value)
        
        # Reshape back to original dimensions
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        
        # Final linear projection
        output = self.output_projection(context_vectors)
        output = self.output_dropout(output)
        
        return output, attention_weights


class SparseAttention(nn.Module):
    """Sparse attention mechanism with fixed attention patterns."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        num_attention_patterns: int = 8,
        dropout_probability: float = 0.1,
        bias: bool = True
    ):
        """Initialize sparse attention.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            num_attention_patterns: Number of attention patterns.
            dropout_probability: Dropout probability.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.num_attention_patterns = num_attention_patterns
        self.head_dimension = embedding_dimension // num_attention_heads
        self.scaling_factor = math.sqrt(self.head_dimension)
        
        # Linear projections
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        
        # Attention pattern parameters
        self.attention_patterns = nn.Parameter(torch.randn(num_attention_patterns, embedding_dimension))
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.output_dropout = nn.Dropout(p=dropout_probability)

    def _create_sparse_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask.

        Args:
            sequence_length: Length of the sequence.
            device: Device to create mask on.

        Returns:
            torch.Tensor: Sparse attention mask.
        """
        # Create random sparse attention patterns
        mask = torch.zeros(sequence_length, sequence_length, device=device)
        
        for i in range(sequence_length):
            # Select random positions for each token
            num_connections = max(1, sequence_length // 8)  # Sparse connectivity
            connections = torch.randperm(sequence_length, device=device)[:num_connections]
            mask[i, connections] = 1
        
        return mask

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sparse attention.

        Args:
            input_embeddings: Input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).
            attention_mask: Additional attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.size()
        
        # Linear projections
        query = self.query_projection(input_embeddings)
        key = self.key_projection(input_embeddings)
        value = self.value_projection(input_embeddings)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.head_dimension).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor
        
        # Create sparse attention mask
        sparse_mask = self._create_sparse_mask(sequence_length, input_embeddings.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply sparse mask
        attention_scores = attention_scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        context_vectors = torch.matmul(attention_weights, value)
        
        # Reshape back to original dimensions
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        
        # Final linear projection
        output = self.output_projection(context_vectors)
        output = self.output_dropout(output)
        
        return output, attention_weights


class AttentionBlock(nn.Module):
    """Complete attention block with residual connection and layer normalization."""

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        dropout_probability: float = 0.1,
        attention_type: str = "self",
        bias: bool = True
    ):
        """Initialize attention block.

        Args:
            embedding_dimension: Dimension of the embeddings.
            num_attention_heads: Number of attention heads.
            dropout_probability: Dropout probability.
            attention_type: Type of attention ("self", "cross", "local", "sparse").
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_type = attention_type
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        
        # Attention mechanism
        if attention_type == "self":
            self.attention = SelfAttention(
                embedding_dimension=embedding_dimension,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                bias=bias
            )
        elif attention_type == "cross":
            self.attention = CrossAttention(
                embedding_dimension=embedding_dimension,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                bias=bias
            )
        elif attention_type == "local":
            self.attention = LocalAttention(
                embedding_dimension=embedding_dimension,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                bias=bias
            )
        elif attention_type == "sparse":
            self.attention = SparseAttention(
                embedding_dimension=embedding_dimension,
                num_attention_heads=num_attention_heads,
                dropout_probability=dropout_probability,
                bias=bias
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(
        self,
        input_embeddings: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of attention block.

        Args:
            input_embeddings: Input embeddings tensor.
            context_embeddings: Context embeddings tensor (for cross-attention).
            attention_mask: Attention mask.
            padding_mask: Padding mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        # Layer normalization
        normalized_embeddings = self.layer_norm(input_embeddings)
        
        # Apply attention
        if self.attention_type == "self":
            output, attention_weights = self.attention(
                normalized_embeddings,
                attention_mask=attention_mask,
                padding_mask=padding_mask
            )
        elif self.attention_type == "cross":
            if context_embeddings is None:
                raise ValueError("Context embeddings required for cross-attention")
            normalized_context = self.layer_norm(context_embeddings)
            output, attention_weights = self.attention(
                normalized_embeddings,
                normalized_context,
                normalized_context,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask
            )
        else:
            output, attention_weights = self.attention(
                normalized_embeddings,
                attention_mask=attention_mask
            )
        
        # Residual connection
        output = input_embeddings + output
        
        return output, attention_weights


def create_attention_mask(
    sequence_length: int,
    attention_type: str = "causal",
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create attention mask.

    Args:
        sequence_length: Length of the sequence.
        attention_type: Type of attention mask ("causal", "full", "local").
        device: Device to create mask on.

    Returns:
        torch.Tensor: Attention mask.
    """
    if device is None:
        device = torch.device("cpu")
    
    if attention_type == "causal":
        # Causal mask (lower triangular)
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, 0.0)
    elif attention_type == "full":
        # Full attention mask (all ones)
        mask = torch.ones(sequence_length, sequence_length, device=device)
    elif attention_type == "local":
        # Local attention mask
        mask = torch.zeros(sequence_length, sequence_length, device=device)
        window_size = min(64, sequence_length // 4)
        for i in range(sequence_length):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            mask[i, start:end] = 1
    else:
        raise ValueError(f"Unknown attention mask type: {attention_type}")
    
    return mask


def create_padding_mask(
    input_ids: torch.Tensor,
    padding_token_id: int = 0
) -> torch.Tensor:
    """Create padding mask from input IDs.

    Args:
        input_ids: Input token IDs of shape (batch_size, sequence_length).
        padding_token_id: ID of the padding token.

    Returns:
        torch.Tensor: Padding mask of shape (batch_size, sequence_length).
    """
    return (input_ids == padding_token_id)


def apply_attention_mask(
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """Apply attention mask to attention scores.

    Args:
        attention_scores: Attention scores tensor.
        attention_mask: Attention mask tensor.

    Returns:
        torch.Tensor: Masked attention scores.
    """
    return attention_scores.masked_fill(attention_mask == 0, float('-inf'))


def compute_attention_weights(
    attention_scores: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Compute attention weights from attention scores.

    Args:
        attention_scores: Attention scores tensor.
        temperature: Temperature for softmax.

    Returns:
        torch.Tensor: Attention weights.
    """
    return F.softmax(attention_scores / temperature, dim=-1)


def apply_attention_weights(
    attention_weights: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """Apply attention weights to values.

    Args:
        attention_weights: Attention weights tensor.
        value: Value tensor.

    Returns:
        torch.Tensor: Weighted values.
    """
    return torch.matmul(attention_weights, value) 