"""
Enhanced Transformer Core Components Module

This module contains the core transformer architecture components
including attention mechanisms, positional encodings, and transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from .transformer_config import TransformerConfig


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) implementation."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply rotary positional encoding."""
        seq_len = x.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Create position embeddings
        freqs = torch.outer(position_ids.float().squeeze(0), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        x_rot = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rot = x_rot.unsqueeze(-1).repeat(1, 1, 1, 2).flatten(-2)
        
        return x * cos + x_rot * sin


class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding implementation."""
    
    def __init__(self, hidden_size: int, max_relative_position: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relative_position = max_relative_position
        
        self.relative_position_embedding = nn.Embedding(
            2 * max_relative_position + 1, hidden_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relative positional encoding."""
        seq_len = x.size(1)
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Convert to positive indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        relative_position_embeddings = self.relative_position_embedding(final_mat)
        
        return x + relative_position_embeddings


class PositionalEncoding(nn.Module):
    """Standard positional encoding implementation."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding."""
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with various optimizations."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Attention layer
        self.attention = MultiHeadAttention(config)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for efficient fine-tuning."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int = 16, 
                 alpha: float = 32.0,
                 dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of LoRA layer."""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class CustomTransformerModel(nn.Module):
    """Custom transformer model with enhanced features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionalEncoding(config.hidden_size, config.max_position_embeddings)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # LoRA layers (if enabled)
        if config.enable_lora:
            self.lora_layers = nn.ModuleList([
                LoRALayer(config.hidden_size, config.hidden_size, config.lora_rank, config.lora_alpha, config.lora_dropout)
                for _ in range(config.num_layers)
            ])
        else:
            self.lora_layers = None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.config.initializer_range)
        
        # Output projection
        nn.init.normal_(self.lm_head.weight, mean=0, std=self.config.initializer_range)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the transformer model."""
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Position embeddings
        position_embeddings = self.position_embedding(token_embeddings)
        
        # Combine embeddings
        hidden_states = self.dropout(token_embeddings + position_embeddings)
        
        # Pass through transformer blocks
        attention_weights = []
        for i, transformer_block in enumerate(self.transformer_blocks):
            hidden_states = transformer_block(hidden_states, attention_mask)
            
            # Apply LoRA if enabled
            if self.lora_layers is not None:
                lora_output = self.lora_layers[i](hidden_states)
                hidden_states = hidden_states + lora_output
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights
        }
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        total_params = self.get_model_size()
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
        
        return {
            'total_parameters': total_params,
            'memory_mb': memory_mb,
            'memory_gb': memory_mb / 1024
        }


