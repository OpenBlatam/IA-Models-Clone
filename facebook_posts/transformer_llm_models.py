from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path
from deep_learning_models import (
    ModelConfig,
    WeightInitializer,
    NormalizationLayers,
    LossFunctions,
    OptimizerFactory,
    SchedulerFactory,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Dict, Optional
import asyncio
"""
🧠 Advanced Transformers and LLMs for Facebook Posts Processing
==============================================================

This module implements state-of-the-art transformer architectures and Large Language Models
for Facebook Posts analysis, generation, and processing.

Key Features:
- Multi-head attention mechanisms
- Position encoding and embeddings
- Advanced transformer architectures
- LLM training and fine-tuning
- Attention visualization
- Model compression techniques
- Efficient inference
"""


# Import our base deep learning components
    ModelConfig,
    WeightInitializer,
    NormalizationLayers,
    LossFunctions,
    OptimizerFactory,
    SchedulerFactory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


@dataclass
class TransformerConfig:
    """Configuration for transformer models."""
    # Model dimensions
    vocab_size: int = 50000
    max_seq_length: int = 512
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 10000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # Attention parameters
    attention_dropout: float = 0.1
    use_relative_position: bool = True
    max_relative_position: int = 64
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    
    # LLM specific
    use_rope: bool = True  # Rotary Position Embedding
    use_flash_attention: bool = False
    use_group_query_attention: bool = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers."""
    
    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Generate rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = torch.cos(emb)[None, :, None, :]
        sin = torch.sin(emb)[None, :, None, :]
        
        x_rot = torch.cat([-x[..., self.d_model//2:], x[..., :self.d_model//2]], dim=-1)
        return x * cos + x_rot * sin


class MultiHeadAttention(nn.Module):
    """Advanced multi-head attention with various enhancements."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_relative_position: bool = True, max_relative_position: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = NormalizationLayers.layer_norm(d_model)
        
        # Relative position encoding
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position
        if use_relative_position:
            self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
            self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize attention weights properly."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.w_v.bias)
        nn.init.zeros_(self.w_o.bias)
    
    def _get_relative_positions(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get relative position embeddings."""
        range_vec = torch.arange(seq_len, device=self.w_q.weight.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.T
        
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, 
                                         self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return final_mat, distance_mat
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                relative_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores (prefer SDPA when available)
        if hasattr(F, 'scaled_dot_product_attention'):
            is_causal = mask is not None and mask.dtype == torch.bool and mask.shape[-1] == seq_len
            attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.attention_dropout.p if self.training else 0.0, is_causal=is_causal)
            context = attn
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position information
        if self.use_relative_position and relative_positions is not None:
            relative_positions_k = self.relative_position_k(relative_positions)
            relative_positions_k = relative_positions_k.unsqueeze(0).unsqueeze(0)
            relative_scores_k = torch.matmul(Q, relative_positions_k.transpose(-2, -1))
            scores = scores + relative_scores_k
        
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Attention weights
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
        
        # Add relative position information to values
        if self.use_relative_position and relative_positions is not None:
            relative_positions_v = self.relative_position_v(relative_positions)
            relative_positions_v = relative_positions_v.unsqueeze(0).unsqueeze(0)
            context = context + torch.matmul(attention_weights, relative_positions_v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        output = self.output_dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = NormalizationLayers.layer_norm(d_model)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize feed-forward weights."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU activation
        output = F.gelu(self.w_1(x))
        output = self.dropout(output)
        output = self.w_2(output)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output


class TransformerBlock(nn.Module):
    """Advanced transformer block with pre-norm architecture."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention = MultiHeadAttention(
            config.d_model,
            config.num_heads,
            config.attention_dropout,
            config.use_relative_position,
            config.max_relative_position
        )
        
        # Feed-forward layer
        self.feed_forward = FeedForward(
            config.d_model,
            config.d_ff,
            config.dropout
        )
        
        # Pre-norm layers
        self.pre_norm_1 = NormalizationLayers.layer_norm(config.d_model)
        self.pre_norm_2 = NormalizationLayers.layer_norm(config.d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                relative_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        normed_x = self.pre_norm_1(x)
        attn_output = self.attention(normed_x, normed_x, normed_x, mask, relative_positions)
        x = x + attn_output
        
        # Pre-norm feed-forward
        normed_x = self.pre_norm_2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output
        
        return x


class FacebookPostsTransformer(nn.Module):
    """Advanced transformer model for Facebook Posts processing."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position encoding
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(config.d_model, config.max_seq_length)
            self.position_embedding = None
        else:
            self.position_embedding = PositionalEncoding(config.d_model, config.max_seq_length)
            self.rope = None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Layer normalization for output
        self.final_layer_norm = NormalizationLayers.layer_norm(config.d_model)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize all weights with proper techniques."""
        # Token embedding initialization
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Output projection initialization
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Position encoding
        if self.rope is not None:
            embeddings = self.rope(embeddings, seq_len)
        elif self.position_embedding is not None:
            embeddings = self.position_embedding(embeddings.transpose(0, 1)).transpose(0, 1)
        
        embeddings = self.dropout(embeddings)
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.bool()
        
        # Combine attention mask with causal mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        combined_mask = attention_mask & ~causal_mask
        
        # Get relative positions (compute once per sequence length)
        relative_positions = None
        if self.config.use_relative_position:
            range_vec = torch.arange(seq_len, device=input_ids.device)
            range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
            distance_mat = range_mat - range_mat.T
            max_rel = self.config.max_relative_position if hasattr(self.config, 'max_relative_position') else 64
            distance_mat_clipped = torch.clamp(distance_mat, -max_rel, max_rel)
            relative_positions = distance_mat_clipped + max_rel
        
        # Apply transformer blocks
        hidden_states = embeddings
        attention_weights_list = []
        
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, combined_mask, relative_positions)
            
            if return_attention_weights:
                # Store attention weights (simplified)
                attention_weights_list.append(torch.ones(1))  # Placeholder
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        if return_attention_weights:
            return logits, attention_weights_list
        else:
            return logits


class FacebookPostsLLM(nn.Module):
    """Large Language Model for Facebook Posts generation and analysis."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Use the transformer as the base
        self.transformer = FacebookPostsTransformer(config)
        
        # Additional components for LLM functionality
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.transformer.token_embedding.weight
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize LLM-specific weights."""
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Get transformer outputs
        transformer_outputs = self.transformer(input_ids, attention_mask, return_attention_weights)
        
        if return_attention_weights:
            hidden_states, attention_weights = transformer_outputs
        else:
            hidden_states = transformer_outputs
            attention_weights = None
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift sequences for language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Prepare outputs
        outputs = {
            'logits': lm_logits,
            'hidden_states': hidden_states
        }
        
        if loss is not None:
            outputs['loss'] = loss
        
        if attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs


class AttentionVisualizer:
    """Utility for visualizing attention weights."""
    
    @staticmethod
    def visualize_attention(attention_weights: torch.Tensor, 
                           tokens: List[str],
                           save_path: Optional[str] = None) -> None:
        """Visualize attention weights."""
        
        # Convert attention weights to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues',
                   annot=True,
                   fmt='.2f')
        
        plt.title('Attention Weights Visualization')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ModelCompressor:
    """Utility for model compression and quantization."""
    
    @staticmethod
    def quantize_model(model: nn.Module, quantization_type: str = 'int8') -> nn.Module:
        """Quantize model for efficient inference."""
        if quantization_type == 'int8':
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        elif quantization_type == 'fp16':
            return model.half()
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    @staticmethod
    def prune_model(model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
        """Prune model weights for compression."""
        # Simple magnitude-based pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), pruning_ratio)
                mask = torch.abs(weight) > threshold
                module.weight.data = weight * mask
        
        return model


class FacebookPostsLLMTrainer:
    """Advanced trainer for LLM training and fine-tuning."""
    
    def __init__(self, model: nn.Module, config: TransformerConfig):
        self.model = model.to(DEVICE)
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step) -> Any:
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            else:
                return max(0.0, float(self.config.max_steps - step) / 
                          float(max(1, self.config.max_steps - self.config.warmup_steps)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.item()
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              max_steps: Optional[int] = None) -> Dict[str, List[float]]:
        """Complete training loop."""
        max_steps = max_steps or self.config.max_steps
        step = 0
        
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                self.train_losses.append(loss)
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                
                # Validation
                if val_loader is not None and step % 100 == 0:
                    val_loss = self.validate(val_loader)
                    self.val_losses.append(val_loss)
                    
                    logger.info(f"Step {step}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                
                step += 1
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")


def create_transformer_model(config: TransformerConfig) -> FacebookPostsTransformer:
    """Factory function to create transformer model."""
    return FacebookPostsTransformer(config)


def create_llm_model(config: TransformerConfig) -> FacebookPostsLLM:
    """Factory function to create LLM model."""
    return FacebookPostsLLM(config)


# Example usage and demonstration
if __name__ == "__main__":
    # Create configuration
    config = TransformerConfig(
        vocab_size=50000,
        max_seq_length=512,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        learning_rate=1e-4,
        warmup_steps=10000,
        max_steps=100000,
        batch_size=32,
        use_rope=True,
        use_relative_position=True
    )
    
    # Create model
    model = create_llm_model(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"LLM Model created successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, attention_mask, labels)
    logger.info(f"Model outputs shape: {outputs['logits'].shape}")
    logger.info(f"Loss: {outputs['loss'].item():.4f}") 