from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
from .attention_mechanisms import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Transformer Architectures
Production-ready transformer models with PyTorch, proper GPU utilization, and mixed precision training.
"""


    AttentionWithPositionalEncoding,
    MultiHeadAttention,
    SinusoidalPositionEmbedding,
    LearnedPositionEmbedding,
    RotaryPositionEmbedding,
    ALiBiPositionEmbedding,
    AttentionConfig
)

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    vocab_size: int = 50257
    max_seq_length: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Positional encoding options
    position_encoding_type: str = "sinusoidal"  # sinusoidal, learned, rope, alibi
    use_relative_position: bool = True
    use_rope: bool = False
    use_alibi: bool = False
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False
    use_8bit: bool = False
    use_4bit: bool = False
    gradient_checkpointing: bool = True
    
    # Output configuration
    output_dir: str = "./transformer_outputs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class FeedForward(nn.Module):
    """Feed-forward network with proper weight initialization."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.intermediate_layer = nn.Linear(hidden_size, intermediate_size)
        self.output_layer = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        nn.init.xavier_uniform_(self.intermediate_layer.weight)
        nn.init.constant_(self.intermediate_layer.bias, 0)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        # Intermediate layer with GELU activation
        x = self.intermediate_layer(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.dropout(x)
        
        # Residual connection and layer normalization
        x = self.layer_norm(residual + x)
        
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer with advanced attention and feed-forward components."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        
        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            max_position_embeddings=config.max_seq_length,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            use_relative_position=config.use_relative_position,
            use_rope=config.use_rope,
            use_alibi=config.use_alibi
        )
        
        # Self-attention with positional encoding
        self.attention = AttentionWithPositionalEncoding(attention_config)
        
        # Feed-forward
        self.feed_forward = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with positional encoding
        attention_output = self.attention(
            hidden_states, attention_mask, position_ids
        )
        
        # Feed-forward
        output = self.feed_forward(attention_output)
        
        return output


class TransformerModel(nn.Module):
    """Complete transformer model with advanced attention mechanisms."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional encoding based on configuration
        if config.position_encoding_type == "learned":
            self.position_embeddings = LearnedPositionEmbedding(
                config.hidden_size, config.max_seq_length, config.dropout
            )
        elif config.position_encoding_type == "rope":
            self.position_embeddings = RotaryPositionEmbedding(
                config.hidden_size // config.num_heads
            )
        elif config.position_encoding_type == "alibi":
            self.position_embeddings = ALiBiPositionEmbedding(
                config.num_heads, config.max_seq_length
            )
        else:  # sinusoidal (default)
            self.position_embeddings = SinusoidalPositionEmbedding(
                config.hidden_size, config.max_seq_length, config.dropout
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights for the entire model."""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=self.config.initializer_range)
        
        # Output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=self.config.initializer_range)
        
        # Tie weights between input embeddings and output projection
        self.output_projection.weight = self.token_embeddings.weight
    
    def get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate attention mask for padding tokens."""
        attention_mask = (input_ids != self.config.pad_token_id).float()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer model."""
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Add positional encoding
        if isinstance(self.position_embeddings, SinusoidalPositionEmbedding):
            embeddings = embeddings.transpose(0, 1)  # (seq_length, batch_size, hidden_size)
            embeddings = self.position_embeddings(embeddings)
            embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_length, hidden_size)
        elif isinstance(self.position_embeddings, LearnedPositionEmbedding):
            embeddings = self.position_embeddings(embeddings, position_ids)
        elif isinstance(self.position_embeddings, RotaryPositionEmbedding):
            embeddings = self.position_embeddings(embeddings, seq_length)
        # ALiBi is applied in the attention mechanism
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits


class TransformerDataset(Dataset):
    """Custom dataset for transformer training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        
    """__init__ function."""
self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = self.texts[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Set labels to input_ids for language modeling
        encoding['labels'] = encoding['input_ids'].clone()
        
        return encoding


class AdvancedTransformerSystem:
    """Advanced transformer system with training and inference capabilities."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = TransformerModel(config)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=100,  # Will be updated during training
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Transformer system initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Positional encoding: {config.position_encoding_type}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with proper loss computation."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            logits = self.model(input_ids)
            
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def generate_text(self, prompt: str, tokenizer, max_length: int = 100,
                     temperature: float = 0.7, top_p: float = 0.9,
                     do_sample: bool = True) -> str:
        """Generate text using the trained transformer model."""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-p sampling
                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.config.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config.__dict__
        }, os.path.join(path, 'transformer_model.pth'))
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(os.path.join(path, 'transformer_model.pth'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Model loaded from: {path}")


def create_transformer_system(vocab_size: int = 50257, hidden_size: int = 768,
                            num_layers: int = 12, use_fp16: bool = True,
                            position_encoding: str = "sinusoidal") -> AdvancedTransformerSystem:
    """Create a transformer system with default configuration."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        fp16=use_fp16,
        position_encoding_type=position_encoding,
        batch_size=4 if use_fp16 else 8,
        num_epochs=5
    )
    return AdvancedTransformerSystem(config)


# Example usage
if __name__ == "__main__":
    # Create transformer system with different positional encodings
    transformer_system = create_transformer_system(
        position_encoding="rope"  # Try: sinusoidal, learned, rope, alibi
    )
    
    # Sample training data (placeholder)
    sample_input_ids = torch.randint(0, 50257, (4, 128))
    sample_labels = sample_input_ids.clone()
    batch = {'input_ids': sample_input_ids, 'labels': sample_labels}
    
    # Training step
    loss_info = transformer_system.train_step(batch)
    print(f"Training loss: {loss_info['loss']:.4f}")
    
    # Save model
    transformer_system.save_model("./transformer_checkpoint") 