from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
from dataclasses import dataclass
import json
import os
from pathlib import Path
    from transformers import (
    from peft import (
    from peft.tuners.lora import LoraLayer
        import matplotlib.pyplot as plt
        import seaborn as sns
from typing import Any, List, Dict, Optional
import asyncio
"""
🧠 Advanced Attention Mechanisms & Efficient Fine-tuning
======================================================

This module implements advanced attention mechanisms, positional encodings,
and efficient fine-tuning techniques like LoRA and P-tuning for Facebook Posts processing.

Key Features:
- Advanced attention mechanisms (Flash Attention, Sparse Attention)
- Multiple positional encoding techniques
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- P-tuning for prompt-based fine-tuning
- AdaLoRA for adaptive rank allocation
- QLoRA for quantized fine-tuning
- Attention visualization and analysis
"""


# Import transformers for LoRA and other techniques
try:
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        PreTrainedTokenizer, PreTrainedModel
    )
        LoraConfig, get_peft_model, TaskType, 
        PeftModel, PeftConfig, prepare_model_for_kbit_training
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/PEFT not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


@dataclass
class AttentionConfig:
    """Configuration for advanced attention mechanisms."""
    # Attention parameters
    d_model: int = 768
    num_heads: int = 12
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1
    
    # Attention type
    attention_type: str = "standard"  # standard, flash, sparse, local
    use_relative_position: bool = True
    max_relative_position: int = 64
    
    # Sparse attention parameters
    sparse_attention_window: int = 128
    sparse_attention_stride: int = 64
    
    # Local attention parameters
    local_attention_window: int = 256
    
    # Flash attention parameters
    use_flash_attention: bool = False
    flash_attention_dropout: float = 0.1


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding techniques."""
    # Basic parameters
    d_model: int = 768
    max_seq_length: int = 2048
    
    # Encoding type
    encoding_type: str = "sinusoidal"  # sinusoidal, learned, rope, alibi, t5
    
    # RoPE parameters
    rope_dim: int = 64
    rope_base: float = 10000.0
    
    # ALiBi parameters
    alibi_heads: int = 12
    
    # T5 relative position parameters
    t5_num_buckets: int = 32
    t5_max_distance: int = 128


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Task type
    task_type: str = "SEQUENCE_CLASSIFICATION"
    
    # Inference parameters
    inference_mode: bool = False


@dataclass
class PTuningConfig:
    """Configuration for P-tuning."""
    # P-tuning parameters
    num_virtual_tokens: int = 20
    encoder_hidden_size: int = 128
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Task type
    task_type: str = "SEQUENCE_CLASSIFICATION"


class AdvancedPositionalEncoding(nn.Module):
    """Advanced positional encoding with multiple techniques."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.encoding_type = config.encoding_type
        
        if config.encoding_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEncoding(config)
        elif config.encoding_type == "learned":
            self.encoding = LearnedPositionalEncoding(config)
        elif config.encoding_type == "rope":
            self.encoding = RotaryPositionalEncoding(config)
        elif config.encoding_type == "alibi":
            self.encoding = ALiBiPositionalEncoding(config)
        elif config.encoding_type == "t5":
            self.encoding = T5RelativePositionalEncoding(config)
        else:
            raise ValueError(f"Unsupported encoding type: {config.encoding_type}")
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        return self.encoding(x, seq_len)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                           (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)
        return x + position_embeddings


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) - Su et al., 2021."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.d_model = config.d_model
        self.rope_dim = config.rope_dim
        self.rope_base = config.rope_base
        
        # Generate rotation matrices
        inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, config.rope_dim, 2).float() / config.rope_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = torch.cos(emb)[None, :, None, :]
        sin = torch.sin(emb)[None, :, None, :]
        
        x_rot = torch.cat([-x[..., self.rope_dim//2:], x[..., :self.rope_dim//2]], dim=-1)
        return x * cos + x_rot * sin


class ALiBiPositionalEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi) - Press et al., 2021."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.alibi_heads = config.alibi_heads
        self.max_seq_length = config.max_seq_length
        
        # Generate ALiBi slopes
        slopes = torch.Tensor(self._get_slopes(config.alibi_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(config.max_seq_length).unsqueeze(0).unsqueeze(0)
        self.register_buffer('alibi', alibi)
    
    def _get_slopes(self, n_heads: int) -> List[float]:
        """Get ALiBi slopes for n heads."""
        def get_slopes_power_of_2(n) -> Optional[Dict[str, Any]]:
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2]
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        return x + self.alibi[:self.alibi_heads, :seq_len, :seq_len]


class T5RelativePositionalEncoding(nn.Module):
    """T5 relative positional encoding."""
    
    def __init__(self, config: PositionalEncodingConfig):
        
    """__init__ function."""
super().__init__()
        self.num_buckets = config.t5_num_buckets
        self.max_distance = config.t5_max_distance
        self.relative_attention_bias = nn.Embedding(config.t5_num_buckets, config.alibi_heads)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative position to bucket index."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        relative_buckets = 0
        if not self.directional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        max_exact = num_buckets // 4
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        
        context_position = torch.arange(seq_len, dtype=torch.long, device=x.device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=x.device)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return x + values


class AdvancedMultiHeadAttention(nn.Module):
    """Advanced multi-head attention with multiple attention types."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.dropout = config.dropout
        
        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Attention type
        self.attention_type = config.attention_type
        
        # Relative position encoding
        if config.use_relative_position:
            self.relative_position_k = nn.Embedding(2 * config.max_relative_position + 1, self.d_k)
            self.relative_position_v = nn.Embedding(2 * config.max_relative_position + 1, self.d_k)
        
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
        
        distance_mat_clipped = torch.clamp(distance_mat, -self.config.max_relative_position, 
                                         self.config.max_relative_position)
        final_mat = distance_mat_clipped + self.config.max_relative_position
        
        return final_mat, distance_mat
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard multi-head attention."""
        batch_size, num_heads, seq_len, d_k = Q.size()
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context
    
    def _sparse_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse attention with sliding window."""
        batch_size, num_heads, seq_len, d_k = Q.size()
        window_size = self.config.sparse_attention_window
        
        # Create sparse attention pattern
        context = torch.zeros_like(Q)
        
        for i in range(0, seq_len, self.config.sparse_attention_stride):
            end_idx = min(i + window_size, seq_len)
            
            # Extract window
            Q_window = Q[:, :, i:end_idx, :]
            K_window = K[:, :, i:end_idx, :]
            V_window = V[:, :, i:end_idx, :]
            
            # Compute attention for window
            scores = torch.matmul(Q_window, K_window.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                mask_window = mask[:, :, i:end_idx, i:end_idx]
                scores = scores.masked_fill(mask_window == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            window_context = torch.matmul(attention_weights, V_window)
            context[:, :, i:end_idx, :] = window_context
        
        return context
    
    def _local_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Local attention with fixed window size."""
        batch_size, num_heads, seq_len, d_k = Q.size()
        window_size = self.config.local_attention_window
        
        # Pad sequences for local attention
        pad_size = window_size // 2
        Q_padded = F.pad(Q, (0, 0, pad_size, pad_size))
        K_padded = F.pad(K, (0, 0, pad_size, pad_size))
        V_padded = F.pad(V, (0, 0, pad_size, pad_size))
        
        context = torch.zeros_like(Q)
        
        for i in range(seq_len):
            start_idx = i
            end_idx = i + window_size
            
            # Extract local window
            Q_local = Q_padded[:, :, start_idx:end_idx, :]
            K_local = K_padded[:, :, start_idx:end_idx, :]
            V_local = V_padded[:, :, start_idx:end_idx, :]
            
            # Compute local attention
            scores = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                mask_local = mask[:, :, i:i+1, start_idx:end_idx]
                scores = scores.masked_fill(mask_local == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            local_context = torch.matmul(attention_weights, V_local)
            context[:, :, i:i+1, :] = local_context
        
        return context
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                relative_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Apply attention based on type
        if self.attention_type == "standard":
            context = self._standard_attention(Q, K, V, mask)
        elif self.attention_type == "sparse":
            context = self._sparse_attention(Q, K, V, mask)
        elif self.attention_type == "local":
            context = self._local_attention(Q, K, V, mask)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        
        # Add relative position information
        if self.config.use_relative_position and relative_positions is not None:
            relative_positions_k = self.relative_position_k(relative_positions)
            relative_positions_k = relative_positions_k.unsqueeze(0).unsqueeze(0)
            relative_scores_k = torch.matmul(Q, relative_positions_k.transpose(-2, -1))
            context = context + relative_scores_k
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        output = self.output_dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output


class LoRAFineTuner:
    """LoRA (Low-Rank Adaptation) fine-tuning implementation."""
    
    def __init__(self, config: LoRAConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.tokenizer = None
    
    def setup_lora(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Setup LoRA for a pre-trained model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers and PEFT are required for LoRA fine-tuning")
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA configuration
        if self.config.target_modules is None:
            self.config.target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=TaskType.SEQUENCE_CLASSIFICATION,
            inference_mode=self.config.inference_mode
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup completed successfully!")
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Train the model with LoRA."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_lora() first.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./lora_facebook_posts",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            fp16=True,
            dataloader_pin_memory=True,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            logging_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        logger.info("LoRA training completed successfully!")
    
    def save_lora(self, path: str):
        """Save LoRA adapter."""
        if self.model is None:
            raise ValueError("Model not set up.")
        
        self.model.save_pretrained(path)
        logger.info(f"LoRA adapter saved to {path}")
    
    def load_lora(self, path: str):
        """Load LoRA adapter."""
        if self.model is None:
            raise ValueError("Model not set up.")
        
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA adapter loaded from {path}")


class PTuningFineTuner:
    """P-tuning fine-tuning implementation."""
    
    def __init__(self, config: PTuningConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.tokenizer = None
    
    def setup_ptuning(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Setup P-tuning for a pre-trained model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers and PEFT are required for P-tuning")
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Create P-tuning configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQUENCE_CLASSIFICATION,
            num_virtual_tokens=self.config.num_virtual_tokens,
            encoder_hidden_size=self.config.encoder_hidden_size,
            encoder_num_layers=self.config.encoder_num_layers,
            encoder_dropout=self.config.encoder_dropout,
        )
        
        # Apply P-tuning
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("P-tuning setup completed successfully!")
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Train the model with P-tuning."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_ptuning() first.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./ptuning_facebook_posts",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=self.config.weight_decay,
            fp16=True,
            dataloader_pin_memory=True,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            logging_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        logger.info("P-tuning training completed successfully!")


class AttentionVisualizer:
    """Advanced attention visualization and analysis."""
    
    @staticmethod
    def visualize_attention_weights(attention_weights: torch.Tensor, 
                                   tokens: List[str],
                                   save_path: Optional[str] = None,
                                   title: str = "Attention Weights") -> None:
        """Visualize attention weights with advanced features."""
        
        # Convert attention weights to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(title)
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def analyze_attention_patterns(attention_weights: torch.Tensor,
                                 tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns and statistics."""
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        analysis = {
            'mean_attention': np.mean(attention_weights),
            'std_attention': np.std(attention_weights),
            'max_attention': np.max(attention_weights),
            'min_attention': np.min(attention_weights),
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-8)),
            'most_attended_tokens': [],
            'least_attended_tokens': []
        }
        
        # Find most and least attended tokens
        token_attention_sums = np.sum(attention_weights, axis=0)
        most_attended_indices = np.argsort(token_attention_sums)[-5:]
        least_attended_indices = np.argsort(token_attention_sums)[:5]
        
        analysis['most_attended_tokens'] = [(tokens[i], token_attention_sums[i]) 
                                          for i in most_attended_indices]
        analysis['least_attended_tokens'] = [(tokens[i], token_attention_sums[i]) 
                                           for i in least_attended_indices]
        
        return analysis


# Example usage and demonstration
if __name__ == "__main__":
    # Test advanced attention mechanisms
    attention_config = AttentionConfig(
        d_model=256,
        num_heads=8,
        attention_type="standard",
        use_relative_position=True
    )
    
    attention = AdvancedMultiHeadAttention(attention_config)
    
    # Test positional encoding
    pos_config = PositionalEncodingConfig(
        d_model=256,
        encoding_type="rope"
    )
    
    pos_encoding = AdvancedPositionalEncoding(pos_config)
    
    # Test with sample data
    batch_size, seq_len, d_model = 2, 64, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Apply positional encoding
    x_with_pos = pos_encoding(x, seq_len)
    
    # Apply attention
    output = attention(x_with_pos, x_with_pos, x_with_pos)
    
    logger.info("Advanced Attention & Fine-tuning Test Results:")
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Positional encoding type: {pos_config.encoding_type}")
    logger.info(f"Attention type: {attention_config.attention_type}")
    
    # Test LoRA if available
    if TRANSFORMERS_AVAILABLE:
        logger.info("Transformers/PEFT available - LoRA and P-tuning ready!")
    else:
        logger.info("Transformers/PEFT not available - LoRA and P-tuning disabled.") 