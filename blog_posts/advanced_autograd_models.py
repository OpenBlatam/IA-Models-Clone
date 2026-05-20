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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
                from flash_attn import flash_attn_func
                import xformers.ops as xops
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced PyTorch Autograd Models for Transformers and LLMs
Comprehensive implementation with proper weight initialization, loss functions,
optimization algorithms, attention mechanisms, positional encodings,
efficient fine-tuning techniques (LoRA/P-tuning), and proper tokenization
"""

    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    PreTrainedTokenizer, PreTrainedModel, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, DataCollatorForTokenClassification
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for advanced autograd models"""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_relative_pos: bool = True
    max_relative_position: int = 32
    use_rope: bool = False
    rope_dim: int = 64
    use_flash_attention: bool = False
    use_xformers: bool = False
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    weight_decay: float = 0.01
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    focal_loss_alpha: float = 1.0
    focal_loss_gamma: float = 2.0


class AdvancedPositionalEncoding(nn.Module):
    """Advanced positional encoding with multiple options"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 encoding_type: str = "sinusoidal", use_learnable: bool = True):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.encoding_type = encoding_type
        self.use_learnable = use_learnable
        self.dropout = Dropout(p=dropout)
        
        if encoding_type == "sinusoidal":
            self._create_sinusoidal_encoding()
        elif encoding_type == "learnable":
            self._create_learnable_encoding()
        elif encoding_type == "rope":
            self._create_rope_encoding()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sinusoidal_encoding(self) -> Any:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        if self.use_learnable:
            self.learnable_pe = nn.Parameter(torch.randn(self.max_len, self.d_model) * 0.02)
        else:
            self.learnable_pe = None
    
    def _create_learnable_encoding(self) -> Any:
        """Create learnable positional encoding"""
        self.learnable_pe = nn.Parameter(torch.randn(self.max_len, self.d_model) * 0.02)
        self.pe = None
    
    def _create_rope_encoding(self) -> Any:
        """Create RoPE (Rotary Position Embedding)"""
        self.rope_dim = min(self.d_model // 2, 64)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        if self.use_learnable:
            self.learnable_pe = nn.Parameter(torch.randn(self.max_len, self.d_model) * 0.02)
        else:
            self.learnable_pe = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        
        if self.encoding_type == "rope":
            return self._apply_rope(x, seq_len)
        
        if self.pe is not None:
            pe = self.pe[:seq_len]
            if self.learnable_pe is not None:
                pe = pe + self.learnable_pe[:seq_len]
        else:
            pe = self.learnable_pe[:seq_len]
        
        x = x + pe
        return self.dropout(x)
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE encoding"""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rope_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rope_dim]
        
        # Apply rotation to first rope_dim dimensions
        x_rope = x[..., :self.rope_dim * 2]
        x_rest = x[..., self.rope_dim * 2:]
        
        x_rope_reshaped = x_rope.view(*x_rope.shape[:-1], -1, 2)
        x_rotated = torch.cat([
            x_rope_reshaped[..., 0:1] * cos - x_rope_reshaped[..., 1:2] * sin,
            x_rope_reshaped[..., 0:1] * sin + x_rope_reshaped[..., 1:2] * cos
        ], dim=-1)
        
        x_rotated = x_rotated.view(*x_rope.shape)
        x = torch.cat([x_rotated, x_rest], dim=-1)
        
        if self.learnable_pe is not None:
            x = x + self.learnable_pe[:seq_len]
        
        return self.dropout(x)


class AdvancedMultiHeadAttention(nn.Module):
    """Advanced multi-head attention with multiple attention types"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attention_type: str = "standard", max_relative_position: int = 32,
                 use_flash_attention: bool = False, use_xformers: bool = False):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.attention_type = attention_type
        self.use_flash_attention = use_flash_attention
        self.use_xformers = use_xformers
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
        # Attention type specific components
        if attention_type == "relative":
            self.max_relative_position = max_relative_position
            self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
            self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        elif attention_type == "local":
            self.local_window_size = max_relative_position
        elif attention_type == "sparse":
            self.sparse_attention = self._create_sparse_attention_pattern()
        
        # Flash attention support
        if use_flash_attention:
            try:
                self.flash_attn_func = flash_attn_func
            except ImportError:
                logger.warning("Flash attention not available, falling back to standard attention")
                self.use_flash_attention = False
        
        # xFormers support
        if use_xformers:
            try:
                self.xops = xops
            except ImportError:
                logger.warning("xFormers not available, falling back to standard attention")
                self.use_xformers = False
    
    def _create_sparse_attention_pattern(self) -> torch.Tensor:
        """Create sparse attention pattern"""
        # Create a simple sparse pattern (can be customized)
        pattern = torch.ones(self.n_heads, self.max_relative_position, self.max_relative_position)
        # Make it sparse by zeroing out some connections
        for i in range(self.n_heads):
            if i % 2 == 0:
                pattern[i, :, ::2] = 0  # Every other column
            else:
                pattern[i, ::2, :] = 0  # Every other row
        return pattern
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention based on type
        if self.use_flash_attention and self.attention_type == "standard":
            context = self._flash_attention(q, k, v, mask)
        elif self.use_xformers and self.attention_type == "standard":
            context = self._xformers_attention(q, k, v, mask)
        elif self.attention_type == "relative":
            context = self._relative_attention(q, k, v, mask)
        elif self.attention_type == "local":
            context = self._local_attention(q, k, v, mask)
        elif self.attention_type == "sparse":
            context = self._sparse_attention(q, k, v, mask)
        else:
            context = self._standard_attention(q, k, v, mask)
        
        # Output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return self.layer_norm(output + x)
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flash attention for memory efficiency"""
        q = q.transpose(1, 2)  # [batch_size, seq_len, n_heads, d_k]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
        return self.flash_attn_func(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
    
    def _xformers_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """xFormers attention for efficiency"""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        return self.xops.memory_efficient_attention(q, k, v, attn_bias=mask)
    
    def _relative_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Relative positional attention"""
        batch_size, n_heads, seq_len, d_k = q.size()
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Relative position scores
        relative_positions = self._get_relative_positions(seq_len, q.device)
        relative_position_k = self.relative_position_k(relative_positions)
        relative_scores = torch.matmul(q, relative_position_k.transpose(-2, -1))
        scores = scores + relative_scores
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Standard context
        context = torch.matmul(attention_weights, v)
        
        # Relative position context
        relative_position_v = self.relative_position_v(relative_positions)
        relative_context = torch.matmul(attention_weights, relative_position_v)
        
        return context + relative_context
    
    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Local attention within a window"""
        batch_size, n_heads, seq_len, d_k = q.size()
        
        # Create local attention mask
        local_mask = torch.ones(seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            local_mask[i, :start] = 0
            local_mask[i, end:] = 0
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse attention with predefined patterns"""
        batch_size, n_heads, seq_len, d_k = q.size()
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse pattern
        sparse_pattern = self.sparse_attention.to(q.device)
        if sparse_pattern.size(0) < n_heads:
            sparse_pattern = sparse_pattern.repeat(n_heads // sparse_pattern.size(0), 1, 1)
        sparse_pattern = sparse_pattern[:n_heads, :seq_len, :seq_len]
        
        scores = scores.masked_fill(sparse_pattern.unsqueeze(0) == 0, -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get relative positions for relative attention"""
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features,
                             rank, alpha, dropout)
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class P_TuningLayer(nn.Module):
    """P-tuning layer for prompt tuning"""
    
    def __init__(self, d_model: int, prompt_length: int = 10, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, d_model) * 0.02)
        self.dropout = Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompts, x], dim=1)


class AdvancedLossFunctions:
    """Advanced loss functions for different tasks"""
    
    @staticmethod
    def focal_loss(predictions: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    @staticmethod
    def label_smoothing_loss(predictions: torch.Tensor, targets: torch.Tensor,
                           smoothing: float = 0.1, reduction: str = 'mean') -> torch.Tensor:
        """Label smoothing loss for better generalization"""
        num_classes = predictions.size(-1)
        smoothed_targets = torch.zeros_like(predictions)
        smoothed_targets.fill_(smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        
        return F.kl_div(F.log_softmax(predictions, dim=-1), smoothed_targets, reduction=reduction)
    
    @staticmethod
    def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                        temperature: float = 0.07, margin: float = 1.0) -> torch.Tensor:
        """Contrastive loss for learning representations"""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Compute positive and negative similarities
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # Compute loss
        positive_loss = -torch.log(torch.exp(positive_similarities) + 1e-8).sum(dim=1)
        negative_loss = torch.log(1 + torch.exp(negative_similarities - margin)).sum(dim=1)
        
        return (positive_loss + negative_loss).mean()
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                    margin: float = 1.0) -> torch.Tensor:
        """Triplet loss for metric learning"""
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()


class AdvancedOptimizers:
    """Advanced optimizers with learning rate scheduling"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: ModelConfig) -> optim.Optimizer:
        """Create optimizer with proper parameter grouping"""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        return optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config: ModelConfig,
                        num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1
        )


class AdvancedTokenizer:
    """Advanced tokenizer with proper sequence handling"""
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512,
                 padding: str = "max_length", truncation: bool = True):
        
    """__init__ function."""
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_text(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize text with proper handling"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return tokenized
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids"""
        return (input_ids != self.tokenizer.pad_token_id).long()
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token ids back to text"""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)


class AdvancedTransformerModel(nn.Module):
    """Advanced transformer model with all optimizations"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = AdvancedPositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout,
            encoding_type="sinusoidal" if not config.use_rope else "rope",
            use_learnable=True
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            self._create_transformer_layer() for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _create_transformer_layer(self) -> nn.Module:
        """Create a transformer layer with advanced attention"""
        return nn.ModuleDict({
            'attention': AdvancedMultiHeadAttention(
                self.config.d_model,
                self.config.n_heads,
                self.config.dropout,
                attention_type="standard",
                max_relative_position=self.config.max_relative_position,
                use_flash_attention=self.config.use_flash_attention,
                use_xformers=self.config.use_xformers
            ),
            'feed_forward': nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_ff),
                self._get_activation(self.config.activation),
                Dropout(self.config.dropout),
                nn.Linear(self.config.d_ff, self.config.d_model)
            ),
            'layer_norm1': LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps),
            'layer_norm2': LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps),
            'dropout': Dropout(self.config.dropout)
        })
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def _init_weights(self) -> Any:
        """Initialize model weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with proper autograd handling"""
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.config.dropout(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, seq_len, device=input_ids.device)
        
        # Transformer layers
        for layer in self.transformer_layers:
            # Self-attention with residual connection
            attn_output = layer['attention'](x, attention_mask)
            x = layer['layer_norm1'](x + layer['dropout'](attn_output))
            
            # Feed-forward with residual connection
            ff_output = layer['feed_forward'](x)
            x = layer['layer_norm2'](x + layer['dropout'](ff_output))
        
        # Output projection
        logits = self.output_layer(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift sequences for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss with label smoothing
            loss = AdvancedLossFunctions.label_smoothing_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                smoothing=self.config.label_smoothing
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x
        }


class AdvancedTrainingPipeline:
    """Advanced training pipeline with all optimizations"""
    
    def __init__(self, model: nn.Module, config: ModelConfig, tokenizer: AdvancedTokenizer):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = AdvancedOptimizers.create_optimizer(model, config)
        self.scheduler = AdvancedOptimizers.create_scheduler(
            self.optimizer, config, num_training_steps=10000
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss functions
        self.loss_functions = AdvancedLossFunctions()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with proper autograd"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch.get('labels', input_ids.clone()).to(self.device)
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.config.mixed_precision and self.scaler is not None:
            with autocast():
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', input_ids.clone()).to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Compute perplexity
            perplexity = torch.exp(loss)
            
            return {
                'loss': loss.item(),
                'perplexity': perplexity.item()
            }
    
    def generate_text(self, prompt: str, max_length: int = 100,
                     temperature: float = 1.0, top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """Generate text using the trained model"""
        self.model.eval()
        
        # Tokenize prompt
        tokenized = self.tokenizer.tokenize_text(prompt)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
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
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=self.device)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode_tokens(input_ids[0])[0]
        return generated_text


# Example usage and testing
def main():
    """Example usage of advanced autograd models"""
    
    # Configuration
    config = ModelConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=6,
        n_heads=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1,
        learning_rate=1e-4,
        warmup_steps=1000
    )
    
    # Create model
    model = AdvancedTransformerModel(config)
    
    # Create tokenizer
    tokenizer = AdvancedTokenizer("gpt2", max_length=512)
    
    # Create training pipeline
    pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
    
    # Example training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Tokenize data
    tokenized_data = tokenizer.tokenize_text(texts)
    
    # Create dataset
    dataset = TensorDataset(
        tokenized_data['input_ids'],
        tokenized_data['attention_mask'],
        tokenized_data['input_ids'].clone()  # Labels for language modeling
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training loop
    for epoch in range(3):
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            # Training step
            train_metrics = pipeline.train_step(batch)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"Loss: {train_metrics['loss']:.4f}")
                print(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
    
    # Generate text
    generated_text = pipeline.generate_text(
        "The future of artificial intelligence",
        max_length=50,
        temperature=0.8
    )
    print(f"Generated text: {generated_text}")


match __name__:
    case "__main__":
    main() 