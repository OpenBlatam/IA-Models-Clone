"""
Enhanced Transformer Models for HeyGen AI.

This module implements state-of-the-art transformer architectures with proper
attention mechanisms, positional encodings, and fine-tuning capabilities.
Follows PyTorch best practices and includes comprehensive error handling.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    PreTrainedTokenizer, PreTrainedModel, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer models with comprehensive settings."""
    
    # Model architecture
    model_name: str = "gpt2"
    model_type: str = "causal_lm"  # causal_lm, seq2seq, encoder
    max_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Optimization settings
    use_fp16: bool = True
    use_mixed_precision: bool = True
    use_distributed: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Device settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        
        if self.use_fp16 and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA not available, falling back to FP32")
            self.use_fp16 = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.
    
    Implements the standard sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return self.dropout(x + self.pe[:x.size(0), :])


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with proper scaling and masking.
    
    Implements scaled dot-product attention with multiple heads as described
    in the original transformer paper.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 use_bias: bool = True, attention_type: str = "standard"):
        """Initialize multi-head attention.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_bias: Whether to use bias in linear layers
            attention_type: Type of attention ("standard", "flash", "xformers")
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_type = attention_type
        
        # Linear transformations for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Layer normalization for attention
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Causal mask for autoregressive models
            attention_mask: Attention mask for padding
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply masks if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Complete transformer block with attention and feed-forward layers."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor
            mask: Causal mask
            attention_mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask, attention_mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


class TransformerModel(nn.Module):
    """Complete transformer model with configurable architecture."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.max_length, config.hidden_size)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.hidden_size, 
            config.max_length, 
            config.dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.dropout
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.max_length)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the transformer model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Target labels for training
            
        Returns:
            Dictionary containing logits and loss (if training)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask, attention_mask=attention_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift sequences for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x
        }


class TransformerManager:
    """Manages transformer models with loading, fine-tuning, and inference capabilities."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Initialize accelerator for distributed training
        if config.use_distributed:
            self.accelerator = Accelerator()
        else:
            self.accelerator = None
    
    def load_pretrained_model(self, model_name: Optional[str] = None) -> None:
        """Load a pre-trained transformer model.
        
        Args:
            model_name: Name of the model to load (uses config if None)
        """
        try:
            model_name = model_name or self.config.model_name
            self.logger.info(f"Loading pre-trained model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.config.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
            elif self.config.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
            
            # Apply LoRA if configured
            if self.config.use_lora:
                self._apply_lora()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable mixed precision
            if self.config.use_fp16:
                self.model = self.model.half()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _apply_lora(self) -> None:
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        try:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.logger.info("LoRA applied successfully")
            
        except Exception as e:
            self.logger.error(f"Error applying LoRA: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.9,
                     do_sample: bool = True) -> str:
        """Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                if self.config.use_fp16:
                    with autocast():
                        outputs = self.model.generate(
                            inputs,
                            max_length=max_length,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise
    
    def save_model(self, output_dir: str) -> None:
        """Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if self.config.use_lora:
                self.model.save_pretrained(output_path)
            else:
                self.model.save_pretrained(output_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
            
            # Save configuration
            self.config.save_pretrained(output_path)
            
            self.logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
