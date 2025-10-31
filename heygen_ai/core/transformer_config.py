"""
Enhanced Transformer Configuration Module

This module contains the core configuration classes and dataclasses
for the enhanced transformer models.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch


@dataclass
class TransformerConfig:
    """Configuration class for enhanced transformer models."""
    
    # Core parameters
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    
    # LoRA parameters
    enable_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Performance parameters
    enable_ultra_performance: bool = False
    performance_mode: str = "balanced"  # "balanced", "speed", "memory", "maximum"
    enable_torch_compile: bool = False
    enable_flash_attention: bool = False
    enable_memory_optimization: bool = False
    mixed_precision: bool = False
    
    # Advanced features
    enable_sparse_attention: bool = False
    enable_linear_attention: bool = False
    enable_memory_efficient_attention: bool = False
    enable_adaptive_attention: bool = False
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Model architecture
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        if self.performance_mode not in ["balanced", "speed", "memory", "maximum"]:
            raise ValueError("performance_mode must be one of: balanced, speed, memory, maximum")
        
        if self.activation_function not in ["gelu", "relu", "swish", "gelu_new"]:
            raise ValueError("activation_function must be one of: gelu, relu, swish, gelu_new")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'dropout': self.dropout,
            'enable_lora': self.enable_lora,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'enable_ultra_performance': self.enable_ultra_performance,
            'performance_mode': self.performance_mode,
            'enable_torch_compile': self.enable_torch_compile,
            'enable_flash_attention': self.enable_flash_attention,
            'enable_memory_optimization': self.enable_memory_optimization,
            'mixed_precision': self.mixed_precision,
            'enable_sparse_attention': self.enable_sparse_attention,
            'enable_linear_attention': self.enable_linear_attention,
            'enable_memory_efficient_attention': self.enable_memory_efficient_attention,
            'enable_adaptive_attention': self.enable_adaptive_attention,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'activation_function': self.activation_function,
            'layer_norm_eps': self.layer_norm_eps,
            'initializer_range': self.initializer_range
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def get_model_size(self) -> int:
        """Calculate approximate model size in parameters."""
        # Embedding parameters
        embedding_params = self.vocab_size * self.hidden_size + self.max_position_embeddings * self.hidden_size
        
        # Transformer layer parameters
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        ffn_params = 2 * self.hidden_size * self.intermediate_size  # FFN up and down projections
        layer_norm_params = 4 * self.hidden_size  # 2 layer norms per layer
        
        layer_params = attention_params + ffn_params + layer_norm_params
        total_layer_params = layer_params * self.num_layers
        
        # Final layer norm
        final_layer_norm_params = self.hidden_size
        
        total_params = embedding_params + total_layer_params + final_layer_norm_params
        
        return total_params
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        model_size = self.get_model_size()
        
        # 4 bytes per parameter (float32)
        model_memory = model_size * 4 / (1024 * 1024)
        
        # Additional memory for activations, gradients, optimizer states
        activation_memory = model_memory * 0.5  # Rough estimate
        gradient_memory = model_memory  # Same as model
        optimizer_memory = model_memory * 2  # Adam optimizer stores momentum and variance
        
        total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
        
        return {
            'model_memory_mb': model_memory,
            'activation_memory_mb': activation_memory,
            'gradient_memory_mb': gradient_memory,
            'optimizer_memory_mb': optimizer_memory,
            'total_memory_mb': total_memory
        }