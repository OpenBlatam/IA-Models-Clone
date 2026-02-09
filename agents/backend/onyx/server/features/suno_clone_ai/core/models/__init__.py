"""
Enhanced Model Architectures

Provides:
- Enhanced music generation models with proper nn.Module implementation
- LoRA adapters for efficient fine-tuning
- Enhanced diffusion models with proper pipelines
- Modular attention mechanisms
- Modular layer components
- Weight initialization strategies
"""

from .enhanced_music_model import (
    EnhancedMusicModel,
    create_enhanced_music_model
)

from .lora_adapter import (
    LoRALayer,
    LoRALinear,
    LoRAAdapter,
    add_lora_to_model
)

from .enhanced_diffusion import (
    EnhancedDiffusionGenerator,
    AudioDiffusionPipeline
)

# Attention mechanisms
from .attention import (
    MultiHeadAttention,
    ScaledDotProductAttention
)

# Layers
from .layers import (
    TransformerBlock,
    FeedForward,
    PositionalEncoding
)

# Initialization
from .initialization import (
    initialize_weights,
    initialize_linear,
    initialize_conv,
    initialize_embedding,
    initialize_layer_norm
)

__all__ = [
    # Enhanced Music Model
    "EnhancedMusicModel",
    "create_enhanced_music_model",
    # LoRA
    "LoRALayer",
    "LoRALinear",
    "LoRAAdapter",
    "add_lora_to_model",
    # Diffusion
    "EnhancedDiffusionGenerator",
    "AudioDiffusionPipeline",
    # Attention
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    # Layers
    "TransformerBlock",
    "FeedForward",
    "PositionalEncoding",
    # Initialization
    "initialize_weights",
    "initialize_linear",
    "initialize_conv",
    "initialize_embedding",
    "initialize_layer_norm"
]

