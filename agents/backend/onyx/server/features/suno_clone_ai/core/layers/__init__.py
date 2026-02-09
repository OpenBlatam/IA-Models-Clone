"""
Neural Network Layers Module

Provides:
- Attention mechanisms
- Normalization layers
- Activation functions
- Embedding layers
- Pooling layers
- Convolutional layers
- Regularization layers
"""

from .attention import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    CrossAttention,
    SelfAttention
)

from .normalization import (
    LayerNorm,
    RMSNorm,
    GroupNorm,
    InstanceNorm
)

from .activation import (
    GELU,
    Swish,
    GLU,
    create_activation
)

from .embedding import (
    PositionalEncoding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    TokenEmbedding
)

from .pooling import (
    AdaptivePooling,
    GlobalPooling,
    create_pooling
)

from .convolution import (
    Conv1dBlock,
    Conv2dBlock,
    DepthwiseConv1d,
    SeparableConv1d
)

from .regularization import (
    Dropout,
    DropPath,
    StochasticDepth
)

__all__ = [
    # Attention
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "CrossAttention",
    "SelfAttention",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    # Activation
    "GELU",
    "Swish",
    "GLU",
    "create_activation",
    # Embedding
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "TokenEmbedding",
    # Pooling
    "AdaptivePooling",
    "GlobalPooling",
    "create_pooling",
    # Convolution
    "Conv1dBlock",
    "Conv2dBlock",
    "DepthwiseConv1d",
    "SeparableConv1d",
    # Regularization
    "Dropout",
    "DropPath",
    "StochasticDepth"
]



