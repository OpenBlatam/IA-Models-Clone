"""
Modular Model Architectures
Separated into individual components for better modularity
"""

# Import from attention module (backward compatibility)
try:
    from .attention import (
        MultiHeadAttention,
        ScaledDotProductAttention,
        AttentionLayer
    )
except ImportError:
    # Fallback to submodule
    from .attention.multi_head import MultiHeadAttention
    from .attention.scaled_dot_product import ScaledDotProductAttention
    from .attention import AttentionLayer  # Keep original if exists
# Import from submodules
from .normalization import (
    LayerNorm,
    BatchNorm1d,
    AdaptiveNormalization
)
from .feedforward import (
    FeedForward,
    GatedFeedForward,
    ResidualFeedForward
)
from .positional_encoding import (
    PositionalEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding
)
from .embeddings import (
    FeatureEmbedding,
    AudioFeatureEmbedding,
    MusicFeatureEmbedding
)
from .activations import (
    GELU,
    Swish,
    Mish,
    GLU,
    ActivationFactory,
    create_activation
)
# Import from submodules
from .pooling import (
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    AdaptivePooling,
    PoolingFactory,
    create_pooling
)
from .dropout import (
    StandardDropout,
    SpatialDropout,
    AlphaDropout,
    DropoutFactory,
    create_dropout
)
from .residual import (
    ResidualConnection,
    PreNormResidual,
    PostNormResidual,
    GatedResidual
)

__all__ = [
    # Attention
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "AttentionLayer",
    # Normalization
    "LayerNorm",
    "BatchNorm1d",
    "AdaptiveNormalization",
    # Feedforward
    "FeedForward",
    "GatedFeedForward",
    "ResidualFeedForward",
    # Positional Encoding
    "PositionalEncoding",
    "LearnedPositionalEncoding",
    "SinusoidalPositionalEncoding",
    # Embeddings
    "FeatureEmbedding",
    "AudioFeatureEmbedding",
    "MusicFeatureEmbedding",
    # Activations
    "GELU",
    "Swish",
    "Mish",
    "GLU",
    "ActivationFactory",
    "create_activation",
    # Pooling
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "AdaptivePooling",
    "PoolingFactory",
    "create_pooling",
    # Dropout
    "StandardDropout",
    "SpatialDropout",
    "AlphaDropout",
    "DropoutFactory",
    "create_dropout",
    # Residual
    "ResidualConnection",
    "PreNormResidual",
    "PostNormResidual",
    "GatedResidual",
]
