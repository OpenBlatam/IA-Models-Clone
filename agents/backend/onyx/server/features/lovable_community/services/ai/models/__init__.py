"""
Custom Model Architectures

Provides custom nn.Module classes for:
- Transformer-based models
- Classification heads
- Regression heads
- Custom architectures
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

"""
Custom Model Architectures Module

Organized into sub-modules:
- heads: Classification and regression heads
- architectures: Complete model architectures
- utils: Weight initialization and visualization
"""

from .heads import (
    ClassificationHead,
    RegressionHead
)

from .architectures import (
    TransformerClassifier,
    MultiTaskModel
)

from .utils import (
    WeightInitializer,
    AttentionVisualizer
)

try:
    from .advanced_architectures import (
        LayerNorm,
        MultiHeadAttention,
        FeedForward,
        TransformerBlock,
        PositionalEncoding,
        AdvancedTransformer,
        WeightInitializer as AdvancedWeightInitializer
    )
except ImportError:
    LayerNorm = None
    MultiHeadAttention = None
    FeedForward = None
    TransformerBlock = None
    PositionalEncoding = None
    AdvancedTransformer = None
    AdvancedWeightInitializer = None

__all__ = [
    "ClassificationHead",
    "RegressionHead",
    "TransformerClassifier",
    "MultiTaskModel",
    "WeightInitializer",
    "AttentionVisualizer",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "PositionalEncoding",
    "AdvancedTransformer",
    "AdvancedWeightInitializer",
]

