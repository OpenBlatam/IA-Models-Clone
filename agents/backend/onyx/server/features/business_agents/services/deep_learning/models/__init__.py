"""Model architectures for deep learning service."""

from .base_model import BaseModel
from .cnn import SimpleCNN
from .lstm import LSTMTextClassifier
from .transformer import TransformerEncoder, PositionalEncoding

# Optional optimized transformer
try:
    from .optimized_transformer import OptimizedTransformerEncoder
    OPTIMIZED_TRANSFORMER_AVAILABLE = True
except ImportError:
    OPTIMIZED_TRANSFORMER_AVAILABLE = False
    OptimizedTransformerEncoder = None

# Optional imports
try:
    from .transformers_models import HuggingFaceModel, CLIPTextEncoder
    TRANSFORMERS_MODELS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_MODELS_AVAILABLE = False
    HuggingFaceModel = None
    CLIPTextEncoder = None

try:
    from .diffusion_models import DiffusionModel, DDPMTrainer
    DIFFUSION_MODELS_AVAILABLE = True
except ImportError:
    DIFFUSION_MODELS_AVAILABLE = False
    DiffusionModel = None
    DDPMTrainer = None

__all__ = [
    "BaseModel",
    "SimpleCNN",
    "LSTMTextClassifier",
    "TransformerEncoder",
    "PositionalEncoding",
]

if OPTIMIZED_TRANSFORMER_AVAILABLE:
    __all__.append("OptimizedTransformerEncoder")

if TRANSFORMERS_MODELS_AVAILABLE:
    __all__.extend(["HuggingFaceModel", "CLIPTextEncoder"])

if DIFFUSION_MODELS_AVAILABLE:
    __all__.extend(["DiffusionModel", "DDPMTrainer"])
