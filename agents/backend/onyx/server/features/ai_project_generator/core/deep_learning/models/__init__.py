"""
Models Module - Custom PyTorch Model Architectures
==================================================

This module contains custom nn.Module classes for various model architectures:
- Transformers
- CNNs
- RNNs/LSTMs/GRUs
- Diffusion models
- Vision Transformers
- Custom architectures

All models follow PyTorch best practices with proper initialization,
normalization, and weight initialization.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .base_model import BaseModel
from .transformer_model import TransformerModel
from .factory import create_model, initialize_weights

# Try to import additional models
try:
    from .cnn_model import CNNModel
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    CNNModel = None

try:
    from .rnn_model import RNNModel
    RNN_AVAILABLE = True
except ImportError:
    RNN_AVAILABLE = False
    RNNModel = None

try:
    from .transformers_integration import (
        TransformersModelWrapper,
        create_transformers_model
    )
    TRANSFORMERS_INTEGRATION_AVAILABLE = True
except ImportError:
    TRANSFORMERS_INTEGRATION_AVAILABLE = False
    TransformersModelWrapper = None
    create_transformers_model = None

try:
    from .diffusion_model import (
        DiffusionModelWrapper,
        create_diffusion_model
    )
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    DiffusionModelWrapper = None
    create_diffusion_model = None

__all__ = [
    "BaseModel",
    "TransformerModel",
    "create_model",
    "initialize_weights",
]

if CNN_AVAILABLE:
    __all__.append("CNNModel")

if RNN_AVAILABLE:
    __all__.append("RNNModel")

if TRANSFORMERS_INTEGRATION_AVAILABLE:
    __all__.extend(["TransformersModelWrapper", "create_transformers_model"])

if DIFFUSION_AVAILABLE:
    __all__.extend(["DiffusionModelWrapper", "create_diffusion_model"])

