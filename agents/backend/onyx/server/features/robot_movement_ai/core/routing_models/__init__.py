"""
Routing Models Package
======================

Módulos de modelos de deep learning para enrutamiento.
"""

from .base_model import BaseRouteModel, ModelConfig
from .mlp_model import MLPRoutePredictor
from .model_factory import ModelFactory
from .ensemble import ModelEnsemble, EnsembleBuilder

__all__ = [
    "BaseRouteModel",
    "ModelConfig",
    "MLPRoutePredictor",
    "ModelFactory",
    "ModelEnsemble",
    "EnsembleBuilder"
]

# Imports condicionales
try:
    from .lora import LoRALayer, LoRALinear, apply_lora_to_model, count_lora_parameters
    __all__.extend([
        "LoRALayer",
        "LoRALinear",
        "apply_lora_to_model",
        "count_lora_parameters"
    ])
except ImportError:
    pass

# Imports condicionales
try:
    from .gnn_model import GCNRoutePredictor, GATRoutePredictor
    __all__.extend(["GCNRoutePredictor", "GATRoutePredictor"])
except ImportError:
    pass

try:
    from .transformer_model import TransformerRouteModel
    __all__.append("TransformerRouteModel")
except ImportError:
    pass

try:
    from .rl_model import DQNRouteAgent
    __all__.append("DQNRouteAgent")
except ImportError:
    pass

