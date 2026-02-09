"""
Unified Models System
=====================
Centralized access to all model-related classes in optimization_core.
"""

# Import model managers
from .model_manager import ModelManager

try:
    from .model_builder import ModelBuilder
except ImportError:
    ModelBuilder = None

# Import model utilities
try:
    from .attention_utils import (
        AttentionUtils,
    )
except ImportError:
    AttentionUtils = None

# Import diffusion models
try:
    from .diffusion_manager import (
        DiffusionManager,
    )
except ImportError:
    DiffusionManager = None

# Import HuggingFace integrations
try:
    from .hf_transformers import (
        HFTransformersModel,
        create_hf_transformers_model,
    )
except ImportError:
    HFTransformersModel = None
    create_hf_transformers_model = None

try:
    from .hf_diffusers import (
        HFDiffusersModel,
        create_hf_diffusers_model,
    )
except ImportError:
    HFDiffusersModel = None
    create_hf_diffusers_model = None


# Unified model factory
def create_model(
    model_type: str = "manager",
    config: dict = None
):
    """
    Unified factory function to create models or model managers.
    
    Args:
        model_type: Type of model to create. Options:
            - "manager" - ModelManager
            - "builder" - ModelBuilder
            - "diffusion" - DiffusionManager
            - "hf_transformers" - HFTransformersModel
            - "hf_diffusers" - HFDiffusersModel
        config: Optional configuration dictionary
    
    Returns:
        The requested model instance
    """
    if config is None:
        config = {}
    
    model_type = model_type.lower()
    
    factory_map = {
        "manager": lambda cfg: ModelManager(**cfg),
        "builder": lambda cfg: ModelBuilder(cfg) if ModelBuilder else None,
        "diffusion": lambda cfg: DiffusionManager(cfg) if DiffusionManager else None,
        "hf_transformers": lambda cfg: create_hf_transformers_model(cfg) if create_hf_transformers_model else None,
        "hf_diffusers": lambda cfg: create_hf_diffusers_model(cfg) if create_hf_diffusers_model else None,
    }
    
    if model_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[model_type]
    model = factory(config)
    
    if model is None:
        raise ImportError(f"Model type '{model_type}' is not available (module not found)")
    
    return model


# Registry of all available models
MODEL_REGISTRY = {
    "manager": {
        "class": ModelManager,
        "module": "models.model_manager",
        "description": "Model manager for loading and saving models",
    },
    "builder": {
        "class": ModelBuilder,
        "module": "models.model_builder",
        "description": "Model builder for constructing models",
    },
    "diffusion": {
        "class": DiffusionManager,
        "module": "models.diffusion_manager",
        "description": "Diffusion model manager",
    },
    "hf_transformers": {
        "class": HFTransformersModel,
        "module": "models.hf_transformers",
        "description": "HuggingFace Transformers model",
    },
    "hf_diffusers": {
        "class": HFDiffusersModel,
        "module": "models.hf_diffusers",
        "description": "HuggingFace Diffusers model",
    },
}


def list_available_models() -> list:
    """List all available model types."""
    return [k for k, v in MODEL_REGISTRY.items() if v["class"] is not None]


def get_model_info(model_type: str) -> dict:
    """
    Get information about a specific model.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary with model information
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    registry_entry = MODEL_REGISTRY[model_type]
    
    if registry_entry["class"] is None:
        raise ImportError(f"Model type '{model_type}' is not available (module not found)")
    
    return {
        "type": model_type,
        "class": registry_entry["class"].__name__,
        "module": registry_entry["module"],
        "description": registry_entry["description"],
    }


__all__ = [
    # Model managers
    "ModelManager",
    "ModelBuilder",
    # Model utilities
    "AttentionUtils",
    # Diffusion models
    "DiffusionManager",
    # HuggingFace integrations
    "HFTransformersModel",
    "create_hf_transformers_model",
    "HFDiffusersModel",
    "create_hf_diffusers_model",
    # Unified factory
    "create_model",
    # Registry
    "MODEL_REGISTRY",
    "list_available_models",
    "get_model_info",
]
