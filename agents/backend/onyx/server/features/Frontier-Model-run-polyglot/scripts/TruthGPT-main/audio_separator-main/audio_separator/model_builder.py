"""
Model builder for audio separator models.

Refactored to:
- Use constants for model types
- Registry pattern for model classes
- Better error handling
- Improved organization
"""

from typing import Optional, Dict, Any, Type
from .model.base_separator import BaseSeparatorModel
from .model.demucs_model import DemucsModel
from .model.spleeter_model import SpleeterModel
from .model.lalal_model import LalalModel
from .model.hybrid_model import HybridSeparatorModel
from .model.constants import (
    DEFAULT_MODEL_TYPE,
    VALID_MODEL_TYPES,
    ERROR_CODE_INVALID_MODEL_TYPE,
    ERROR_CODE_UNEXPECTED_MODEL_TYPE,
    ERROR_CODE_MODEL_BUILD_FAILED
)
from .exceptions import AudioModelError, AudioValidationError
from .logger import logger

# ════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ════════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: Dict[str, Type[BaseSeparatorModel]] = {
    "demucs": DemucsModel,
    "spleeter": SpleeterModel,
    "lalal": LalalModel,
    "hybrid": HybridSeparatorModel,
}


def register_model(model_type: str, model_class: Type[BaseSeparatorModel]) -> None:
    """
    Register a new model type.
    
    Args:
        model_type: Model type identifier
        model_class: Model class to register
    """
    _MODEL_REGISTRY[model_type.lower()] = model_class
    logger.info(f"Registered model type: {model_type}")


def get_registered_models() -> list:
    """
    Get list of registered model types.
    
    Returns:
        List of registered model type names
    """
    return list(_MODEL_REGISTRY.keys())


# ════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_audio_separator_model(
    model_type: str = DEFAULT_MODEL_TYPE,
    **kwargs
) -> BaseSeparatorModel:
    """
    Build an audio separator model.
    
    Args:
        model_type: Type of model to build ('demucs', 'spleeter', 'lalal', 'hybrid')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized separator model
        
    Raises:
        AudioValidationError: If model_type is invalid
        AudioModelError: If model initialization fails
        
    Examples:
        >>> model = build_audio_separator_model('demucs', variant='htdemucs')
        >>> model = build_audio_separator_model('spleeter', stems=4)
        >>> model = build_audio_separator_model('hybrid', models=['demucs', 'spleeter'])
    """
    model_type = model_type.lower()
    
    # Validate model type
    if model_type not in _MODEL_REGISTRY:
        raise AudioValidationError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(_MODEL_REGISTRY.keys())}",
            component="ModelBuilder",
            error_code=ERROR_CODE_INVALID_MODEL_TYPE
        )
    
    logger.info(f"Building {model_type} model with kwargs: {kwargs}")
    
    try:
        # Get model class from registry
        model_class = _MODEL_REGISTRY[model_type]
        
        # Build model
        model = model_class(**kwargs)
        
        logger.info(f"Successfully built {model_type} model")
        return model
        
    except (AudioValidationError, AudioModelError):
        raise
    except Exception as e:
        raise AudioModelError(
            f"Failed to build {model_type} model: {str(e)}",
            component="ModelBuilder",
            error_code=ERROR_CODE_MODEL_BUILD_FAILED,
            details={"model_type": model_type, "kwargs": kwargs, "error": str(e)}
        ) from e

