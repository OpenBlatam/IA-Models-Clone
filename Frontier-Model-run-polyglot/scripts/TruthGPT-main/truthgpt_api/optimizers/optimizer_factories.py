"""
Optimizer Factory Functions
===========================

Factory functions for creating optimizers from different sources:
- TensorFlow-inspired optimizers
- Core unified optimizers
- PyTorch optimizers (fallback)

Single Responsibility: Create optimizer instances from various sources.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Dict

from .core_detector import is_module_available, is_optimization_core_available
from .parameter_mapper import OptimizerParameterMapper
from .optimizer_constants import DEFAULT_LEARNING_RATE, AMSGRAD_SUPPORTED, TENSORFLOW_OPTIMIZER_TYPES, normalize_optimizer_type

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════


# Module paths for optimization_core
TENSORFLOW_OPTIMIZER_PATH = (
    'optimization_core.optimizers.tensorflow.tensorflow_inspired_optimizer.TensorFlowInspiredOptimizer'
)
UNIFIED_OPTIMIZER_PATH = 'optimization_core.optimizers.core.unified_optimizer.UnifiedOptimizer'

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

# Mapping of optimizer types to PyTorch optimizer classes (initialized lazily)
PYTORCH_OPTIMIZER_MAP = {
    'adam': None,  # Will be set dynamically when torch.optim is imported
    'sgd': None,
    'rmsprop': None,
    'adagrad': None,
    'adamw': None,
}


def _create_optimizer_from_module(
    module_name: str,
    class_path: str,
    optimizer_type: str,
    learning_rate: float,
    **kwargs
) -> Optional[Any]:
    """
    Generic function to create optimizer from a module using dynamic imports.
    
    Args:
        module_name: Name of the module to check availability ('tensorflow' or 'core')
        class_path: Full class path (e.g., 'optimization_core.optimizers.tensorflow.tensorflow_inspired_optimizer.TensorFlowInspiredOptimizer')
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer instance or None if not available
    """
    if not is_module_available(module_name):
        return None
    
    try:
        from importlib import import_module
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        optimizer_class = getattr(module, class_name)
        return optimizer_class(
            learning_rate=learning_rate,
            optimizer_type=optimizer_type.lower(),
            **kwargs
        )
    except (ImportError, AttributeError, TypeError) as e:
        logger.debug(f"{class_name} creation failed: {e}")
        return None


def create_tensorflow_optimizer(
    optimizer_type: str,
    learning_rate: float,
    **kwargs
) -> Optional[Any]:
    """
    Create a TensorFlow-inspired optimizer from optimization_core.
    
    Args:
        optimizer_type: Type of optimizer (adam, sgd, etc.)
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer instance or None if not available
    """
    return _create_optimizer_from_module(
        module_name='tensorflow',
        class_path=TENSORFLOW_OPTIMIZER_PATH,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **kwargs
    )


def create_core_optimizer(
    optimizer_type: str,
    learning_rate: float,
    **kwargs
) -> Optional[Any]:
    """
    Create a unified optimizer from optimization_core.
    
    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer instance or None if not available
    """
    return _create_optimizer_from_module(
        module_name='core',
        class_path=UNIFIED_OPTIMIZER_PATH,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **kwargs
    )


def create_optimizer_from_core(
    optimizer_type: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    **kwargs
) -> Optional[Any]:
    """
    Create an optimizer from optimization_core.
    
    Tries different optimization_core modules in order:
    1. TensorFlow-inspired optimizers (for adam, sgd)
    2. Core unified optimizer (fallback)
    
    Args:
        optimizer_type: Type of optimizer (adam, sgd, rmsprop, etc.)
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer instance or None if not available
    """
    if not is_optimization_core_available():
        return None
    
    optimizer_type = normalize_optimizer_type(optimizer_type)
    
    # Try tensorflow optimizers first (for specific types)
    if optimizer_type in TENSORFLOW_OPTIMIZER_TYPES:
        optimizer = create_tensorflow_optimizer(optimizer_type, learning_rate, **kwargs)
        if optimizer is not None:
            return optimizer
    
    # Try core unified optimizer as fallback
    return create_core_optimizer(optimizer_type, learning_rate, **kwargs)


def create_pytorch_optimizer(
    optimizer_type: str,
    parameters: Any,
    learning_rate: float,
    **kwargs
) -> Any:
    """
    Create a PyTorch optimizer as fallback.
    
    Uses OptimizerParameterMapper for consistent parameter mapping.
    
    Args:
        optimizer_type: Type of optimizer
        parameters: Model parameters to optimize
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        PyTorch optimizer instance
    
    Raises:
        ValueError: If optimizer type is not supported
        ImportError: If PyTorch is not available
    """
    # Initialize PyTorch optimizer map if not already done
    _initialize_pytorch_map()
    
    # Use OptimizerParameterMapper for consistent parameter mapping
    pytorch_kwargs = OptimizerParameterMapper.map_to_pytorch(
        optimizer_type,
        learning_rate,
        **kwargs
    )
    
    opt_type = optimizer_type.lower()
    
    # Ensure AMSGrad is properly set for supported optimizers
    _apply_amsgrad_config(opt_type, pytorch_kwargs, kwargs)
    
    optimizer_class = PYTORCH_OPTIMIZER_MAP.get(opt_type)
    if optimizer_class is None:
        supported = ', '.join(PYTORCH_OPTIMIZER_MAP.keys())
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Supported types: {supported}")
    
    return optimizer_class(parameters, lr=learning_rate, **pytorch_kwargs)


def _apply_amsgrad_config(
    optimizer_type: str,
    pytorch_kwargs: Dict[str, Any],
    original_kwargs: Dict[str, Any]
) -> None:
    """
    Apply AMSGrad configuration to PyTorch kwargs if applicable.
    
    Args:
        optimizer_type: Type of optimizer (lowercase)
        pytorch_kwargs: Dictionary of PyTorch kwargs (modified in place)
        original_kwargs: Original kwargs to check for AMSGrad setting
    """
    if optimizer_type not in AMSGRAD_SUPPORTED:
        return
    
    if 'amsgrad' not in pytorch_kwargs:
        pytorch_kwargs['amsgrad'] = original_kwargs.get('amsgrad', False)
    
    if pytorch_kwargs.get('amsgrad', False):
        logger.debug(f"✅ Using AMSGrad variant for {optimizer_type}")


def _initialize_pytorch_map() -> None:
    """Initialize PyTorch optimizer map if not already done."""
    if PYTORCH_OPTIMIZER_MAP['adam'] is None:
        import torch.optim as optim
        PYTORCH_OPTIMIZER_MAP.update({
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adamw': optim.AdamW,
        })


def _map_tensorflow_to_pytorch_params(
    optimizer_type: str,
    learning_rate: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Map TensorFlow-like parameter names to PyTorch parameter names.
    
    Uses OptimizerParameterMapper for consistency.
    
    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        **kwargs: Dictionary with TensorFlow-like parameters
    
    Returns:
        Dictionary with PyTorch-compatible parameters
    """
    return OptimizerParameterMapper.map_to_pytorch(optimizer_type, learning_rate, **kwargs)

