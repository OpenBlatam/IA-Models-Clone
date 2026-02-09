"""
Model Initialization Utilities
===============================

Main interface for model initialization, analysis, and management.

Refactored with modular architecture:
- initialization/: Weight initialization strategies
- analysis/: Model analysis and inspection
- management/: Layer and gradient management
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, List, Union, Callable

# Import refactored modules
from .initialization import WeightInitializer, InitializationStrategies
from .analysis import ModelAnalyzer, ParameterCounter
from .management import LayerManager, GradientManager

logger = logging.getLogger(__name__)


class ModelInitializer:
    """
    Main interface for model initialization and management.
    
    This class provides a unified interface to all model utilities:
    - Weight initialization with multiple strategies
    - Model analysis and inspection
    - Layer management (freezing/unfreezing)
    - Gradient management
    """
    
    # Initialization strategies (for backward compatibility)
    INIT_XAVIER = "xavier"
    INIT_HE = "he"
    INIT_KAIMING = "kaiming"
    INIT_ORTHOGONAL = "orthogonal"
    INIT_ZEROS = "zeros"
    INIT_ONES = "ones"
    INIT_NORMAL = "normal"
    
    @staticmethod
    def initialize_weights(
        module: nn.Module,
        strategy: str = INIT_XAVIER,
        **kwargs
    ) -> None:
        """
        Initialize model weights using specified strategy.
        
        Args:
            module: PyTorch module to initialize
            strategy: Initialization strategy
            **kwargs: Additional strategy-specific parameters
        """
        WeightInitializer.initialize_weights(module, strategy, **kwargs)
    
    @staticmethod
    def initialize_modules(
        modules: List[nn.Module],
        strategy: str = INIT_XAVIER,
        **kwargs
    ) -> None:
        """
        Initialize multiple modules.
        
        Args:
            modules: List of PyTorch modules to initialize
            strategy: Initialization strategy
            **kwargs: Additional parameters
        """
        for module in modules:
            WeightInitializer.initialize_weights(module, strategy, **kwargs)
    
    @staticmethod
    def count_parameters(
        model: nn.Module,
        trainable_only: bool = False,
        by_layer: bool = False
    ) -> Union[int, Dict[str, int]]:
        """
        Count model parameters with optional layer breakdown.
        
        Args:
            model: PyTorch model
            trainable_only: If True, only count trainable parameters
            by_layer: If True, return breakdown by layer
            
        Returns:
            Parameter count or dictionary by layer
        """
        return ParameterCounter.count_parameters(model, trainable_only, by_layer)
    
    @staticmethod
    def get_model_info(
        model: nn.Module,
        include_architecture: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: PyTorch model
            include_architecture: Include architecture details
            
        Returns:
            Dictionary with model information
        """
        return ModelAnalyzer.get_model_info(model, include_architecture)
    
    @staticmethod
    def move_to_device(
        module: nn.Module,
        device: torch.device
    ) -> nn.Module:
        """
        Move module to device.
        
        Args:
            module: PyTorch module
            device: Target device
            
        Returns:
            Module moved to device
        """
        return module.to(device)
    
    @staticmethod
    def freeze_layers(
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        freeze_all: bool = False
    ) -> None:
        """Freeze specified layers or all layers."""
        LayerManager.freeze_layers(model, layer_names, freeze_all)
    
    @staticmethod
    def unfreeze_layers(
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        unfreeze_all: bool = False
    ) -> None:
        """Unfreeze specified layers or all layers."""
        LayerManager.unfreeze_layers(model, layer_names, unfreeze_all)
    
    @staticmethod
    def get_frozen_layers(model: nn.Module) -> List[str]:
        """Get list of frozen layer names."""
        return LayerManager.get_frozen_layers(model)
    
    @staticmethod
    def clone_model(model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        return LayerManager.clone_model(model)
    
    @staticmethod
    def compare_models(
        model1: nn.Module,
        model2: nn.Module
    ) -> Dict[str, Any]:
        """Compare two models."""
        return ModelAnalyzer.compare_models(model1, model2)
    
    @staticmethod
    def get_gradient_norm(model: nn.Module) -> float:
        """Calculate gradient norm."""
        return GradientManager.get_gradient_norm(model)
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0
    ) -> float:
        """Clip gradients to prevent explosion."""
        return GradientManager.clip_gradients(model, max_norm)
    
    @staticmethod
    def zero_gradients(model: nn.Module) -> None:
        """Zero all gradients."""
        GradientManager.zero_gradients(model)
    
    @staticmethod
    def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
        """Get information about each layer."""
        return ModelAnalyzer.get_layer_info(model)
    
    @staticmethod
    def apply_to_layers(
        model: nn.Module,
        func: Callable[[nn.Module, str], None],
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> None:
        """Apply function to selected layers."""
        LayerManager.apply_to_layers(model, func, layer_filter)
    
    @staticmethod
    def export_model_summary(
        model: nn.Module,
        output_path: Optional[str] = None
    ) -> str:
        """Export model summary to string or file."""
        return ModelAnalyzer.export_model_summary(model, output_path)
    
    # Convenience access to sub-modules
    @staticmethod
    def get_initialization_strategies():
        """Get available initialization strategies."""
        return InitializationStrategies
    
    @staticmethod
    def get_parameter_counter():
        """Get parameter counter utility."""
        return ParameterCounter
    
    @staticmethod
    def get_model_analyzer():
        """Get model analyzer utility."""
        return ModelAnalyzer
    
    @staticmethod
    def get_layer_manager():
        """Get layer manager utility."""
        return LayerManager
    
    @staticmethod
    def get_gradient_manager():
        """Get gradient manager utility."""
        return GradientManager
