"""
Main API for Refactored Enhanced Transformer Models

This module provides the main API for the refactored
transformer architecture with simplified interfaces.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from .factories import ModelFactoryRegistry
from .management import EnhancedConfigManager, EnhancedModelManager, ModelRegistry
from ..transformer_config import TransformerConfig


# Global instances
_factory_registry = ModelFactoryRegistry()
_config_manager = EnhancedConfigManager()
_model_manager = EnhancedModelManager(_factory_registry)
_model_registry = ModelRegistry()


def create_transformer_model(config: TransformerConfig, 
                           model_type: str = "standard",
                           factory_name: str = "enhanced") -> nn.Module:
    """
    Create a transformer model based on configuration and type.
    
    Args:
        config: Transformer configuration
        model_type: Type of model to create
        factory_name: Factory to use for creation
        
    Returns:
        Created transformer model
    """
    return _model_manager.create_model(config, model_type, factory_name)


def create_attention_mechanism(attention_type: str, 
                              config: TransformerConfig) -> nn.Module:
    """
    Create an attention mechanism based on type and configuration.
    
    Args:
        attention_type: Type of attention mechanism
        config: Transformer configuration
        
    Returns:
        Created attention mechanism
    """
    factory = _factory_registry.get_factory("attention")
    return factory.create_attention(attention_type, config)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    return _model_manager.get_model_info(model)


def get_supported_types(factory_name: str = "enhanced") -> List[str]:
    """
    Get supported model types for a factory.
    
    Args:
        factory_name: Name of the factory
        
    Returns:
        List of supported types
    """
    return _factory_registry.get_supported_types(factory_name)


def load_config(config_path: str) -> TransformerConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    return _config_manager.load_config(config_path)


def save_config(config: TransformerConfig, config_path: str, format: str = 'json') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
        format: File format ('json' or 'yaml')
    """
    _config_manager.save_config(config, config_path, format)


def save_model(model: nn.Module, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model to file.
    
    Args:
        model: Model to save
        model_path: Path to save model
        metadata: Optional metadata to save
    """
    _model_manager.save_model(model, model_path, metadata)


def load_model(model_path: str, config: Optional[TransformerConfig] = None) -> nn.Module:
    """
    Load model from file.
    
    Args:
        model_path: Path to model file
        config: Optional configuration for model recreation
        
    Returns:
        Loaded model
    """
    return _model_manager.load_model(model_path, config)


def benchmark_model(model: nn.Module, input_shape: tuple, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape for benchmarking
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    return _model_manager.benchmark_model(model, input_shape, num_runs)


def optimize_model(model: nn.Module, optimization_type: str = "memory") -> nn.Module:
    """
    Optimize model for different criteria.
    
    Args:
        model: Model to optimize
        optimization_type: Type of optimization ('memory', 'speed', 'accuracy')
        
    Returns:
        Optimized model
    """
    return _model_manager.optimize_model(model, optimization_type)


def register_model(name: str, model: nn.Module, config: TransformerConfig) -> None:
    """
    Register a model in the registry.
    
    Args:
        name: Name for the model
        model: Model to register
        config: Model configuration
    """
    _model_registry.register_model(name, model, config)


def get_registered_model(name: str) -> Optional[nn.Module]:
    """
    Get a registered model by name.
    
    Args:
        name: Name of the model
        
    Returns:
        Registered model or None
    """
    return _model_registry.get_model(name)


def list_registered_models() -> List[str]:
    """
    List all registered model names.
    
    Returns:
        List of registered model names
    """
    return _model_registry.list_models()


class EnhancedTransformerAPI:
    """Main API class for the refactored transformer system."""
    
    def __init__(self):
        self.factory_registry = _factory_registry
        self.config_manager = _config_manager
        self.model_manager = _model_manager
        self.model_registry = _model_registry
    
    def create_model(self, config: TransformerConfig, model_type: str = "standard", factory_name: str = "enhanced") -> nn.Module:
        """Create a transformer model."""
        return create_transformer_model(config, model_type, factory_name)
    
    def create_attention(self, attention_type: str, config: TransformerConfig) -> nn.Module:
        """Create an attention mechanism."""
        return create_attention_mechanism(attention_type, config)
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information."""
        return get_model_info(model)
    
    def get_supported_types(self, factory_name: str = "enhanced") -> List[str]:
        """Get supported model types."""
        return get_supported_types(factory_name)
    
    def load_config(self, config_path: str) -> TransformerConfig:
        """Load configuration."""
        return load_config(config_path)
    
    def save_config(self, config: TransformerConfig, config_path: str, format: str = 'json') -> None:
        """Save configuration."""
        save_config(config, config_path, format)
    
    def save_model(self, model: nn.Module, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model."""
        save_model(model, model_path, metadata)
    
    def load_model(self, model_path: str, config: Optional[TransformerConfig] = None) -> nn.Module:
        """Load model."""
        return load_model(model_path, config)
    
    def benchmark_model(self, model: nn.Module, input_shape: tuple, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark model."""
        return benchmark_model(model, input_shape, num_runs)
    
    def optimize_model(self, model: nn.Module, optimization_type: str = "memory") -> nn.Module:
        """Optimize model."""
        return optimize_model(model, optimization_type)
    
    def register_model(self, name: str, model: nn.Module, config: TransformerConfig) -> None:
        """Register model."""
        register_model(name, model, config)
    
    def get_registered_model(self, name: str) -> Optional[nn.Module]:
        """Get registered model."""
        return get_registered_model(name)
    
    def list_registered_models(self) -> List[str]:
        """List registered models."""
        return list_registered_models()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "factories": self.factory_registry.list_factories(),
            "cached_configs": self.config_manager.get_cached_configs(),
            "cached_models": self.model_manager.get_cached_models(),
            "registered_models": self.model_registry.list_models()
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.config_manager.clear_cache()
        self.model_manager.clear_cache()
    
    def reset(self):
        """Reset the entire system."""
        self.clear_cache()
        self.model_registry.clear()

