"""
Model Utilities for TruthGPT API
===============================

TensorFlow-like model utility functions.
"""

import torch
import pickle
import json
from typing import Any, Dict, Optional


def save_model(model: Any, filepath: str, save_format: str = 'pytorch'):
    """
    Save model to file.
    
    Similar to tf.keras.models.save_model, this function saves
    a model to the specified filepath.
    
    Args:
        model: Model to save
        filepath: Path to save the model
        save_format: Format to save in ('pytorch', 'json')
    """
    if save_format == 'pytorch':
        # Save PyTorch model
        torch.save(model.state_dict(), filepath)
    elif save_format == 'json':
        # Save model configuration as JSON
        config = {
            'model_type': type(model).__name__,
            'config': getattr(model, 'get_config', lambda: {})()
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def load_model(filepath: str, 
               model_class: Optional[Any] = None,
               load_format: str = 'pytorch') -> Any:
    """
    Load model from file.
    
    Similar to tf.keras.models.load_model, this function loads
    a model from the specified filepath.
    
    Args:
        filepath: Path to load the model from
        model_class: Model class to instantiate
        load_format: Format to load from ('pytorch', 'json')
        
    Returns:
        Loaded model
    """
    if load_format == 'pytorch':
        # Load PyTorch model
        if model_class is None:
            raise ValueError("model_class must be provided for PyTorch format")
        
        model = model_class()
        model.load_state_dict(torch.load(filepath))
        return model
    elif load_format == 'json':
        # Load model configuration from JSON
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    else:
        raise ValueError(f"Unsupported load format: {load_format}")


def get_model_config(model: Any) -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        model: Model to get configuration for
        
    Returns:
        Model configuration dictionary
    """
    if hasattr(model, 'get_config'):
        return model.get_config()
    else:
        return {
            'model_type': type(model).__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }


