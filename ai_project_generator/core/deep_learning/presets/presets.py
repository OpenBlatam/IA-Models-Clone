"""
Presets - Pre-configured Settings
==================================

Pre-configured settings for common deep learning scenarios.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


# Model Presets
MODEL_PRESETS = {
    'transformer_small': {
        'type': 'transformer',
        'vocab_size': 10000,
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 1024,
        'max_seq_len': 512,
        'dropout': 0.1
    },
    'transformer_medium': {
        'type': 'transformer',
        'vocab_size': 30000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1
    },
    'transformer_large': {
        'type': 'transformer',
        'vocab_size': 50000,
        'd_model': 1024,
        'num_heads': 16,
        'num_layers': 12,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1
    },
    'cnn_small': {
        'type': 'cnn',
        'in_channels': 3,
        'num_classes': 10,
        'conv_channels': [32, 64, 128],
        'use_residual': False,
        'dropout': 0.5
    },
    'cnn_medium': {
        'type': 'cnn',
        'in_channels': 3,
        'num_classes': 100,
        'conv_channels': [64, 128, 256, 512],
        'use_residual': True,
        'dropout': 0.5
    },
    'rnn_small': {
        'type': 'rnn',
        'vocab_size': 10000,
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'rnn_type': 'lstm',
        'bidirectional': True,
        'dropout': 0.3
    }
}

# Training Presets
TRAINING_PRESETS = {
    'fast': {
        'num_epochs': 5,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'scheduler': 'none',
        'use_mixed_precision': False,
        'gradient_accumulation_steps': 1
    },
    'standard': {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'early_stopping_patience': 5
    },
    'production': {
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'early_stopping_patience': 10,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01
    },
    'large_batch': {
        'num_epochs': 20,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'optimizer': 'adamw',
        'scheduler': 'linear',
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 8
    }
}

# Optimizer Presets
OPTIMIZER_PRESETS = {
    'adam_fast': {
        'optimizer_type': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'betas': (0.9, 0.999)
    },
    'adamw_standard': {
        'optimizer_type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999)
    },
    'sgd_momentum': {
        'optimizer_type': 'sgd',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True
    }
}

# Data Presets
DATA_PRESETS = {
    'small': {
        'batch_size': 32,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 2
    },
    'medium': {
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 3
    },
    'large': {
        'batch_size': 128,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4
    }
}


def get_model_preset(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model preset configuration.
    
    Args:
        name: Preset name
        overrides: Configuration overrides
        
    Returns:
        Model configuration dictionary
    """
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {name}. Available: {list(MODEL_PRESETS.keys())}")
    
    config = MODEL_PRESETS[name].copy()
    
    if overrides:
        config.update(overrides)
    
    logger.info(f"Using model preset: {name}")
    return config


def get_training_preset(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get training preset configuration.
    
    Args:
        name: Preset name
        overrides: Configuration overrides
        
    Returns:
        Training configuration dictionary
    """
    if name not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset: {name}. Available: {list(TRAINING_PRESETS.keys())}")
    
    config = TRAINING_PRESETS[name].copy()
    
    if overrides:
        config.update(overrides)
    
    logger.info(f"Using training preset: {name}")
    return config


def get_optimizer_preset(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get optimizer preset configuration.
    
    Args:
        name: Preset name
        overrides: Configuration overrides
        
    Returns:
        Optimizer configuration dictionary
    """
    if name not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown optimizer preset: {name}. Available: {list(OPTIMIZER_PRESETS.keys())}")
    
    config = OPTIMIZER_PRESETS[name].copy()
    
    if overrides:
        config.update(overrides)
    
    logger.info(f"Using optimizer preset: {name}")
    return config


def get_data_preset(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get data preset configuration.
    
    Args:
        name: Preset name
        overrides: Configuration overrides
        
    Returns:
        Data configuration dictionary
    """
    if name not in DATA_PRESETS:
        raise ValueError(f"Unknown data preset: {name}. Available: {list(DATA_PRESETS.keys())}")
    
    config = DATA_PRESETS[name].copy()
    
    if overrides:
        config.update(overrides)
    
    logger.info(f"Using data preset: {name}")
    return config


def list_presets(preset_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available presets.
    
    Args:
        preset_type: Type of preset ('model', 'training', 'optimizer', 'data') or None for all
        
    Returns:
        Dictionary of preset types and names
    """
    all_presets = {
        'model': list(MODEL_PRESETS.keys()),
        'training': list(TRAINING_PRESETS.keys()),
        'optimizer': list(OPTIMIZER_PRESETS.keys()),
        'data': list(DATA_PRESETS.keys())
    }
    
    if preset_type:
        if preset_type not in all_presets:
            raise ValueError(f"Unknown preset type: {preset_type}")
        return {preset_type: all_presets[preset_type]}
    
    return all_presets



