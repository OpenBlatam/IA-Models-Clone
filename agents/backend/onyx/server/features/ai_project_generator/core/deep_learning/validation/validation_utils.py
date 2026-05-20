"""
Validation Utilities
====================

Advanced validation utilities.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


def validate_dataset(
    dataset: Dataset,
    check_labels: bool = True,
    check_data_types: bool = True,
    check_shapes: bool = True
) -> Dict[str, Any]:
    """
    Validate dataset.
    
    Args:
        dataset: PyTorch dataset
        check_labels: Check label consistency
        check_data_types: Check data types
        check_shapes: Check data shapes
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if len(dataset) == 0:
        results['valid'] = False
        results['errors'].append("Dataset is empty")
        return results
    
    # Sample validation
    sample = dataset[0]
    
    if isinstance(sample, (list, tuple)):
        data, label = sample[0], sample[1] if len(sample) > 1 else None
    elif isinstance(sample, dict):
        data = sample.get('input_ids', sample.get('inputs', sample.get('data', None)))
        label = sample.get('labels', sample.get('target', None))
    else:
        data = sample
        label = None
    
    # Check data types
    if check_data_types:
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            results['warnings'].append(f"Data is not tensor/array: {type(data)}")
        if label is not None and not isinstance(label, (torch.Tensor, np.ndarray, int, float)):
            results['warnings'].append(f"Label is not tensor/array/scalar: {type(label)}")
    
    # Check shapes
    if check_shapes and isinstance(data, (torch.Tensor, np.ndarray)):
        results['stats']['data_shape'] = tuple(data.shape) if hasattr(data, 'shape') else None
        if label is not None and isinstance(label, (torch.Tensor, np.ndarray)):
            results['stats']['label_shape'] = tuple(label.shape) if hasattr(label, 'shape') else None
    
    # Check labels
    if check_labels and label is not None:
        # Sample more labels to check consistency
        labels = []
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            if isinstance(sample, (list, tuple)):
                lbl = sample[1] if len(sample) > 1 else None
            elif isinstance(sample, dict):
                lbl = sample.get('labels', sample.get('target', None))
            else:
                lbl = None
            
            if lbl is not None:
                if isinstance(lbl, torch.Tensor):
                    labels.append(lbl.item() if lbl.numel() == 1 else lbl.numpy())
                elif isinstance(lbl, np.ndarray):
                    labels.append(lbl.item() if lbl.size == 1 else lbl)
                else:
                    labels.append(lbl)
        
        if labels:
            unique_labels = set(labels)
            results['stats']['num_classes'] = len(unique_labels)
            results['stats']['unique_labels'] = sorted(list(unique_labels))
    
    results['stats']['dataset_size'] = len(dataset)
    
    if results['errors']:
        results['valid'] = False
    
    return results


def validate_model_config(
    config: Dict[str, Any],
    model_type: str
) -> Dict[str, Any]:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration
        model_type: Model type
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Type-specific validation
    if model_type == 'transformer':
        required = ['vocab_size', 'd_model']
        for key in required:
            if key not in config:
                results['errors'].append(f"Missing required parameter: {key}")
        
        if 'd_model' in config and 'num_heads' in config:
            if config['d_model'] % config['num_heads'] != 0:
                results['errors'].append("d_model must be divisible by num_heads")
    
    elif model_type == 'cnn':
        if 'input_channels' not in config:
            results['warnings'].append("input_channels not specified")
    
    elif model_type == 'rnn':
        if 'input_size' not in config:
            results['warnings'].append("input_size not specified")
    
    # General validation
    if 'dropout' in config:
        if not 0 <= config['dropout'] <= 1:
            results['errors'].append("dropout must be between 0 and 1")
    
    if results['errors']:
        results['valid'] = False
    
    return results


def validate_training_config(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required parameters
    if 'num_epochs' not in config:
        results['warnings'].append("num_epochs not specified")
    elif config['num_epochs'] <= 0:
        results['errors'].append("num_epochs must be positive")
    
    if 'batch_size' not in config:
        results['warnings'].append("batch_size not specified")
    elif config['batch_size'] <= 0:
        results['errors'].append("batch_size must be positive")
    
    if 'learning_rate' not in config:
        results['warnings'].append("learning_rate not specified")
    elif config['learning_rate'] <= 0:
        results['errors'].append("learning_rate must be positive")
    
    # Check optimizer
    if 'optimizer' in config:
        valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
        if config['optimizer'].lower() not in valid_optimizers:
            results['warnings'].append(f"Unknown optimizer: {config['optimizer']}")
    
    # Check scheduler
    if 'scheduler' in config:
        valid_schedulers = ['cosine', 'step', 'exponential', 'plateau']
        if config['scheduler'].lower() not in valid_schedulers:
            results['warnings'].append(f"Unknown scheduler: {config['scheduler']}")
    
    if results['errors']:
        results['valid'] = False
    
    return results


class ValidationSuite:
    """
    Comprehensive validation suite.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.validations = []
    
    def add_validation(self, name: str, validation_fn: callable) -> None:
        """
        Add validation function.
        
        Args:
            name: Validation name
            validation_fn: Validation function
        """
        self.validations.append({'name': name, 'fn': validation_fn})
    
    def run_all(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run all validations.
        
        Args:
            *args: Arguments for validation functions
            **kwargs: Keyword arguments for validation functions
            
        Returns:
            Validation results
        """
        results = {
            'all_passed': True,
            'validations': {}
        }
        
        for validation in self.validations:
            try:
                result = validation['fn'](*args, **kwargs)
                results['validations'][validation['name']] = result
                
                if isinstance(result, dict) and not result.get('valid', True):
                    results['all_passed'] = False
            except Exception as e:
                results['validations'][validation['name']] = {
                    'valid': False,
                    'error': str(e)
                }
                results['all_passed'] = False
        
        return results



