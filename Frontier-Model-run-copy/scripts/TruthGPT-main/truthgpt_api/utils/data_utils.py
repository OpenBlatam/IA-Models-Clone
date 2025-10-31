"""
Data Utilities for TruthGPT API
==============================

TensorFlow-like data utility functions.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional


def to_categorical(y: Union[np.ndarray, torch.Tensor], 
                   num_classes: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert class vector to binary class matrix.
    
    Similar to tf.keras.utils.to_categorical, this function converts
    class vectors to binary class matrices.
    
    Args:
        y: Class vector to be converted
        num_classes: Number of classes
        
    Returns:
        Binary class matrix
    """
    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
        is_tensor = True
    else:
        y_np = y
        is_tensor = False
    
    if num_classes is None:
        num_classes = np.max(y_np) + 1
    
    # Convert to categorical
    categorical = np.eye(num_classes)[y_np]
    
    if is_tensor:
        return torch.from_numpy(categorical).float()
    else:
        return categorical


def normalize(x: Union[np.ndarray, torch.Tensor], 
              axis: int = -1, 
              order: int = 2) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize input array.
    
    Similar to tf.keras.utils.normalize, this function normalizes
    the input array along the specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to normalize
        order: Order of the norm
        
    Returns:
        Normalized array
    """
    if isinstance(x, torch.Tensor):
        # PyTorch normalization
        norm = torch.norm(x, p=order, dim=axis, keepdim=True)
        return x / (norm + 1e-8)
    else:
        # NumPy normalization
        norm = np.linalg.norm(x, ord=order, axis=axis, keepdims=True)
        return x / (norm + 1e-8)


def get_data(filepath: str, 
             test_split: float = 0.2,
             validation_split: float = 0.2,
             random_seed: int = 42) -> Tuple:
    """
    Load and split data.
    
    Similar to tf.keras.datasets functions, this function loads
    data and splits it into train/validation/test sets.
    
    Args:
        filepath: Path to data file
        test_split: Fraction of data for testing
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # This is a placeholder implementation
    # In practice, you'd load data from the filepath
    
    # For demonstration, create dummy data
    np.random.seed(random_seed)
    
    # Generate dummy data
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    x = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_test - n_val
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data
    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    
    return (x_train, y_train, x_val, y_val, x_test, y_test)


