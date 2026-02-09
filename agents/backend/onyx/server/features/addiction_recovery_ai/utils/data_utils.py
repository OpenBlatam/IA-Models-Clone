"""
Data Utilities
Helper functions for data processing
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features
    
    Args:
        features: Feature array
        mean: Mean values (if None, computed)
        std: Std values (if None, computed)
        
    Returns:
        Normalized features, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std


def split_data(
    data: List[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train/val/test
    
    Args:
        data: Data list
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        shuffle: Shuffle data
        seed: Random seed
        
    Returns:
        Train, validation, test splits
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if seed is not None:
        np.random.seed(seed)
    
    if shuffle:
        data = data.copy()
        np.random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def create_sequences(
    data: List[Dict[str, Any]],
    sequence_length: int,
    feature_keys: List[str],
    target_key: Optional[str] = None
) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    """
    Create sequences from data
    
    Args:
        data: Data list
        sequence_length: Sequence length
        feature_keys: Feature keys
        target_key: Target key (optional)
        
    Returns:
        Sequences and optionally targets
    """
    sequences = []
    targets = [] if target_key else None
    
    for i in range(len(data) - sequence_length + 1):
        sequence = []
        for j in range(sequence_length):
            sample = data[i + j]
            features = [float(sample.get(key, 0.0)) for key in feature_keys]
            sequence.append(features)
        
        sequences.append(np.array(sequence))
        
        if target_key:
            target = float(data[i + sequence_length - 1].get(target_key, 0.0))
            targets.append(target)
    
    return sequences, targets


def augment_data(
    data: np.ndarray,
    noise_level: float = 0.01,
    num_augmentations: int = 1
) -> np.ndarray:
    """
    Augment data with noise
    
    Args:
        data: Data array
        noise_level: Noise level
        num_augmentations: Number of augmentations
        
    Returns:
        Augmented data
    """
    augmented = [data]
    
    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_level, data.shape)
        augmented.append(data + noise)
    
    return np.vstack(augmented)


def balance_dataset(
    data: List[Dict[str, Any]],
    target_key: str,
    method: str = "undersample"
) -> List[Dict[str, Any]]:
    """
    Balance dataset
    
    Args:
        data: Data list
        target_key: Target key
        method: Balancing method (undersample, oversample)
        
    Returns:
        Balanced data
    """
    # Get class distribution
    classes = {}
    for sample in data:
        label = sample.get(target_key)
        if label not in classes:
            classes[label] = []
        classes[label].append(sample)
    
    # Find minority class
    min_count = min(len(samples) for samples in classes.values())
    
    balanced = []
    
    if method == "undersample":
        # Undersample all classes to minority count
        for samples in classes.values():
            balanced.extend(np.random.choice(samples, min_count, replace=False))
    
    elif method == "oversample":
        # Oversample all classes to majority count
        max_count = max(len(samples) for samples in classes.values())
        for samples in classes.values():
            if len(samples) < max_count:
                # Oversample
                additional = np.random.choice(samples, max_count - len(samples), replace=True)
                balanced.extend(samples)
                balanced.extend(additional)
            else:
                balanced.extend(samples)
    
    return balanced


def create_cross_validation_splits(
    data: List[Any],
    n_splits: int = 5,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> List[Tuple[List[Any], List[Any]]]:
    """
    Create cross-validation splits
    
    Args:
        data: Data list
        n_splits: Number of splits
        shuffle: Shuffle data
        seed: Random seed
        
    Returns:
        List of (train, val) splits
    """
    if seed is not None:
        np.random.seed(seed)
    
    if shuffle:
        data = data.copy()
        np.random.shuffle(data)
    
    n = len(data)
    fold_size = n // n_splits
    
    splits = []
    for i in range(n_splits):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n
        
        val_data = data[val_start:val_end]
        train_data = data[:val_start] + data[val_end:]
        
        splits.append((train_data, val_data))
    
    return splits













