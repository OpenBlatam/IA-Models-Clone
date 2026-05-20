"""
Advanced Data Pipeline for Recovery AI
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import logging
from functools import partial

logger = logging.getLogger(__name__)


class RecoveryDataset(Dataset):
    """Dataset for recovery data"""
    
    def __init__(
        self,
        features: List[Dict[str, float]],
        targets: Optional[List[float]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset
        
        Args:
            features: List of feature dictionaries
            targets: Optional target values
            transform: Optional transform function
        """
        self.features = features
        self.targets = targets if targets is not None else [0.0] * len(features)
        self.transform = transform
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get item"""
        feature_dict = self.features[idx]
        
        # Convert to tensor
        feature_list = [
            feature_dict.get("days_sober", 0) / 365.0,
            feature_dict.get("cravings_level", 5) / 10.0,
            feature_dict.get("stress_level", 5) / 10.0,
            feature_dict.get("support_level", 5) / 10.0,
            feature_dict.get("mood_score", 5) / 10.0,
            feature_dict.get("sleep_quality", 5) / 10.0,
            feature_dict.get("exercise_frequency", 2) / 7.0,
            feature_dict.get("therapy_sessions", 0) / 10.0,
            feature_dict.get("medication_compliance", 1.0),
            feature_dict.get("social_activity", 3) / 7.0
        ]
        
        features_tensor = torch.tensor(feature_list, dtype=torch.float32)
        target_tensor = torch.tensor([self.targets[idx]], dtype=torch.float32)
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        return features_tensor, target_tensor


class SequenceDataset(Dataset):
    """Dataset for sequence data"""
    
    def __init__(
        self,
        sequences: List[List[Dict[str, float]]],
        targets: Optional[List[float]] = None,
        max_length: int = 30
    ):
        """
        Initialize sequence dataset
        
        Args:
            sequences: List of sequences
            targets: Optional target values
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.targets = targets if targets is not None else [0.0] * len(sequences)
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get item"""
        sequence = self.sequences[idx]
        
        # Convert to tensor
        seq_data = []
        for day in sequence[-self.max_length:]:
            seq_data.append([
                day.get("cravings_level", 5) / 10.0,
                day.get("stress_level", 5) / 10.0,
                day.get("mood_score", 5) / 10.0,
                day.get("triggers_count", 0) / 10.0,
                day.get("consumed", 0.0)
            ])
        
        # Pad to fixed length
        while len(seq_data) < self.max_length:
            seq_data.insert(0, [0.0] * 5)
        
        sequence_tensor = torch.tensor(seq_data, dtype=torch.float32)
        target_tensor = torch.tensor([self.targets[idx]], dtype=torch.float32)
        
        return sequence_tensor, target_tensor


class DataPipeline:
    """Advanced data pipeline"""
    
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        """
        Initialize data pipeline
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        
        logger.info("DataPipeline initialized")
    
    def create_loader(
        self,
        dataset: Dataset,
        **kwargs
    ) -> DataLoader:
        """
        Create optimized DataLoader
        
        Args:
            dataset: Dataset
            **kwargs: Additional DataLoader arguments
        
        Returns:
            Optimized DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=kwargs.get("batch_size", self.batch_size),
            shuffle=kwargs.get("shuffle", self.shuffle),
            num_workers=kwargs.get("num_workers", self.num_workers),
            pin_memory=kwargs.get("pin_memory", self.pin_memory),
            persistent_workers=kwargs.get("persistent_workers", self.persistent_workers),
            prefetch_factor=kwargs.get("prefetch_factor", 2) if self.num_workers > 0 else None
        )
    
    def create_train_val_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create train and validation loaders
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            train_batch_size: Training batch size
            val_batch_size: Validation batch size
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = self.create_loader(
            train_dataset,
            batch_size=train_batch_size or self.batch_size,
            shuffle=True
        )
        
        val_loader = self.create_loader(
            val_dataset,
            batch_size=val_batch_size or self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader


class DataAugmentationPipeline:
    """Data augmentation pipeline"""
    
    def __init__(
        self,
        augmentation_prob: float = 0.5,
        noise_level: float = 0.05
    ):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentation_prob: Probability of applying augmentation
            noise_level: Noise level
        """
        self.augmentation_prob = augmentation_prob
        self.noise_level = noise_level
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation"""
        if np.random.random() > self.augmentation_prob:
            return features
        
        # Add noise
        if np.random.random() < 0.5:
            noise = torch.randn_like(features) * self.noise_level
            features = features + noise
        
        # Scale
        if np.random.random() < 0.5:
            scale = torch.empty(features.shape).uniform_(0.9, 1.1)
            features = features * scale
        
        return features.clamp(0, 1)

