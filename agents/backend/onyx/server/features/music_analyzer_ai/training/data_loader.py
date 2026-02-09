"""
Optimized Data Loader for Music Analysis Training
Efficient data loading with caching, augmentation, and batching
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import hashlib

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class MusicSample:
    """Music sample for training"""
    audio_path: str
    features: np.ndarray
    label: int
    metadata: Dict[str, Any]
    sample_id: str


class MusicDataset(Dataset):
    """
    PyTorch Dataset for music analysis
    Supports caching, augmentation, and efficient loading
    """
    
    def __init__(
        self,
        samples: List[MusicSample],
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.samples = samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.augment = augment
        self.augment_prob = augment_prob
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Try to load from cache
        if self.use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{sample.sample_id}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                    features = cached_data["features"]
                except Exception:
                    features = sample.features
            else:
                features = sample.features
                # Cache it
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump({"features": features}, f)
                except Exception:
                    pass
        else:
            features = sample.features
        
        # Augmentation
        if self.augment and np.random.random() < self.augment_prob:
            features = self._augment_features(features)
        
        # Convert to tensor
        if TORCH_AVAILABLE:
            features_tensor = torch.FloatTensor(features)
            label_tensor = torch.LongTensor([sample.label])
        else:
            features_tensor = features
            label_tensor = sample.label
        
        return {
            "features": features_tensor,
            "label": label_tensor,
            "sample_id": sample.sample_id,
            "metadata": sample.metadata
        }
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation to features"""
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
        
        # Scale
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            features = features * scale
        
        return features


class MusicDataLoader:
    """
    Optimized DataLoader with:
    - Efficient batching
    - Caching
    - Augmentation
    - Weighted sampling
    - Prefetching
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.cache_dir = cache_dir
    
    def create_loader(
        self,
        dataset: MusicDataset,
        weighted_sampling: bool = False,
        class_weights: Optional[Dict[int, float]] = None
    ) -> DataLoader:
        """
        Create optimized DataLoader with best practices for deep learning
        
        Args:
            dataset: MusicDataset instance
            weighted_sampling: Whether to use weighted sampling for imbalanced data
            class_weights: Dictionary mapping class indices to weights
        
        Returns:
            Optimized DataLoader instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoader")
        
        # Weighted sampling for imbalanced datasets
        sampler = None
        if weighted_sampling and class_weights:
            weights = [class_weights.get(sample.label, 1.0) for sample in dataset.samples]
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            shuffle = False
        else:
            shuffle = self.shuffle
        
        # Create DataLoader with optimized settings
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),  # Only pin if CUDA available
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,  # For consistent batch sizes
            timeout=0 if self.num_workers > 0 else None,  # No timeout for data loading
            generator=None  # Use default generator for reproducibility
        )


def create_train_val_test_split(
    samples: List[MusicSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True
) -> Tuple[List[MusicSample], List[MusicSample], List[MusicSample]]:
    """Create train/validation/test splits"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if shuffle:
        import random
        random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples

