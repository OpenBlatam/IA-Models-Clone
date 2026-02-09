"""
Data Configuration Module

Data loading configuration dataclasses.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    shuffle: bool = True
    cache_dir: Optional[str] = None
    use_augmentation: bool = False
    augmentation_prob: float = 0.5



