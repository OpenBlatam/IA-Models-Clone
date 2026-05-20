from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial

    from sklearn.datasets import make_classification
    import time
from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    max_sequence_length: int = 512
    num_folds: int = 5
    stratified: bool = True

class OptimizedDataset(Dataset):
    """Optimized dataset with efficient data loading and caching."""
    
    def __init__(self, data: Union[np.ndarray, List], labels: Optional[Union[np.ndarray, List]] = None, 
                 transform=None, cache_data: bool = True):
        
    """__init__ function."""
self.data = torch.tensor(data, dtype=torch.float32) if isinstance(data, np.ndarray) else data
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.transform = transform
        self.cache_data = cache_data
        
        if cache_data and len(self.data) < 10000:  # Cache small datasets in memory
            self.data = self.data.to('cpu')
            if self.labels is not None:
                self.labels = self.labels.to('cpu')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        if self.labels is not None:
            return item, self.labels[idx]
        return item

class DataLoaderManager:
    """Manages efficient data loading with proper splits and cross-validation."""
    
    def __init__(self, config: DataConfig):
        
    """__init__ function."""
self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fold_splits = []
        
    def create_data_loaders(self, data: np.ndarray, labels: Optional[np.ndarray] = None,
                           dataset_name: str = "dataset") -> Dict[str, DataLoader]:
        """Create train/validation/test data loaders with proper splits."""
        
        # Preprocess data
        data_processed, labels_processed = self._preprocess_data(data, labels)
        
        # Create dataset
        dataset = OptimizedDataset(data_processed, labels_processed, cache_data=True)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Create data loaders with optimized settings
        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_loader(val_dataset, shuffle=False)
        test_loader = self._create_loader(test_dataset, shuffle=False)
        
        logger.info(f"Created data loaders for {dataset_name}: "
                   f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'datasets': {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
        }
    
    def create_cross_validation_loaders(self, data: np.ndarray, labels: np.ndarray,
                                      dataset_name: str = "dataset") -> List[Dict[str, DataLoader]]:
        """Create cross-validation data loaders."""
        
        # Preprocess data
        data_processed, labels_processed = self._preprocess_data(data, labels)
        
        # Create dataset
        dataset = OptimizedDataset(data_processed, labels_processed, cache_data=True)
        
        # Initialize cross-validation
        if self.config.stratified:
            kfold = StratifiedKFold(n_splits=self.config.num_folds, shuffle=True, 
                                  random_state=self.config.random_seed)
        else:
            kfold = KFold(n_splits=self.config.num_folds, shuffle=True, 
                         random_state=self.config.random_seed)
        
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(data_processed, labels_processed)):
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            
            train_loader = self._create_loader(train_dataset, shuffle=True)
            val_loader = self._create_loader(val_dataset, shuffle=False)
            
            fold_loaders.append({
                'fold': fold,
                'train': train_loader,
                'val': val_loader,
                'train_indices': train_idx,
                'val_indices': val_idx
            })
            
            logger.info(f"Fold {fold}: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        self.fold_splits = fold_loaders
        return fold_loaders
    
    def create_weighted_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        """Create weighted sampler for imbalanced datasets."""
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))
    
    def _preprocess_data(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data with scaling and encoding."""
        
        # Scale features
        if data.ndim == 2:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = data
        
        # Encode labels if provided
        labels_encoded = None
        if labels is not None:
            labels_encoded = self.label_encoder.fit_transform(labels)
        
        return data_scaled, labels_encoded
    
    def _create_loader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create optimized DataLoader with best practices."""
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers
        )
    
    def save_preprocessors(self, filepath: str):
        """Save preprocessors for later use."""
        preprocessors = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        with open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            pickle.dump(preprocessors, f)
        logger.info(f"Saved preprocessors to {filepath}")
    
    def load_preprocessors(self, filepath: str):
        """Load preprocessors from file."""
        with open(filepath, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            preprocessors = pickle.load(f)
        self.scaler = preprocessors['scaler']
        self.label_encoder = preprocessors['label_encoder']
        logger.info(f"Loaded preprocessors from {filepath}")

class AsyncDataLoader:
    """Asynchronous data loader for high-throughput scenarios."""
    
    def __init__(self, data_loader: DataLoader, max_queue_size: int = 100):
        
    """__init__ function."""
self.data_loader = data_loader
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
    
    async def start(self) -> Any:
        """Start async data loading."""
        self._running = True
        asyncio.create_task(self._load_data())
    
    async def stop(self) -> Any:
        """Stop async data loading."""
        self._running = False
    
    async def _load_data(self) -> Any:
        """Load data asynchronously."""
        loop = asyncio.get_event_loop()
        
        for batch in self.data_loader:
            if not self._running:
                break
            
            # Load data in thread pool to avoid blocking
            await loop.run_in_executor(None, lambda: asyncio.create_task(self.queue.put(batch)))
    
    async def get_batch(self) -> Optional[Dict[str, Any]]:
        """Get next batch asynchronously."""
        return await self.queue.get()

# Utility functions for data loading
def load_csv_data(filepath: str, target_column: str = None, 
                  feature_columns: List[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    
    if target_column and target_column in df.columns:
        labels = df[target_column].values
        if feature_columns:
            features = df[feature_columns].values
        else:
            features = df.drop(columns=[target_column]).values
    else:
        features = df.values
        labels = None
    
    return features, labels

def create_synthetic_dataset(n_samples: int = 10000, n_features: int = 100, 
                           n_classes: int = 2, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic dataset for testing."""
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features // 2,
        random_state=42,
        noise=noise
    )
    
    return X, y

def benchmark_data_loading(data_loader: DataLoader, num_epochs: int = 3) -> Dict[str, float]:
    """Benchmark data loading performance."""
    
    start_time = time.time()
    total_batches = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch in data_loader:
            total_batches += 1
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    avg_batch_time = total_time / total_batches
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'batches_per_second': total_batches / total_time,
        'total_batches': total_batches
    }

# Example usage
if __name__ == "__main__":
    # Create synthetic data
    X, y = create_synthetic_dataset(n_samples=10000, n_features=100, n_classes=3)
    
    # Initialize data loader manager
    config = DataConfig(
        batch_size=64,
        num_workers=4,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_folds=5
    )
    
    manager = DataLoaderManager(config)
    
    # Create regular data loaders
    loaders = manager.create_data_loaders(X, y, "synthetic_data")
    
    # Create cross-validation loaders
    cv_loaders = manager.create_cross_validation_loaders(X, y, "synthetic_data")
    
    # Benchmark performance
    benchmark_results = benchmark_data_loading(loaders['train'])
    logger.info(f"Benchmark results: {benchmark_results}")
    
    # Save preprocessors
    manager.save_preprocessors("preprocessors.pkl") 