"""
Data Loading Service - Carga eficiente de datos
=================================================

Sistema profesional para carga eficiente de datos con PyTorch DataLoader.
Sigue mejores prácticas de data loading y preprocessing.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class DataLoaderConfig:
    """Configuración de DataLoader"""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    timeout: float = 0.0


@dataclass
class DataSplitConfig:
    """Configuración de división de datos"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify: bool = False  # For classification tasks


class DataLoadingService:
    """Servicio para carga eficiente de datos"""
    
    def __init__(self):
        """Inicializar servicio"""
        logger.info(f"DataLoadingService initialized (PyTorch: {TORCH_AVAILABLE})")
    
    def create_dataloader(
        self,
        dataset: Dataset,
        config: DataLoaderConfig,
        is_distributed: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ) -> DataLoader:
        """
        Crear DataLoader optimizado.
        
        Args:
            dataset: Dataset de PyTorch
            config: Configuración del DataLoader
            is_distributed: Si es entrenamiento distribuido
            rank: Rank del proceso (para distributed)
            world_size: Tamaño del mundo (para distributed)
        
        Returns:
            DataLoader optimizado
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Create sampler for distributed training
        sampler = None
        if is_distributed:
            if rank is None or world_size is None:
                raise ValueError("rank and world_size required for distributed training")
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=config.shuffle
            )
            # Don't shuffle in DataLoader when using DistributedSampler
            shuffle = False
        else:
            shuffle = config.shuffle
        
        # Determine number of workers
        # Use 0 workers on Windows or if pin_memory is False
        num_workers = config.num_workers
        if not config.pin_memory:
            num_workers = 0  # pin_memory requires workers > 0
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=config.pin_memory and torch.cuda.is_available(),
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
            persistent_workers=config.persistent_workers if num_workers > 0 else False,
            timeout=config.timeout,
        )
        
        logger.info(
            f"DataLoader created: batch_size={config.batch_size}, "
            f"num_workers={num_workers}, pin_memory={config.pin_memory}"
        )
        
        return dataloader
    
    def split_dataset(
        self,
        dataset: Dataset,
        config: DataSplitConfig
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Dividir dataset en train/val/test.
        
        Args:
            dataset: Dataset completo
            config: Configuración de división
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Validate ratios
        total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Calculate sizes
        total_size = len(dataset)
        train_size = int(config.train_ratio * total_size)
        val_size = int(config.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        logger.info(
            f"Dataset split: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_cross_validation_splits(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        seed: int = 42
    ) -> List[Tuple[Dataset, Dataset]]:
        """
        Crear splits para cross-validation.
        
        Args:
            dataset: Dataset completo
            n_splits: Número de folds
            seed: Seed para reproducibilidad
        
        Returns:
            Lista de (train_dataset, val_dataset) para cada fold
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        from sklearn.model_selection import KFold
        
        total_size = len(dataset)
        indices = np.arange(total_size)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        splits = []
        for train_indices, val_indices in kf.split(indices):
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            splits.append((train_subset, val_subset))
        
        logger.info(f"Created {n_splits} cross-validation splits")
        return splits
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Obtener información del dataset.
        
        Args:
            dataset: Dataset
        
        Returns:
            Información del dataset
        """
        info = {
            "size": len(dataset),
            "type": type(dataset).__name__,
        }
        
        # Try to get sample
        try:
            sample = dataset[0]
            if isinstance(sample, (tuple, list)):
                info["num_outputs"] = len(sample)
                if len(sample) > 0:
                    if isinstance(sample[0], torch.Tensor):
                        info["input_shape"] = list(sample[0].shape)
                    if len(sample) > 1 and isinstance(sample[1], torch.Tensor):
                        info["target_shape"] = list(sample[1].shape)
            elif isinstance(sample, torch.Tensor):
                info["input_shape"] = list(sample.shape)
            elif isinstance(sample, dict):
                info["keys"] = list(sample.keys())
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        info[f"{key}_shape"] = list(value.shape)
        except Exception as e:
            logger.warning(f"Could not get sample from dataset: {e}")
        
        return info
    
    def optimize_dataloader_config(
        self,
        dataset_size: int,
        batch_size: int,
        available_memory_gb: Optional[float] = None
    ) -> DataLoaderConfig:
        """
        Optimizar configuración de DataLoader basado en recursos.
        
        Args:
            dataset_size: Tamaño del dataset
            batch_size: Batch size deseado
            available_memory_gb: Memoria disponible en GB (None = auto-detect)
        
        Returns:
            DataLoaderConfig optimizado
        """
        config = DataLoaderConfig(batch_size=batch_size)
        
        # Optimize num_workers
        if torch.cuda.is_available():
            # Use more workers for GPU training
            config.num_workers = min(8, torch.cuda.device_count() * 2)
        else:
            # Use fewer workers for CPU
            config.num_workers = min(4, max(1, torch.get_num_threads() // 2))
        
        # Enable pin_memory if CUDA available
        config.pin_memory = torch.cuda.is_available()
        
        # Enable persistent_workers for efficiency
        config.persistent_workers = config.num_workers > 0
        
        # Drop last batch if dataset is large
        config.drop_last = dataset_size > 1000
        
        logger.info(f"Optimized DataLoader config: {config}")
        return config




