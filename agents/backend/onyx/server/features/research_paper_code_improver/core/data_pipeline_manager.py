"""
Data Pipeline Manager - Gestor de pipelines de datos para ML
=============================================================
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig:
    """Configuración de pipeline de datos"""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False


class MLDataset(Dataset):
    """Dataset base para ML"""
    
    def __init__(self, data: List[Dict[str, Any]], transform: Optional[Callable] = None):
        self.data = data
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class DataPipelineManager:
    """Gestor de pipelines de datos"""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.datasets: Dict[str, Dataset] = {}
        self.data_loaders: Dict[str, DataLoader] = {}
        self.transforms: Dict[str, Callable] = {}
    
    def create_dataset(
        self,
        name: str,
        data: List[Dict[str, Any]],
        transform: Optional[Callable] = None
    ) -> Dataset:
        """Crea un dataset"""
        dataset = MLDataset(data, transform)
        self.datasets[name] = dataset
        logger.info(f"Dataset {name} creado con {len(data)} muestras")
        return dataset
    
    def create_dataloader(
        self,
        dataset_name: str,
        dataloader_name: Optional[str] = None,
        config: Optional[DataPipelineConfig] = None
    ) -> DataLoader:
        """Crea un DataLoader"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} no encontrado")
        
        dataset = self.datasets[dataset_name]
        config = config or self.config
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and torch.cuda.is_available(),
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False
        )
        
        name = dataloader_name or dataset_name
        self.data_loaders[name] = dataloader
        logger.info(f"DataLoader {name} creado")
        return dataloader
    
    def add_transform(self, name: str, transform: Callable):
        """Agrega una transformación"""
        self.transforms[name] = transform
    
    def apply_transform(self, name: str, data: Any) -> Any:
        """Aplica una transformación"""
        if name not in self.transforms:
            raise ValueError(f"Transformación {name} no encontrada")
        return self.transforms[name](data)
    
    def split_dataset(
        self,
        dataset_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, Dataset]:
        """Divide un dataset en train/val/test"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} no encontrado")
        
        dataset = self.datasets[dataset_name]
        total_size = len(dataset)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        splits = {
            f"{dataset_name}_train": train_dataset,
            f"{dataset_name}_val": val_dataset,
            f"{dataset_name}_test": test_dataset
        }
        
        for name, split_dataset in splits.items():
            self.datasets[name] = split_dataset
        
        logger.info(f"Dataset {dataset_name} dividido: train={train_size}, val={val_size}, test={test_size}")
        return splits
    
    def get_dataloader(self, name: str) -> Optional[DataLoader]:
        """Obtiene un DataLoader"""
        return self.data_loaders.get(name)
    
    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Obtiene un dataset"""
        return self.datasets.get(name)




