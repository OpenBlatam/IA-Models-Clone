"""
Dataset Manager - Gestor de datasets
=====================================
Gestión eficiente de datasets con caching y preprocessing
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pickle
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuración de dataset"""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2


class PrototypeDataset(Dataset):
    """Dataset para prototipos"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None
    ):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        if self.tokenizer:
            # Tokenizar texto si hay tokenizer
            if "text" in item:
                item["input_ids"] = self.tokenizer(
                    item["text"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
        
        return item


class DatasetManager:
    """Gestor de datasets"""
    
    def __init__(self, cache_dir: str = "./storage/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, Dataset] = {}
        self.dataloaders: Dict[str, DataLoader] = {}
    
    def create_dataset(
        self,
        dataset_id: str,
        data: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None
    ) -> Dataset:
        """Crea un dataset"""
        dataset = PrototypeDataset(data, transform, tokenizer)
        self.datasets[dataset_id] = dataset
        logger.info(f"Created dataset: {dataset_id} with {len(data)} samples")
        return dataset
    
    def create_dataloader(
        self,
        dataset_id: str,
        config: DatasetConfig,
        sampler: Optional[Any] = None
    ) -> DataLoader:
        """Crea un DataLoader"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset = self.datasets[dataset_id]
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if sampler is None else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor,
            sampler=sampler
        )
        
        self.dataloaders[dataset_id] = dataloader
        logger.info(f"Created DataLoader: {dataset_id}")
        return dataloader
    
    def save_dataset(self, dataset_id: str, filepath: Optional[str] = None):
        """Guarda dataset en cache"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        filepath = filepath or self.cache_dir / f"{dataset_id}.pkl"
        dataset = self.datasets[dataset_id]
        
        with open(filepath, "wb") as f:
            pickle.dump(dataset.data, f)
        
        logger.info(f"Saved dataset {dataset_id} to {filepath}")
    
    def load_dataset(self, dataset_id: str, filepath: Optional[str] = None) -> Dataset:
        """Carga dataset desde cache"""
        filepath = filepath or self.cache_dir / f"{dataset_id}.pkl"
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        dataset = PrototypeDataset(data)
        self.datasets[dataset_id] = dataset
        logger.info(f"Loaded dataset {dataset_id} from {filepath}")
        return dataset
    
    def split_dataset(
        self,
        dataset_id: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, Dataset]:
        """Divide dataset en train/val/test"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset = self.datasets[dataset_id]
        total_size = len(dataset)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        splits = {
            f"{dataset_id}_train": train_data,
            f"{dataset_id}_val": val_data,
            f"{dataset_id}_test": test_data
        }
        
        for split_id, split_dataset in splits.items():
            self.datasets[split_id] = split_dataset
        
        logger.info(f"Split dataset {dataset_id}: train={train_size}, val={val_size}, test={test_size}")
        return splits




