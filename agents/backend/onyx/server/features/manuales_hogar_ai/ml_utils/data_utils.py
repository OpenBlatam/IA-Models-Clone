"""
Data Utils - Utilidades de Procesamiento de Datos
==================================================

Utilidades para procesamiento de datos y construcción de datasets.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Procesador de datos para pipelines de ML.
    """
    
    def __init__(self):
        self.transforms: List[Callable] = []
        self.target_transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable) -> None:
        """
        Agregar transformación a datos.
        
        Args:
            transform: Función de transformación
        """
        self.transforms.append(transform)
    
    def add_target_transform(self, transform: Callable) -> None:
        """
        Agregar transformación a targets.
        
        Args:
            transform: Función de transformación
        """
        self.target_transforms.append(transform)
    
    def process(self, data: Any, target: Optional[Any] = None) -> Tuple[Any, Optional[Any]]:
        """
        Procesar datos.
        
        Args:
            data: Datos a procesar
            target: Target opcional
            
        Returns:
            Tupla (datos procesados, target procesado)
        """
        for transform in self.transforms:
            data = transform(data)
        
        if target is not None:
            for transform in self.target_transforms:
                target = transform(target)
        
        return data, target


class DatasetBuilder:
    """
    Builder para crear datasets PyTorch.
    """
    
    @staticmethod
    def create_from_arrays(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> Dataset:
        """
        Crear dataset desde arrays numpy.
        
        Args:
            X: Features
            y: Targets (opcional)
            transform: Transformación de datos
            target_transform: Transformación de targets
            
        Returns:
            Dataset
        """
        class ArrayDataset(Dataset):
            def __init__(self, X, y, transform, target_transform):
                self.X = X
                self.y = y
                self.transform = transform
                self.target_transform = target_transform
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                x = self.X[idx]
                if self.transform:
                    x = self.transform(x)
                
                if self.y is not None:
                    y = self.y[idx]
                    if self.target_transform:
                        y = self.target_transform(y)
                    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
                else:
                    return torch.tensor(x, dtype=torch.float32)
        
        return ArrayDataset(X, y, transform, target_transform)
    
    @staticmethod
    def create_from_dicts(
        data: List[Dict[str, Any]],
        feature_keys: List[str],
        target_key: Optional[str] = None,
        transform: Optional[Callable] = None
    ) -> Dataset:
        """
        Crear dataset desde lista de diccionarios.
        
        Args:
            data: Lista de diccionarios
            feature_keys: Keys de features
            target_key: Key de target (opcional)
            transform: Transformación
            
        Returns:
            Dataset
        """
        class DictDataset(Dataset):
            def __init__(self, data, feature_keys, target_key, transform):
                self.data = data
                self.feature_keys = feature_keys
                self.target_key = target_key
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                features = [item[key] for key in self.feature_keys]
                features = np.array(features, dtype=np.float32)
                
                if self.transform:
                    features = self.transform(features)
                
                if self.target_key:
                    target = item[self.target_key]
                    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
                else:
                    return torch.tensor(features, dtype=torch.float32)
        
        return DictDataset(data, feature_keys, target_key, transform)


class DataLoaderBuilder:
    """
    Builder para crear DataLoaders con configuración.
    """
    
    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Crear DataLoader.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            shuffle: Si mezclar datos
            num_workers: Número de workers
            pin_memory: Si usar pin memory
            drop_last: Si descartar último batch incompleto
            
        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last
        )
    
    @staticmethod
    def split_dataset(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ) -> Tuple[Dataset, ...]:
        """
        Dividir dataset en train/val/test.
        
        Args:
            dataset: Dataset completo
            train_ratio: Proporción de entrenamiento
            val_ratio: Proporción de validación
            test_ratio: Proporción de test
            
        Returns:
            Tupla de datasets
        """
        total_size = len(dataset)
        
        if val_ratio is None and test_ratio is None:
            # Solo train/val
            train_size = int(train_ratio * total_size)
            val_size = total_size - train_size
            return random_split(dataset, [train_size, val_size])
        
        elif test_ratio is None:
            # Train/val
            train_size = int(train_ratio * total_size)
            val_size = total_size - train_size
            return random_split(dataset, [train_size, val_size])
        
        else:
            # Train/val/test
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            return random_split(dataset, [train_size, val_size, test_size])




